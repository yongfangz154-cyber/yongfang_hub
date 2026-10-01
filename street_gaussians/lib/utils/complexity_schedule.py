"""Complexity schedule.

score_affects_selection=false: fixed stride, scores are telemetry only.
score_affects_selection=true (v2.2): score gates actor Adam only
(update_exclude). Every candidate obj_* stays in the raster. Densify and the
region mask follow the fixed stride rule and do not read the score.
Geometry score stays 0.
"""

import math
import time

import numpy as np
import torch

from lib.utils.box_utils import bbox_to_corner3d, get_bound_2d_mask
from lib.utils.general_utils import quaternion_to_matrix_numpy


def geometry_score(name):
    """v2 hook. v1 always returns 0 and must not read depth or point count."""
    del name
    return 0.0


def assert_model_names_not_prefixes(names):
    names = list(names)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if a.startswith(b) or b.startswith(a):
                raise AssertionError(
                    f"model names must not be prefixes of each other: {a!r} vs {b!r}"
                )


class ComplexityState:
    def __init__(self, gaussians, schedule_cfg):
        self.gaussians = gaussians
        self.stride = max(int(schedule_cfg.get('bkgd_update_stride', 4)), 1)
        self.force_full_every = int(schedule_cfg.get('force_full_every', 8))
        self.actor_budget_ratio = float(schedule_cfg.get('actor_budget_ratio', 1.0))
        self.w_m = float(schedule_cfg.get('w_m', 1.0))
        self.w_a = float(schedule_cfg.get('w_a', 1.0))
        self.w_g = float(schedule_cfg.get('w_g', 0.0))
        self.beta_m = float(schedule_cfg.get('motion_ema_beta', 0.1))
        self.beta_a = float(schedule_cfg.get('appearance_ema_beta', 0.1))
        self.motion_norm_mps = float(schedule_cfg.get('motion_norm_mps', 10.0))
        self.a_init = float(schedule_cfg.get('a_init', 1.0))
        assert_model_names_not_prefixes(list(gaussians.model_name_id.keys()))

        self.M = {}
        self.A = {}
        self.selected_counts = {}
        names = []
        if gaussians.include_background:
            names.append('background')
        names.extend(list(gaussians.obj_list))
        for name in names:
            self.M[name] = 0.0
            self.A[name] = self.a_init
            self.selected_counts[name] = 0

        self.bkgd_credit = 0.0
        self.n_steps = 0
        self.n_forced_full = 0
        self.n_empty_supervision = 0
        self.n_bkgd_selected = 0
        self.score_s = 0.0
        self.backward_s = 0.0
        self.last_scores = {}
        self.last_n_candidates = 0
        self.last_n_selected_actors = 0
        self.last_n_raster_actors = 0
        self.last_n_adam_actors = 0
        self.n_raster_actor_mismatch = 0
        self.last_dropped = []
        self.nonforce_selected_actors = 0
        self.nonforce_candidate_actors = 0
        self.score_affects_selection = bool(schedule_cfg.get('score_affects_selection', False))

    def eligible(self, camera):
        frame = camera.meta['frame']
        names = []
        if self.gaussians.include_background:
            names.append('background')
        if self.gaussians.include_obj:
            for name in self.gaussians.obj_list:
                model = getattr(self.gaussians, name)
                if model.start_frame <= frame <= model.end_frame:
                    names.append(name)
        return names

    def _ema(self, old, raw, beta):
        return (1.0 - beta) * old + beta * raw

    def _track_key(self, track_id):
        pose = self.gaussians.actor_pose
        if pose is None:
            return None
        if track_id in pose.obj_info:
            return track_id
        try:
            as_int = int(track_id)
        except (TypeError, ValueError):
            return None
        if as_int in pose.obj_info:
            return as_int
        return None

    def _update_motion(self, camera, eligible):
        pose = self.gaussians.actor_pose
        if pose is None:
            return
        timestamp = camera.meta['timestamp']
        with torch.no_grad():
            for name in eligible:
                if not name.startswith('obj_'):
                    continue
                model = getattr(self.gaussians, name)
                key = self._track_key(model.track_id)
                if key is None:
                    continue
                track_idx = pose.obj_info[key]['track_idx']
                if track_idx.shape[0] < 2:
                    continue
                try:
                    ind1, ind2 = pose.find_closest_indices(key, timestamp)
                except AssertionError:
                    continue
                frame1, col1 = ind1[0], ind1[1]
                frame2, col2 = ind2[0], ind2[1]
                t1 = pose.timestamps[frame1.detach().cpu() if torch.is_tensor(frame1) else frame1]
                t2 = pose.timestamps[frame2.detach().cpu() if torch.is_tensor(frame2) else frame2]
                dt = abs(float(t2) - float(t1))
                if dt < 1e-3:
                    continue
                trans1 = pose.input_trans[frame1, col1]
                trans2 = pose.input_trans[frame2, col2]
                speed = float(torch.norm(trans2 - trans1).detach().cpu()) / dt
                norm = self.motion_norm_mps if self.motion_norm_mps > 0 else 1.0
                raw = min(speed / norm, 1.0)
                self.M[name] = self._ema(self.M[name], raw, self.beta_m)

    def _score(self, name):
        return (
            self.w_m * self.M.get(name, 0.0)
            + self.w_a * self.A.get(name, self.a_init)
            + self.w_g * geometry_score(name)
        )

    def _cost(self, name):
        return int(getattr(self.gaussians, name).get_xyz.shape[0])

    def _track_id(self, name):
        track_id = getattr(self.gaussians, name).track_id
        try:
            return int(track_id)
        except (TypeError, ValueError):
            return 10 ** 9

    def _keep_count(self, n_candidates):
        if n_candidates <= 0:
            return 0
        if self.actor_budget_ratio >= 1.0 - 1e-12:
            return n_candidates
        prod = self.actor_budget_ratio * n_candidates
        if abs(prod - round(prod)) < 1e-8:
            k = int(round(prod))
        else:
            k = int(math.ceil(prod))
        return min(n_candidates, max(1, k))

    def _select_actors(self, actors, scores, force_actors):
        """High score first. Equal scores keep the smaller track_id."""
        ordered = sorted(
            actors,
            key=lambda name: (-scores.get(name, 0.0), self._track_id(name), name),
        )
        if force_actors:
            return ordered, []
        k = self._keep_count(len(ordered))
        chosen = ordered[:k]
        dropped = ordered[k:]
        return chosen, dropped

    def decide(self, camera, iteration):
        """§5.1. Returns selected, raster_exclude, update_exclude, forced."""
        t0 = time.perf_counter()
        eligible = self.eligible(camera)
        self._update_motion(camera, eligible)
        scores = {name: self._score(name) for name in eligible}
        self.last_scores = scores

        # score_affects_selection=false: fixed stride. Scores do not change the set.
        if not self.score_affects_selection:
            skip_bkgd = self.stride > 1 and (iteration % self.stride != 0)
            selected = [name for name in eligible if name.startswith('obj_')]
            if 'background' in eligible and not skip_bkgd:
                selected.append('background')
            exclude = [name for name in eligible if name not in selected]
            densify_exclude = list(exclude)
            if skip_bkgd:
                if 'background' not in densify_exclude:
                    densify_exclude.append('background')
                for name in self.gaussians.obj_list:
                    if name not in densify_exclude:
                        densify_exclude.append(name)
            forced = False
            self.n_steps += 1
            for name in selected:
                self.selected_counts[name] = self.selected_counts.get(name, 0) + 1
            if 'background' in selected:
                self.n_bkgd_selected += 1
            actors_now = [name for name in eligible if name.startswith('obj_')]
            self.last_n_candidates = len(actors_now)
            self.last_n_selected_actors = len(actors_now)
            self.last_n_raster_actors = len(actors_now)
            self.last_n_adam_actors = len(actors_now)
            self.last_dropped = []
            self.score_s += time.perf_counter() - t0
            return {
                'eligible': eligible,
                'selected': selected,
                'raster_exclude': list(exclude),
                'update_exclude': list(exclude),
                'densify_exclude': densify_exclude,
                'forced': forced,
                'scores': scores,
            }

        # v2: stride still owns background. Score only ranks actors.
        skip_bkgd = self.stride > 1 and (iteration % self.stride != 0)
        force_actors = self.force_full_every > 0 and (iteration % self.force_full_every == 0)
        actors = [name for name in eligible if name.startswith('obj_')]
        selected_actors, dropped = self._select_actors(actors, scores, force_actors)
        selected = list(selected_actors)
        if 'background' in eligible and not skip_bkgd:
            selected.append('background')
        if force_actors:
            self.n_forced_full += 1
        else:
            self.nonforce_candidate_actors += len(actors)
            self.nonforce_selected_actors += len(selected_actors)
        self.last_n_candidates = len(actors)
        self.last_n_selected_actors = len(selected_actors)
        self.last_dropped = dropped
        # v2.2: score gates Adam only. Raster keeps every candidate car.
        # Densify matches the fixed rule and does not read the actor score.
        update_exclude = [name for name in eligible if name not in selected]
        raster_exclude = ['background'] if skip_bkgd and 'background' in eligible else []
        if skip_bkgd:
            densify_exclude = ['background'] if 'background' in eligible else []
            for name in self.gaussians.obj_list:
                if name not in densify_exclude:
                    densify_exclude.append(name)
        else:
            densify_exclude = []
        self.last_n_raster_actors = sum(1 for name in actors if name not in raster_exclude)
        self.last_n_adam_actors = sum(1 for name in actors if name not in update_exclude)
        if self.last_n_raster_actors != len(actors):
            self.n_raster_actor_mismatch += 1
        self.n_steps += 1
        for name in selected:
            self.selected_counts[name] = self.selected_counts.get(name, 0) + 1
        if 'background' in selected:
            self.n_bkgd_selected += 1
        self.score_s += time.perf_counter() - t0
        return {
            'eligible': eligible,
            'selected': selected,
            'raster_exclude': list(raster_exclude),
            'update_exclude': list(update_exclude),
            'densify_exclude': densify_exclude,
            'forced': force_actors,
            'scores': scores,
        }

    def project_track_mask(self, camera, obj_name):
        """Per-track box via get_bound_2d_mask. None if this frame has no sample."""
        pose_mod = self.gaussians.actor_pose
        if pose_mod is None or not hasattr(camera, 'extrinsic'):
            return None
        model = getattr(self.gaussians, obj_name)
        key = self._track_key(model.track_id)
        if key is None:
            return None
        track_idx = pose_mod.obj_info[key]['track_idx']
        frame_idx = int(camera.meta['frame_idx'])
        rows = track_idx[track_idx[:, 0] == frame_idx]
        if rows.shape[0] == 0:
            return None
        frame_i = int(rows[0, 0])
        col_i = int(rows[0, 1])
        trans = pose_mod.input_trans[frame_i, col_i].detach().cpu().numpy()
        rot = pose_mod.input_rots[frame_i, col_i].detach().cpu().numpy()
        obj_pose = np.eye(4)
        obj_pose[:3, :3] = quaternion_to_matrix_numpy(rot)
        obj_pose[:3, 3] = trans
        length, width, height = model.obj_meta['length'], model.obj_meta['width'], model.obj_meta['height']
        bbox = np.array([
            [-length, -width, -height],
            [length, width, height],
        ]) * 0.5
        corners_local = bbox_to_corner3d(bbox)
        corners_local = np.concatenate([corners_local, np.ones_like(corners_local[..., :1])], axis=-1)
        corners_vehicle = corners_local @ obj_pose.T
        ext = camera.extrinsic.detach().cpu().numpy()
        K = camera.K.detach().cpu().numpy()
        H = int(camera.image_height)
        W = int(camera.image_width)
        mask = get_bound_2d_mask(
            corners_3d=corners_vehicle[..., :3],
            K=K,
            pose=np.linalg.inv(ext),
            H=H,
            W=W,
        )
        return mask.astype(bool)

    def update_appearance(self, camera, image, gt, eligible, selected):
        """§3.2. Hold A when the entity was not rendered or its box is empty."""
        boxes = {}
        for name in eligible:
            if name.startswith('obj_'):
                boxes[name] = self.project_track_mask(camera, name)
        with torch.no_grad():
            for name, box in boxes.items():
                if name not in selected or box is None or int(box.sum()) == 0:
                    continue
                raw = self._masked_l1(image, gt, box)
                if raw is None:
                    continue
                self.A[name] = self._ema(self.A[name], raw, self.beta_a)
            if 'background' in selected and 'background' in self.A:
                static = np.ones((int(camera.image_height), int(camera.image_width)), dtype=bool)
                for box in boxes.values():
                    if box is not None:
                        static = np.logical_and(static, np.logical_not(box))
                if int(static.sum()) > 0:
                    raw = self._masked_l1(image, gt, static)
                    if raw is not None:
                        self.A['background'] = self._ema(self.A['background'], raw, self.beta_a)
        return boxes

    def _masked_l1(self, image, gt, box_hw):
        region = torch.from_numpy(np.ascontiguousarray(box_hw)).to(device=image.device, dtype=torch.bool)
        if int(region.sum()) == 0:
            return None
        err = (image.detach() - gt.detach()).abs()[:, region]
        if err.numel() == 0:
            return None
        return float(err.mean())

    def selection_mask(self, camera, boxes, selected_actors, full_image_mask):
        """Unused. v2.1 built the loss from selected boxes only. train.py does not call this."""
        H = int(camera.image_height)
        W = int(camera.image_width)
        union = np.zeros((H, W), dtype=bool)
        for name in selected_actors:
            box = boxes.get(name)
            if box is not None:
                union = np.logical_or(union, box)
        region = torch.from_numpy(np.ascontiguousarray(union)).to(device=full_image_mask.device, dtype=torch.bool)
        if region.dim() == 2:
            region = region.unsqueeze(0)
        if full_image_mask.dim() == 2:
            full_image_mask = full_image_mask.unsqueeze(0)
        return torch.logical_and(full_image_mask[:1], region[:1])

    def inclusion_summary(self):
        denom = max(self.n_steps, 1)
        parts = []
        for name in sorted(self.selected_counts):
            rate = self.selected_counts[name] / denom
            parts.append(f"{name}:{rate:.3f}")
        bkgd_rate = self.n_bkgd_selected / denom
        return bkgd_rate, " ".join(parts)

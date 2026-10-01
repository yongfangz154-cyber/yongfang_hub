import os
import torch
from random import randint
from lib.utils.loss_utils import l1_loss, l2_loss, psnr, ssim
from lib.utils.img_utils import save_img_torch, visualize_depth_numpy
from lib.models.street_gaussian_renderer import StreetGaussianRenderer
from lib.models.street_gaussian_model import StreetGaussianModel
from lib.utils.general_utils import safe_state
from lib.utils.camera_utils import Camera
from lib.utils.cfg_utils import save_cfg
from lib.models.scene import Scene
from lib.datasets.dataset import Dataset
from lib.config import cfg
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from lib.utils.system_utils import searchForMaxIteration
from lib.utils.complexity_schedule import ComplexityState
import time
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def _schedule_cfg():
    return cfg.train.get('schedule', None)

def _skip_background_this_step(iteration):
    """True when this step is off the background stride. Stride<=1 never skips."""
    schedule_cfg = _schedule_cfg()
    if schedule_cfg is None:
        return False
    stride = int(schedule_cfg.get('bkgd_update_stride', 1))
    if stride <= 1:
        return False
    return iteration % stride != 0

def get_update_exclude_list(iteration):
    """Hook B: skip background Adam/densify unless iteration hits the stride.

    Default (schedule.enable=False) returns [] so optimizer and densify
    call sets match the original Street Gaussians training loop.
    """
    schedule_cfg = _schedule_cfg()
    if schedule_cfg is None or (not schedule_cfg.get('enable', False)):
        return []
    if _skip_background_this_step(iteration):
        return ['background']
    return []

def get_raster_exclude_list(iteration):
    """Hook A: omit background from the rasterize graph off-stride.

    Default (schedule.skip_bkgd_raster=False) returns [] so render()
    still concatenates background + actors every step.
    Actors, ActorPose, and the sky cubemap are never put on this list.
    """
    schedule_cfg = _schedule_cfg()
    if schedule_cfg is None or (not schedule_cfg.get('skip_bkgd_raster', False)):
        return []
    if _skip_background_this_step(iteration):
        return ['background']
    return []

def count_gaussians(gaussians: StreetGaussianModel):
    n_bkgd = int(gaussians.background.get_xyz.shape[0]) if gaussians.include_background else 0
    n_obj = 0
    if gaussians.include_obj:
        for name in gaussians.obj_list:
            n_obj += int(getattr(gaussians, name).get_xyz.shape[0])
    return n_bkgd, n_obj

def training():
    training_args = cfg.train
    optim_args = cfg.optim
    data_args = cfg.data

    start_iter = 0
    tb_writer = prepare_output_and_logger()
    dataset = Dataset()
    gaussians = StreetGaussianModel(dataset.scene_info.metadata)
    scene = Scene(gaussians=gaussians, dataset=dataset)

    gaussians.training_setup()
    try:
        if cfg.loaded_iter == -1:
            loaded_iter = searchForMaxIteration(cfg.trained_model_dir)
        else:
            loaded_iter = cfg.loaded_iter
        ckpt_path = os.path.join(cfg.trained_model_dir, f'iteration_{loaded_iter}.pth')
        state_dict = torch.load(ckpt_path)
        start_iter = state_dict['iter']
        print(f'Loading model from {ckpt_path}')
        gaussians.load_state_dict(state_dict)
    except:
        pass

    print(f'Starting from {start_iter}')
    save_cfg(cfg, cfg.model_path, epoch=start_iter)

    gaussians_renderer = StreetGaussianRenderer()

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0
    psnr_dict = {}
    progress_bar = tqdm(range(start_iter, training_args.iterations))
    start_iter += 1

    viewpoint_stack = None
    schedule_cfg = cfg.train.get('schedule', None)
    schedule_enable = bool(schedule_cfg.get('enable', False)) if schedule_cfg is not None else False
    schedule_skip_raster = bool(schedule_cfg.get('skip_bkgd_raster', False)) if schedule_cfg is not None else False
    schedule_mask_loss = bool(schedule_cfg.get('mask_loss_on_skip', False)) if schedule_cfg is not None else False
    schedule_freeze_actor_densify = bool(schedule_cfg.get('freeze_actor_densify_on_skip', False)) if schedule_cfg is not None else False
    schedule_complexity = bool(schedule_cfg.get('complexity_enable', False)) if schedule_cfg is not None else False
    schedule_log_interval = int(schedule_cfg.get('log_interval', 100)) if schedule_cfg is not None else 100
    schedule_log_time = bool(schedule_cfg.get('log_time', False)) if schedule_cfg is not None else False
    complexity_state = ComplexityState(gaussians, schedule_cfg) if schedule_complexity else None
    loop_wall_start = time.time()
    print(
        f"[schedule] enable={schedule_enable} skip_bkgd_raster={schedule_skip_raster} "
        f"mask_loss_on_skip={schedule_mask_loss} freeze_actor_densify_on_skip={schedule_freeze_actor_densify} "
        f"complexity_enable={schedule_complexity} "
        f"bkgd_update_stride={int(schedule_cfg.get('bkgd_update_stride', 1)) if schedule_cfg is not None else 1} "
        f"log_interval={schedule_log_interval} log_time={schedule_log_time}"
    )
    if schedule_complexity:
        print("[schedule] complexity_enable overrides enable/skip_bkgd_raster/mask_loss_on_skip/freeze_actor_densify_on_skip")
    for iteration in range(start_iter, training_args.iterations + 1):
    
        iter_start.record()
        if schedule_complexity:
            exclude_list = []
            raster_exclude = []
        else:
            exclude_list = get_update_exclude_list(iteration)
            raster_exclude = get_raster_exclude_list(iteration)
        update_bkgd = 'background' not in exclude_list
        raster_bkgd = 'background' not in raster_exclude
        forced_full = False
        score_affects_selection = False
        selected_names = []
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        
        viewpoint_cam: Camera = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        
        gt_image = viewpoint_cam.original_image
        mask = viewpoint_cam.guidance['mask'] if 'mask' in viewpoint_cam.guidance else torch.ones_like(gt_image[0:1]).bool()
        gt_image = gt_image.cuda(non_blocking=True) if not gt_image.is_cuda else gt_image
        mask = mask.cuda(non_blocking=True) if not mask.is_cuda else mask
        if 'lidar_depth' in viewpoint_cam.guidance:
            lidar_depth = viewpoint_cam.guidance['lidar_depth']
            lidar_depth = lidar_depth.cuda(non_blocking=True) if not lidar_depth.is_cuda else lidar_depth
        if 'sky_mask' in viewpoint_cam.guidance:
            sky_mask = viewpoint_cam.guidance['sky_mask']
            sky_mask = sky_mask.cuda(non_blocking=True) if not sky_mask.is_cuda else sky_mask
        obj_bound = None
        if 'obj_bound' in viewpoint_cam.guidance:
            obj_bound = viewpoint_cam.guidance['obj_bound']
            obj_bound = obj_bound.cuda(non_blocking=True) if not obj_bound.is_cuda else obj_bound

        if schedule_complexity:
            decision = complexity_state.decide(viewpoint_cam, iteration)
            selected_names = decision['selected']
            raster_exclude = decision['raster_exclude']
            exclude_list = decision['update_exclude']
            update_bkgd = 'background' not in exclude_list
            raster_bkgd = 'background' not in raster_exclude
            forced_full = decision['forced']
            score_affects_selection = complexity_state.score_affects_selection

        render_pkg = gaussians_renderer.render(
            viewpoint_cam, gaussians, exclude_list=raster_exclude)
        # Captured before the debug render_object() call, which reparses the graph.
        n_raster = int(gaussians.num_gaussians)
        image, acc, viewspace_point_tensor, visibility_filter, radii = render_pkg["rgb"], render_pkg['acc'], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        depth = render_pkg['depth'] # [1, H, W]

        scalar_dict = dict()
        # v2.2: score gates Adam only. Supervision matches the fixed rule:
        # background skip uses obj_bound; background in uses the full image and sky.
        apply_region_loss = ((not schedule_complexity) and schedule_mask_loss and (not raster_bkgd)) or (
            schedule_complexity and (not raster_bkgd))
        region_mask = None
        region_frac = 1.0
        skipped_image_loss = False
        if schedule_complexity:
            complexity_state.update_appearance(
                viewpoint_cam, image, gt_image, decision['eligible'], selected_names)
        if apply_region_loss:
            if obj_bound is None or obj_bound.shape[-2:] != mask.shape[-2:]:
                skipped_image_loss = True
                region_frac = 0.0
                if schedule_complexity:
                    complexity_state.n_empty_supervision += 1
                print(f"[ITER {iteration}] schedule mask: missing obj_bound, skip image/sky/lidar loss")
            else:
                ob = obj_bound
                if ob.dim() == 2:
                    ob = ob.unsqueeze(0)
                ob = ob[:1].bool()
                region_mask = torch.logical_and(mask, ob)
                region_frac = float(region_mask.detach().float().mean())
                if int(region_mask.sum()) == 0:
                    skipped_image_loss = True
                    if schedule_complexity:
                        complexity_state.n_empty_supervision += 1
                    print(f"[ITER {iteration}] schedule mask: empty obj_bound, skip image/sky/lidar loss")

        loss = image.new_zeros(())
        if apply_region_loss and skipped_image_loss:
            scalar_dict['l1_loss'] = 0.0
        else:
            rgb_mask = region_mask if apply_region_loss else mask
            Ll1 = l1_loss(image, gt_image, rgb_mask)
            scalar_dict['l1_loss'] = Ll1.item()
            loss = (1.0 - optim_args.lambda_dssim) * optim_args.lambda_l1 * Ll1 + optim_args.lambda_dssim * (1.0 - ssim(image, gt_image, mask=rgb_mask))
    
        # sky loss. Raster-skip steps drop it: without background, full-image
        # sky loss pushes actors/sky to explain the road and buildings.
        if (not apply_region_loss) and optim_args.lambda_sky > 0 and gaussians.include_sky and sky_mask is not None:
            acc = torch.clamp(acc, min=1e-6, max=1.-1e-6)
            sky_loss = torch.where(sky_mask, -torch.log(1 - acc), -torch.log(acc)).mean()
            if len(optim_args.lambda_sky_scale) > 0:
                sky_loss *= optim_args.lambda_sky_scale[viewpoint_cam.meta['cam']]
            scalar_dict['sky_loss'] = sky_loss.item()
            loss += optim_args.lambda_sky * sky_loss
        
        # obj_acc uses a box mask. Complexity steps must not feed guidance['obj_bound']
        # (the union of every track) when some actors are outside the selected set.
        run_obj_acc = (
            optim_args.lambda_reg > 0
            and gaussians.include_obj
            and iteration >= optim_args.densify_until_iter
        )
        if run_obj_acc and obj_bound is not None:
            pos_bound = obj_bound
            if pos_bound is not None:
                render_pkg_obj = gaussians_renderer.render_object(viewpoint_cam, gaussians, parse_camera_again=False)
                image_obj, acc_obj = render_pkg_obj["rgb"], render_pkg_obj['acc']
                acc_obj = torch.clamp(acc_obj, min=1e-6, max=1.-1e-6)
                obj_acc_loss = torch.where(pos_bound,
                    -(acc_obj * torch.log(acc_obj) +  (1. - acc_obj) * torch.log(1. - acc_obj)),
                    -torch.log(1. - acc_obj)).mean()
                scalar_dict['obj_acc_loss'] = obj_acc_loss.item()
                loss += optim_args.lambda_reg * obj_acc_loss

        # lidar depth loss
        if optim_args.lambda_depth_lidar > 0 and lidar_depth is not None:            
            depth_mask = torch.logical_and((lidar_depth > 0.), mask)
            if apply_region_loss and region_mask is not None:
                depth_mask = torch.logical_and(depth_mask, region_mask)
            elif apply_region_loss:
                depth_mask = torch.zeros_like(depth_mask)
            n_depth = int(depth_mask.sum())
            if n_depth > 0:
                expected_depth = depth / (render_pkg['acc'] + 1e-10)
                depth_error = torch.abs((expected_depth[depth_mask] - lidar_depth[depth_mask]))
                k = max(1, int(0.95 * depth_error.size(0)))
                depth_error, _ = torch.topk(depth_error, k, largest=False)
                lidar_depth_loss = depth_error.mean()
                scalar_dict['lidar_depth_loss'] = lidar_depth_loss
                loss += optim_args.lambda_depth_lidar * lidar_depth_loss
                    
        # color correction loss
        if optim_args.lambda_color_correction > 0 and gaussians.use_color_correction:
            color_correction_reg_loss = gaussians.color_correction.regularization_loss(viewpoint_cam)
            scalar_dict['color_correction_reg_loss'] = color_correction_reg_loss.item()
            loss += optim_args.lambda_color_correction * color_correction_reg_loss
                    
        scalar_dict['loss'] = loss.item()

        if loss.requires_grad:
            bwd_t0 = time.perf_counter()
            loss.backward()
            if schedule_complexity:
                complexity_state.backward_s += time.perf_counter() - bwd_t0
        iter_end.record()
                
        is_save_images = True
        if is_save_images and (iteration % 1000 == 0):
            # row0: gt_image, image, depth
            # row1: acc, image_obj, acc_obj
            depth_colored, _ = visualize_depth_numpy(depth.detach().cpu().numpy().squeeze(0))
            depth_colored = depth_colored[..., [2, 1, 0]] / 255.
            depth_colored = torch.from_numpy(depth_colored).permute(2, 0, 1).float().cuda()
            row0 = torch.cat([gt_image, image, depth_colored], dim=2)
            acc = acc.repeat(3, 1, 1)
            with torch.no_grad():
                render_pkg_obj = gaussians_renderer.render_object(viewpoint_cam, gaussians)
                image_obj, acc_obj = render_pkg_obj["rgb"], render_pkg_obj['acc']
            acc_obj = acc_obj.repeat(3, 1, 1)
            row1 = torch.cat([acc, image_obj, acc_obj], dim=2)
            image_to_show = torch.cat([row0, row1], dim=1)
            image_to_show = torch.clamp(image_to_show, 0.0, 1.0)
            os.makedirs(f"{cfg.model_path}/log_images", exist_ok = True)
            save_img_torch(image_to_show, f"{cfg.model_path}/log_images/{iteration}.jpg")
        
        with torch.no_grad():
            
            # Log
            tensor_dict = dict()

            if iteration % 10 == 0:                    
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                ema_psnr_for_log = 0.4 * psnr(image, gt_image, mask).mean().float() + 0.6 * ema_psnr_for_log
                progress_bar.set_postfix({"Exp": f"{cfg.task}-{cfg.exp_name}", 
                                          "Loss": f"{ema_loss_for_log:.{7}f},", 
                                          "PSNR": f"{ema_psnr_for_log:.{4}f}"})
            progress_bar.update(1)
            # if iteration == training_args.iterations:
            #     progress_bar.close()

            # Save ply
            if (iteration in training_args.save_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification. Complexity: Adam uses update_exclude; densify adds
            # every candidate actor when background was not selected.
            if schedule_complexity:
                densify_exclude = list(decision['densify_exclude'])
                freeze_actor_densify = any(name.startswith('obj_') for name in densify_exclude)
            else:
                freeze_actor_densify = schedule_freeze_actor_densify and (not raster_bkgd)
                densify_exclude = list(exclude_list)
                if freeze_actor_densify and gaussians.include_obj:
                    densify_exclude = list(set(densify_exclude) | set(gaussians.obj_list))
            if iteration < optim_args.densify_until_iter:
                gaussians.set_visibility(include_list=list(set(gaussians.model_name_id.keys()) - set(['sky'])))
                gaussians.set_max_radii2D(radii, visibility_filter, exclude_list=densify_exclude)
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, exclude_list=densify_exclude)
                
                prune_big_points = iteration > optim_args.opacity_reset_interval

                if iteration > optim_args.densify_from_iter:
                    if iteration % optim_args.densification_interval == 0:
                        scalars, tensors = gaussians.densify_and_prune(
                            max_grad=optim_args.densify_grad_threshold,
                            min_opacity=optim_args.min_opacity,
                            prune_big_points=prune_big_points,
                            exclude_list=densify_exclude,
                        )

                        if scalars is not None:
                            scalar_dict.update(scalars)
                        if tensors is not None:
                            tensor_dict.update(tensors)
                        
            # Reset opacity
            if iteration < optim_args.densify_until_iter:
                if iteration % optim_args.opacity_reset_interval == 0:
                    gaussians.reset_opacity()
                if data_args.white_background and iteration == optim_args.densify_from_iter:
                    gaussians.reset_opacity()

            schedule_active = schedule_enable or schedule_skip_raster or schedule_mask_loss or schedule_complexity
            if (
                iteration % schedule_log_interval == 0
                or (schedule_active and iteration % schedule_log_interval == 1)
                or (schedule_active and iteration <= 8)
            ):
                n_bkgd, n_obj = count_gaussians(gaussians)
                if schedule_complexity:
                    credit_now = complexity_state.bkgd_credit
                    print(
                        f"[ITER {iteration}] exclude_list={exclude_list} "
                        f"update_background={update_bkgd} raster_exclude={raster_exclude} "
                        f"raster_background={raster_bkgd} n_raster={n_raster} "
                        f"region_loss={apply_region_loss} region_frac={region_frac:.4f} "
                        f"skipped_image_loss={skipped_image_loss} freeze_actor_densify={freeze_actor_densify} "
                        f"selected={selected_names} densify_exclude={densify_exclude} forced_full={int(forced_full)} "
                        f"bkgd_credit={credit_now:.4f} "
                        f"actor_budget_ratio={complexity_state.actor_budget_ratio:.3f} "
                        f"n_candidates={complexity_state.last_n_candidates} "
                        f"n_selected_actors={complexity_state.last_n_selected_actors} "
                        f"n_raster_actors={complexity_state.last_n_raster_actors} "
                        f"n_adam_actors={complexity_state.last_n_adam_actors} "
                        f"dropped={complexity_state.last_dropped} "
                        f"score_affects_selection={score_affects_selection} "
                        f"n_bkgd={n_bkgd} n_obj={n_obj}"
                    )
                else:
                    print(
                        f"[ITER {iteration}] exclude_list={exclude_list} "
                        f"update_background={update_bkgd} raster_exclude={raster_exclude} "
                        f"raster_background={raster_bkgd} n_raster={n_raster} "
                        f"region_loss={apply_region_loss} region_frac={region_frac:.4f} "
                        f"skipped_image_loss={skipped_image_loss} freeze_actor_densify={freeze_actor_densify} "
                        f"n_bkgd={n_bkgd} n_obj={n_obj}"
                    )
                if schedule_complexity:
                    bkgd_rate, rates = complexity_state.inclusion_summary()
                    score_str = " ".join(
                        f"{k}:{complexity_state.last_scores.get(k, 0):.3f}"
                        for k in sorted(complexity_state.last_scores)
                    )
                    print(
                        f"[ITER {iteration}] inclusion bkgd_rate={bkgd_rate:.3f} {rates} "
                        f"scores {score_str}"
                    )
                if schedule_log_time:
                    torch.cuda.synchronize()
                    print(f"[ITER {iteration}] cuda_fwd_bwd_ms={iter_start.elapsed_time(iter_end):.2f}")
                if torch.is_tensor(loss) and torch.isnan(loss).any():
                    print(f"[ITER {iteration}] WARNING: loss is NaN")

            training_report(tb_writer, iteration, scalar_dict, tensor_dict, training_args.test_iterations, scene, gaussians_renderer)

            # Optimizer step
            if iteration < training_args.iterations:
                gaussians.update_optimizer(exclude_list=exclude_list)

            if (iteration in training_args.checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                state_dict = gaussians.save_state_dict(is_final=(iteration == training_args.iterations))
                state_dict['iter'] = iteration
                ckpt_path = os.path.join(cfg.trained_model_dir, f'iteration_{iteration}.pth')
                torch.save(state_dict, ckpt_path)

    progress_bar.close()
    wall_s = time.time() - loop_wall_start
    n_bkgd, n_obj = count_gaussians(gaussians)
    extra = ""
    if schedule_complexity:
        bkgd_rate, rates = complexity_state.inclusion_summary()
        extra = (
            f" complexity_enable=True n_forced_full={complexity_state.n_forced_full} "
            f"n_empty_supervision={complexity_state.n_empty_supervision} "
            f"bkgd_inclusion_rate={bkgd_rate:.4f} score_affects_selection={complexity_state.score_affects_selection} "
            f"actor_budget_ratio={complexity_state.actor_budget_ratio:.3f} "
            f"nonforce_actor_keep="
            f"{(complexity_state.nonforce_selected_actors / complexity_state.nonforce_candidate_actors) if complexity_state.nonforce_candidate_actors else 1.0:.4f} "
            f"n_raster_actor_mismatch={complexity_state.n_raster_actor_mismatch} "
            f"score_s={complexity_state.score_s:.1f} backward_s={complexity_state.backward_s:.1f} "
            f"inclusion {rates}"
        )
    print(
        f"[DONE] wall_s={wall_s:.1f} n_bkgd={n_bkgd} n_obj={n_obj} "
        f"schedule.enable={schedule_enable} schedule.skip_bkgd_raster={schedule_skip_raster} "
        f"schedule.mask_loss_on_skip={schedule_mask_loss} "
        f"schedule.freeze_actor_densify_on_skip={schedule_freeze_actor_densify}"
        f"{extra}"
    )


def prepare_output_and_logger():
    
    # if cfg.model_path == '':
    #     if os.getenv('OAR_JOB_ID'):
    #         unique_str = os.getenv('OAR_JOB_ID')
    #     else:
    #         unique_str = str(uuid.uuid4())
    #     cfg.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(cfg.model_path))

    os.makedirs(cfg.model_path, exist_ok=True)
    os.makedirs(cfg.trained_model_dir, exist_ok=True)
    os.makedirs(cfg.record_dir, exist_ok=True)
    if not cfg.resume:
        os.system('rm -rf {}/*'.format(cfg.record_dir))
        os.system('rm -rf {}/*'.format(cfg.trained_model_dir))

    with open(os.path.join(cfg.model_path, "cfg_args"), 'w') as cfg_log_f:
        viewer_arg = dict()
        viewer_arg['sh_degree'] = cfg.model.gaussian.sh_degree
        viewer_arg['white_background'] = cfg.data.white_background
        viewer_arg['source_path'] = cfg.source_path
        viewer_arg['model_path']= cfg.model_path
        cfg_log_f.write(str(Namespace(**viewer_arg)))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(cfg.record_dir)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, scalar_stats, tensor_stats, testing_iterations, scene: Scene, renderer: StreetGaussianRenderer):
    if tb_writer:
        try:
            for key, value in scalar_stats.items():
                tb_writer.add_scalar('train/' + key, value, iteration)
            for key, value in tensor_stats.items():
                tb_writer.add_histogram('train/' + key, value, iteration)
        except:
            print('Failed to write to tensorboard')
            
            
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test/test_view', 'cameras' : scene.getTestCameras()},
                              {'name': 'test/train_view', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderer.render(viewpoint, scene.gaussians)["rgb"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    
                    if hasattr(viewpoint, 'original_mask'):
                        mask = viewpoint.original_mask.cuda().bool()
                    else:
                        mask = torch.ones_like(gt_image[0]).bool()
                    l1_test += l1_loss(image, gt_image, mask).mean().double()
                    psnr_test += psnr(image, gt_image, mask).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("test/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('test/points_total', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    print("Optimizing " + cfg.model_path)

    # Initialize system state (RNG)
    safe_state(cfg.train.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(cfg.train.detect_anomaly)
    training()

    # All done
    print("\nTraining complete.")
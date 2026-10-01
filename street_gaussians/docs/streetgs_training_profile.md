# Street Gaussians 训练管线画像

本报告只摸底、不改训练算法。结论均指向本仓库真实文件/函数。本阶段未加计时探针；`train.py` 里已有未使用的 CUDA Event，可作为默认不影响数值的只读计时入口（见 §5）。

---

## 1. 基线身份

**这是纯 Street Gaussians（ZJU3DV 官方仓库），不是 4DGS / 其它 fork 的混体。**

| 证据 | 位置 |
|------|------|
| README 论文与 clone 地址 | `README.md`：ECCV 2024 *Street Gaussians*；`git clone https://github.com/zju3dv/street_gaussians.git` |
| remote | `origin` = `https://github.com/zju3dv/street_gaussians.git` |
| HEAD | `0924b0c` *Update README.md*（Yunzhi Yan, 2025-07-04） |
| 核心表示 | 背景静态 3DGS + 每 track 一个 actor 高斯 + 可选天空 cubemap；**没有** deformation MLP / HexPlane / 4D 网格 |
| 光栅化 | `submodules/diff-gaussian-rasterization`（Inria 3DGS 差分光栅，扩展了 depth/alpha/semantic） |
| 邻近搜索 | `submodules/simple-knn` |
| Waymo 读取 | `submodules/simple-waymo-open-dataset-reader` |

工作区有未提交改动（`lib/utils/cfg_utils.py`、`script/waymo/colmap_waymo_full.py`），属于路径/COLMAP 工程修补，**不改变** densify / loss / 调度语义。

本机已跑通的 example 入口：

```bash
python train.py --config configs/example/waymo_train_002.yaml
python train.py --config configs/example/waymo_train_031.yaml
python render.py --config configs/example/waymo_train_002.yaml mode trajectory
python render.py --config configs/example/waymo_train_031.yaml mode trajectory
```

批处理脚本：`script/train_waymo_example.sh`（对 `031`、`002` 循环上述 `train.py`）。

配置加载链：`lib/config/config.py` 默认值 → `lib/utils/cfg_utils.py:make_cfg` 合并 yaml（无 `parent_cfg` 时直接 merge）。关键字段：

- `data.type: Waymo` → `lib/datasets/dataset.py` 的 `readWaymoFullInfo`
- `data.source_path`: `./data/waymo/training/{002,031}`
- `model.nsg.include_bkgd/include_obj/include_sky: true`，`opt_track: true`
- `train.iterations: 50000`（example；`lib/config/config.py` 默认 30000）
- `data.split_train: 1` → **全部选中帧都当训练**，测试相机集合为空（见 §2.1）

论文实验配置在 `configs/experiments_waymo/`；本机基线实际用的是 `configs/example/`。

---

## 2. 训练一步（step）做什么

入口：`train.py:training()`。一步 = 随机抽 **1 张相机图** → 合成渲染 → 标量 loss → `backward` →（条件）densify/prune → 各子模型 `optimizer.step`。

### 2.1 如何采样：相机 / 时间戳 / view 数

**每步恰好 1 个 view，没有 batch。**

```73:77:train.py
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        
        viewpoint_cam: Camera = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
```

- 相机列表来自 `Scene.getTrainCameras()` → `Dataset.train_cameras[1.0]`（`lib/models/scene.py:49-50`，`lib/datasets/dataset.py:56-58`）。
- 训练开始时 `cfg.data.shuffle` 默认 True，对 `train_cameras` 做一次 shuffle（`dataset.py:52-54`）。
- 栈抽空后整表复制再随机 pop，等价于无放回扫完一轮再洗。
- **没有** `cfg.train.importance_sampling` 的实现（该开关在 `lib/config/config.py:45` 为 `False`，全库无引用）。

一张 `Camera` 上的时间/传感器元数据在 `lib/datasets/waymo_full_readers.py:109-115` 写入 `metadata`：

| 字段 | 含义 |
|------|------|
| `frame` | Waymo 原始帧号 |
| `frame_idx` | 相对 `selected_frames[0]` 的索引 |
| `cam` | 0/1/2 = FRONT / FRONT_LEFT / FRONT_RIGHT（example 的 `data.cameras`） |
| `timestamp` | 该相机曝光时间（减过 offset） |
| `ego_pose` | 该相机时刻的自车位姿 |
| `is_val` | 是否验证帧 |

`selected_frames`：002 为 `[98, 198]`，031 为 `[0, 198]`。`split_train: 1` 时 `get_val_frames`（`lib/utils/data_utils.py:36-47`）把所有帧放进 train，`val_frames` 为空。因此训练循环看不到测试相机。

### 2.2 前向：哪些高斯 / actor 参与 rasterize

调用：`StreetGaussianRenderer.render`（`lib/models/street_gaussian_renderer.py:89-119`）→ `parse_camera` → `render_kernel`。

1. `set_visibility(include_list)` 默认包含 `model_name_id` 全部键：`background` + 每个 `obj_{track_id:03d}`（`street_gaussian_renderer.py:99-103`）。天空高斯模型不在此列表里（天空是 cubemap）。
2. `StreetGaussianModel.parse_camera`（`street_gaussian_model.py:219-285`）按 **当前相机的 `frame`** 建 scene graph：
   - 背景：若 visible，整块 `background.get_xyz` 入图。
   - 物体：仅当 `start_frame <= frame <= end_frame` 且 `get_visibility(obj_*)` 为真，才加入 `graph_obj_list`。
   - 每个入图 actor 的局部 xyz/rot 用 `ActorPose.get_tracking_{translation,rotation}(track_id, camera)` 变到世界系，再乘 `ego_pose`。
3. `get_xyz` / `get_opacity` / `get_scaling` / `get_rotation` / `get_features` 把 **当前 graph 上的高斯沿 dim0 拼接** 成一次 rasterize（`street_gaussian_model.py:287-449`）。
4. `render_kernel`（`street_gaussian_renderer.py:122-280`）调用 `make_rasterizer`（`lib/utils/camera_utils.py:194-227`）→ `diff_gaussian_rasterization.GaussianRasterizer`。`prefiltered=False`，**无 tile skip / 预过滤开关被打开**。
5. 天空：`include_sky` 时 `SkyCubeMap.forward`（`lib/models/sky_cubemap.py:77+`），用 nvdiffrast 查 cubemap，再  
   `rgb += sky_color * (1 - acc)`。传给 cubemap 的 `acc` 是 **`.detach()`** 的（`street_gaussian_renderer.py:109`），只影响 cubemap 采样 mask，合成权重 `(1-acc)` 仍可回传到高斯。

Actor 外观是时间相关的：**`GaussianModelActor.get_features_fourier(frame)`**（`lib/models/gaussian_model_actor.py:71-80`）对 DC 做 IDFT，`fourier_dim=5`（example yaml）。背景 SH **不**走 Fourier。几何上 actor 仍是 **规范系 3D 高斯 + 刚体 track 位姿**；`deformable=True` 仅对 pedestrian 关掉对称翻转（`waymo_utils.py:246-249`），**没有变形网络**。

### 2.3 Loss 与反传对象

全部在 `train.py:100-142`，一次 `loss.backward()`（`train.py:142`），图连通的参数都会收到梯度。

| 项 | 条件（example 002/031） | 作用对象 |
|----|-------------------------|----------|
| L1 + DSSIM | 始终；mask 缺省为全图 1（`guidance` 无 `mask` 键） | 合成 `rgb` → 入图的 bkgd+actor 高斯 +（经 1-acc）天空 cubemap |
| sky CE | `lambda_sky=0.05` 且 `include_sky` 且有 `sky_mask` | 对 `acc`：天空像素推 `acc→0`，非天空推 `acc→1` |
| obj acc 正则 | `lambda_reg=0.1` 且 **`iteration >= densify_until_iter`（25000）** | **额外一次** `render_object`（只 rasterize actors，`parse_camera_again=False`）；`obj_bound` 内鼓励 acc 别 0/1 极端，外鼓励 `acc→0` |
| LiDAR 深度 | `lambda_depth_lidar=0.1` | `expected_depth = depth / (acc+eps)` vs `guidance['lidar_depth']`，取 95% 小误差 |
| color correction | 默认 `use_color_correction=False` | 不启用 |
| semantic / mono depth / mono normal | 配置里有 `lambda_*`，**`train.py` 未使用** | 本基线无 |

`StreetGaussianModel.get_box_reg_loss`（`street_gaussian_model.py:588-595`）**从未被 `train.py` 调用**。yaml 里的 `prune_box_interval` 同样无 Python 引用。

反传后 densify 用的是 rasterizer 返回的 `viewspace_points.grad`（2D 均值梯度）和 `visibility_filter = (radii > 0)`。

### 2.4 Densify / prune：何时、用什么分数

`train.py:186-208`：仅当 `iteration < densify_until_iter`（example=25000）。

每步（未到 until）：

1. `set_visibility` 去掉 sky 键（sky 本就不是高斯）。
2. `set_max_radii2D` / `add_densification_stats` 按 `graph_gaussian_range` **切回各子模型**（`street_gaussian_model.py:551-571`）。
3. 梯度累加两列：`[:,0]` = `||grad_xy||`（屏幕空间），`[:,1]` = `||grad_z||`（AbsGS 用）。
4. `iteration > densify_from_iter(500)` 且 `% densification_interval(100)==0` 时调用 `gaussians.densify_and_prune`。
5. `% opacity_reset_interval(3000)==0` 时 `reset_opacity`。
6. `prune_big_points` 在 `iteration > opacity_reset_interval` 后为真。

子模型策略不同：

- **背景** `GaussianModelBkgd.densify_and_prune`（`gaussian_model_bkgd.py:74-114`）：阈值 `densify_grad_threshold_bkgd=0.0006`；example 里 `densify_grad_abs_bkgd: True` → 用 **深度方向梯度**；extent = `scene_radius`；可按世界尺度 prune。
- **物体** `GaussianModelActor.densify_and_prune`（`gaussian_model_actor.py:204-261`）：阈值 `densify_grad_threshold_obj=0.0002`；`densify_grad_abs_obj: False` → 用 **xy 梯度**；extent = 物体 bbox；还可把采样点落在 tracking box 外的高斯 prune 掉。行人/`random_initialization` 强制用 xy 梯度。

这是 **3DGS 式梯度 densify**，按高斯点、不是按像素难度、也不是按深度 LOD。

### 2.5 背景 vs 动态物体：是否分开优化

**是。每个实体自己的 Adam 与参数组。**

| 实体 | 类 | 可学习量 |
|------|----|----------|
| 背景 | `GaussianModelBkgd` | `_xyz, _features_dc/rest, _opacity, _scaling, _rotation, _semantic`（`gaussian_model.py:292-300`） |
| 每个 `obj_{id}` | `GaussianModelActor` | 同上，但 DC 是 Fourier 系数；规范系 xyz；`flip_prob` 训练时随机镜像（002=0.5，031=0） |
| 轨迹 | `ActorPose` | `opt_trans`（每帧每槽 Δxyz）+ `opt_rots`（每帧每槽 Δyaw）；`opt_track=True`（`actor_pose.py:20-26, 46-66`） |
| 天空 | `SkyCubeMap` | `sky_cube_map` `[6, 1024, 1024, 3]` |
| 颜色校正 / 相机 pose 校正 | 默认关 | `cfg.model.use_color_correction/use_pose_correction = False` |

`StreetGaussianModel.update_optimizer`（`street_gaussian_model.py:532-549`）**依次**对每个子模型、actor_pose、sky 做 `step + zero_grad`。已有 `exclude_list` 参数，训练循环目前传空。

SH 阶从 0 每 1000 step `oneupSHdegree`（`train.py:70-71`），bkgd 与 obj 一起升。

### 2.6 显存大户（定性）

不必精确 profiler，从代码结构可排：

1. **一次 rasterize 的拼接高斯**：031 初始化日志背景约 67 万点 + 若干 actor；`geomBuffer/binningBuffer/imgBuffer` 在 backward 里 `save_for_backward`（`submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py:101`）。
2. **全分辨率图**（example 约 1920×1280）的 rgb/acc/depth 与 GT、`sky_mask`、`lidar_depth`、`obj_bound`。
3. **天空 cubemap** 6×1024×1024×3 + nvdiffrast。
4. **densify 之后** 点数继续涨；`screenspace_points` 为 `[N,3]` 且 `retain_grad`（`street_gaussian_renderer.py:164-169`）。
5. `iteration >= 25000` 后每步多一次 `render_object`（`train.py:114-116`）。
6. 每 1000 步 `log_images` 再渲一次 object（`train.py:146-163`）。
7. `cfg.to_cuda` 默认 False（`config.py:21`）；example yaml 未打开。打开则会把全部训练 `Camera` 纹理常驻 GPU（`scene.py:25-28`）。

---

## 3. 仓库里已经存在的「分配」（基线，不是新贡献）

| 机制 | 有/无 | 位置 |
|------|-------|------|
| 梯度/误差引导 densify | **有**（屏幕空间 xyz 梯度，不是像素误差图） | `train.py:186-200`；`street_gaussian_model.py:562-571`；bkgd/obj 各自 `densify_and_prune` |
| 按深度或距离的 LOD（渲染时减点数/减 SH） | **无** | 光栅 `prefiltered=False`；无层级高斯。背景 prune 只用世界尺度阈值，不是 LOD |
| 按 actor 分别优化 | **有** | 每 track 一个 `GaussianModelActor` + 独立 Adam；`ActorPose` 独立优化 track |
| 按帧剔除不可见 actor（不进 rasterize） | **有** | `parse_camera` 的 `start_frame/end_frame` 窗口 |
| tile skip / 稀疏渲染 | **无** | 标准 tile rasterizer，无 skip 开关 |
| 混合精度 AMP | **无** | 无 `autocast` / `GradScaler` |
| 其它跳过 backward / 降算力 | **弱、局部** | 见下 |

其它已有、但不要当成新调度的逻辑：

- `visibility_filter = radii > 0`：视锥外高斯不进 densify 统计（3DGS 原版）。
- 天空 `acc.detach()`：cubemap 采样 mask 不回传到高斯 acc（`street_gaussian_renderer.py:109`）。
- `lambda_reg` 的 object 二次渲染只在 densify 结束后开启。
- `exclude_list` 已在 `update_optimizer` / `densify_and_prune` / `training_setup` 预留，**训练循环未用**。
- `cfg.train.importance_sampling`、`optim.prune_box_interval`、`get_box_reg_loss`、`lambda_semantic`：**配置或函数存在，训练路径未接上**。

---

## 4. 现成信号（训练时能否直接读）

数据根：`data/waymo/training/002` 与 `031`。下表以这两个 example 目录 + 训练循环为准。

| 信号 | 磁盘 | 训练时 Camera 上 | 说明 |
|------|------|------------------|------|
| 深度 / LiDAR | **有** `lidar_depth/*.npy` | **有** `guidance['lidar_depth']` | `waymo_full_readers.py:43-45,133-142`；`camera_utils.py:loadguidance`；loss 在 `train.py:124-132` |
| sky mask | **有** `sky_mask/` | **有** `guidance['sky_mask']` | 仅 `mode=='train'` 加载；`train.py` sky loss + `SkyCubeMap` |
| 动态 mask PNG | **有** `dynamic_mask/` | **不直接读 PNG** | 读取代码被注释（`waymo_full_readers.py:127-131`）。实际 `guidance['obj_bound']` 是 **3D box 投影并集**（`waymo_utils.py:407-437`） |
| 3D box / tracking id | **有** `track/track_info.txt` + `track_camera_vis.json` | **有** | `get_obj_pose_tracking`；`metadata['obj_meta']` 含 `track_id, class, length/width/height, start/end_frame`；逐步可通过 `pc.graph_obj_list` / `ActorPose` 拿到当前帧活跃 id |
| 时间戳 | **有** `timestamps.json` | **有** `camera.meta['timestamp']` | 相机曝光时间与 frame 时间戳都有 |
| 光流 | **无** | **无** | 数据目录与 `lib/` 均无 flow |
| 语义分割图 | **无**（002/031 无 sem 目录） | **无** | `cfg.data.use_semantic=False`；`lambda_semantic` 未接入 `train.py`。仅 KITTI 脚本 `script/kitti/generate_semantic_mask.py` |
| 像素级实例 ID | **无** | **无** | 只有 box 投影的二值 `obj_bound`（多车并集）+ track id 的 3D 实体，没有 per-pixel instance map |
| 单目深度 / 法向 | 脚本有、数据无 | **无** | `script/waymo/generate_mono_{depth,normal}.py`；`cfg.data.use_mono_depth/normal=False`；002/031 目录无对应文件夹 |
| COLMAP 点 | 训练时现场跑 | 初始化 PCD，不是逐步信号 | `data.use_colmap: true` |

类别标签（vehicle/pedestrian/cyclist）在 `obj_meta['class']`，训练时可按 actor 读取；**不是**图像语义分割。

---

## 5. 调度最小插入点（只建议，不改代码）

现有拼接-一次-rasterize 结构里，**按实体打断**比按 tile 容易得多：actor 已经是独立 `nn.Module` + 独立 optimizer，`parse_camera` 已经按帧做子集。

### 钩子 A（首选）：`StreetGaussianModel.parse_camera` + `set_visibility` / `render(..., exclude_list=)`

- 文件：`lib/models/street_gaussian_model.py:219-251`，`lib/models/street_gaussian_renderer.py:89-103`
- 能打断：某 step 是否把某个 `obj_*` 或 `background` 送进 rasterize（从而 **forward+backward 一起省**）。
- 实体粒度：天然是 track / 背景块（背景目前是一整块，要切空间块需再分模型或 mask）。
- 风险：不进图的高斯该 step 无梯度、也无 densify 统计；长期跳过背景会与现有 25000 步前 densify 强耦合；天空合成依赖前景 `acc`。

### 钩子 B（改动面最小）：`StreetGaussianModel.update_optimizer(exclude_list=...)` 与 `densify_and_prune(exclude_list=...)`

- 文件：`street_gaussian_model.py:532-549`、`573-586`；调用点 `train.py:196, 216`
- 能打断：仍 rasterize 全体（图像一致），但 **跳过某实体的 Adam 与/或 densify**。这是「3D 更新频率」旋钮，不必动 CUDA。
- 对 3D vs 4D：可只 `step` actor 的 Fourier DC、冻 `_xyz`；或只冻 `ActorPose`（关 4D 刚体更新、留 3D 规范系）。当前没有按 param group 名 skip 的现成 API，但每个模型自己的 Adam `param_groups` 已按 `xyz/f_dc/...` 拆开（`gaussian_model.py:292-300`）。
- 风险：rasterize/backward 成本几乎不降（大头仍在 kernel）；densify 统计若仍累加、却不 prune，计数会漂。`exclude_list` 跳过 densify 时务必同时停该模型的 `add_densification_stats` 或清 accum。

### 钩子 C：`train.py` 相机采样（`73-77`）

- 能打断：选哪些相机/时刻（例如动态物体占比高的 view 多训）。
- **不是**实体级算力分配，只是 view 课表；可作辅助，不要当主贡献。
- 风险：与 `split_train`、验证口径纠缠；容易变成「只训难像素/难视角」，滑向 TurboGS 那条像素难度轴。

**不建议作为第一刀：** `diff_gaussian_rasterization` 的 tile/backward。那是改 kernel，本阶段明确禁止当主任务。

### 已有只读计时（未接线，默认不影响数值）

```54:55:train.py
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
```

`iter_start.record()` / `iter_end.record()` 已在逐步调用（`train.py:66,144`），但 **从未 `elapsed_time()`**。若以后加日志：只在 `no_grad` 里读事件、由 `cfg.debug` 或环境变量关掉即可，不改 loss/ densify。本阶段未打开。

---

## 6. 建议的下一阶段实验切口（仍不实现）

1. **先动钩子 B：降低背景更新频率，动态 actor 保持每步 full。** 背景点数量级更大（031 初始化 ~6.7e5 vs 单车 2e4），是算力主体；且背景是纯 3D，和「把 4D 预算留给运动实体」一致。实现上对 `update_optimizer(exclude_list=['background'])` 做 stride，densify 同步 skip。
2. **不要先做相机课表或像素难度采样。** 会和 TurboGS 抢同一条轴；本仓库也没有误差图引导的现成采样。
3. **不要先切 background 空间块。** `GaussianModelBkgd` 是单块，`set_background_mask` 目前是 `pass`（`gaussian_model_bkgd.py:41-42`）。没有现成 tile/块实体。
4. **复杂度分数优先复用现成量，不要先造光流/语义。** 可用：`obj_meta['class']`、box 尺寸、track 上 `opt_trans` 的帧差、LiDAR 覆盖、`obj_bound` 面积、该帧 `graph_obj_list` 长度。缺：光流、像素实例、语义图。
5. **把「4D 更新」定义成两件已存在的事，而不是新网络：** (i) actor Fourier DC；(ii) `ActorPose` 的 Δpose。最小对照可以是：背景少 `step`，actor 的 pose+Fourier 每步都 `step`。固定总 iteration 预算，比先改 kernel 更能验证主张。

---

## 10 行总结

1. 基线是 **ZJU3DV Street Gaussians 官方仓库**（非 4DGS fork）；本机 example 为 `train.py --config configs/example/waymo_train_{002,031}.yaml`。  
2. 一步 = **随机 1 张** Waymo 多相机图；按该帧 timestamp/frame 组装背景 + 窗口内 actor，**一次**拼接 rasterize，再复合天空 cubemap。  
3. Loss：L1+SSIM + sky + LiDAR 深度；densify 结束后加 object-acc 正则。一次 `backward` 打到所有入图参数。  
4. 背景 / 每辆车 / 轨迹 / 天空 **分模型、分 Adam**；actor 外观 Fourier（4D），几何为规范 3D + 刚体 track。  
5. 已有分配：梯度 densify（bkgd/obj 阈值不同）、按帧剔除 actor、frustum 过滤；**无 LOD、无 AMP、无 tile skip、无像素难度采样。**  
6. 训练时能直接读：LiDAR 深度、sky mask、box 投影 `obj_bound`、track id/3D box/类别、时间戳。  
7. **没有**：光流、语义分割图、像素实例 ID、单目深度/法向（002/031 磁盘上也没有）。`dynamic_mask/` 有文件但训练读的是 box 投影。  
8. 最佳插入点：`parse_camera`/`exclude_list` 与 `update_optimizer(exclude_list)`；最小实验 = **背景降更新频率，actor 保持 full**。  
9. 最缺的调度信号是 **像素级运动/实例**（光流、instance mask）；实体级调度应先吃 track+box，而不是先补这些。  
10. 本阶段 **零训练逻辑改动**；CUDA Event 已在逐步 record，尚未打印。

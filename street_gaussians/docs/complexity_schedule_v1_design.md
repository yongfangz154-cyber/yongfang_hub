# 复杂度调度 v1 设计

状态：v1 默认路径是「固定调度 + 分数遥测」。  
本版不按复杂度选人。`score_affects_selection=false` 时，谁入选、谁光栅、谁 Adam、谁 densify、skip 步用哪块区域损失，都与固定 A+B+mask+冻 densify 同一套规则。\(M\)/\(A\) 照算并写入日志，不改名单。`score_affects_selection=true` 的现行接法是 v2.2（分数只闸车辆 Adam），见 `docs/complexity_schedule_v2_design.md`。§5.1 信用账户不再执行。  
对照基线仍是官方 Street Gaussians（`train.schedule` 全部关闭）。固定规则对照是 A+B + `obj_bound` 区域损失 + 背景 skip 时冻结全部车辆 densify。

已拍板、实现时不得改口径：

1. 调度单元是每个 actor（一条 track）和一块不切分的 background。sky 与 ActorPose 不进入选择集合，见 §2。
2. 实现并维护每个实体的运动 \(M\) 与外观残差 \(A\)（EMA）。几何 \(G\) 的接口返回 0。分数写入日志。不新训复杂度网络。默认路径下分数不改名单。
3. `score_affects_selection=false` 时，背景入选当且仅当 `iteration % bkgd_update_stride == 0`（stride>1），与钩子 A/B 相同。不用信用账户。候选车全部入选。
4. 同一条件下，skip 步的 RGB/LiDAR 用固定路的 `guidance['obj_bound']`，sky loss 关掉；`densify_exclude` 含 `background` 和全部 `obj_list`。这是 densify 与 Adam 解耦，和固定路一致，不是按分冻车。逐 track 框只用于 \(A\) 的遥测。
5. `score_affects_selection=true` 的现行接法是 v2.2，见 `docs/complexity_schedule_v2_design.md`：分数只决定车辆 Adam。§5.1 的信用账户和 §5.2 的入选框损失是历史记录，不是这条 true 路径。
6. 背景 MVP 不切空间块。不把 `actor_budget_ratio` 默认改成小于 1。

---

## 1. 问题与需求

仓库是 ZJU3DV Street Gaussians。一步训练抽 1 张相机，`StreetGaussianRenderer.render` 把 background 和当前帧窗口内的 `obj_*` 拼成一次光栅化，再复合 sky cubemap。各实体有自己的 Adam。

已经做过的对照（002，2000 iter，train_view 5 张，无 NaN）说明固定规则不够：

| 阶段 | 做了什么 | 结果 |
|------|----------|------|
| 官方 off | 每步全图光栅化 + 全图损失 | PSNR@2000 25.56，墙钟 313.7 s，车辆点 13.0 万 |
| 钩子 B | `get_update_exclude_list`：stride=4 时 background 少做 Adam/densify，仍全图光栅化 | 墙钟 330.5 s，没有变快；PSNR 23.35 |
| 钩子 A+B | `get_raster_exclude_list`：同一 stride 上 background 不进 rasterize | 墙钟 250.3 s（约快 20%），但 PSNR 从 21.12 掉到 18.90，车辆点涨到 19.8 万 |
| 区域损失 | skip 步 RGB/LiDAR 只用 `guidance['obj_bound']`，并关掉 sky loss | PSNR 回到 22.28，且 1000→2000 不再掉；车辆点涨到 46.2 万 |
| 再冻车辆 densify | skip 步 `densify_exclude` 含全部 `obj_*` | PSNR 22.29，车辆点 14.8 万，墙钟 196.0 s |

空框说明监督必须跟着「这一步真正画了谁」走。`obj_bound` 是该帧全部 track 的框投影并集。2000 步里有 793 步并集为空，这些步跳过了图像损失和反传。非空时 `region_frac` 中位数约 0.043。墙钟里有一块是「空监督、没做 backward」，不能算成少渲染背景的收益。

固定 stride 把整块背景当成永远简单、每辆车当成永远复杂。v1 默认不改这个假设：它就是这条固定规则，外加 \(M\)/\(A\) 遥测。002 上把信用账户配成 stride=4、`force_full_every=8` 时，1..2000 的背景入选集合与 `iteration % 4 == 0` 对称差为 0，但默认路径仍必须直接调用 stride，不能靠这个巧合。信用和按分选人是后续版本。

---

## 2. 实体定义与边界

| 单元 | 代码里的名字 | v1 默认是否按分数进出 rasterize / Adam / densify |
|------|----------------|--------------------------------------------------|
| background | `model_name_id` 的 `'background'`，一个 `GaussianModelBkgd` | 否。只由信用账户和 `force_full_every` 决定。不切块。`GaussianModelBkgd.set_background_mask` 当前是空操作，没有空间块可切 |
| actor | `obj_{track_id:03d}`，在 `StreetGaussianModel.setup_functions` 里创建 | 否。`actor_budget_ratio=1` 时非强制步入选全部候选。分数只排序并写日志，不改变名单 |
| sky | `SkyCubeMap`，不在 `model_name_id` 里 | 否。`StreetGaussianRenderer.render` 在光栅化之后总是复合 cubemap。`StreetGaussianModel.update_optimizer` 总是调用 `sky_cubemap.update_optimizer`。v1 不把 sky 放进选择集合 |
| ActorPose | 一个模块盖住所有 track 的 `opt_trans` / `opt_rots` | 否。`update_optimizer` 没有按 track 跳过 pose 的参数。v1 每步都 `actor_pose.update_optimizer()`。\(M\) 可读轨迹，但不关 pose 的 Adam |
| pose correction / color correction | 若开启，同样在 `update_optimizer` 末尾无条件 step | v1 不改。example 的 color correction 默认关 |

本步候选集合 \(E\)（只有这些名字能进 exclude）：

- 若 `include_background`：`'background'` 永远在 \(E\) 里。它不看帧窗口。
- actor：`obj_model.start_frame <= camera.meta['frame'] <= obj_model.end_frame` 的 `obj_*`。这个窗口条件和 `StreetGaussianModel.parse_camera` 里组建 `graph_obj_list` 的帧条件相同，但此时还没调用 `set_visibility`。
- 帧窗外的 track：不在 \(E\)，本步不能被选中，也不更新它的 \(M\)/\(A\)。它本来就不会进 `graph_obj_list`。
- 在窗口内但投影框为空（在相机后面、框退化）：仍在 \(E\) 里，\(M\) 仍可更新，\(A\) 保持不变。默认 `actor_budget_ratio=1` 时它仍入选。框为空只影响 \(A\) 本步不更新，以及监督掩码里没有这块像素。

sky 的损失：只有「本步选中了 background，且选中了全部候选 actor」时才计算现有的全图 sky loss。其他步关掉，避免天空去补没画出来的背景或没入选的车。

---

## 3. \(M\)、\(A\) 的定义

状态挂在训练循环里，键为 `'background'` 和 `obj_{track_id:03d}`。不要新建网络，不要把分数写进 loss。

记 EMA 系数 \(\beta_m = 0.1\)，\(\beta_a = 0.1\)（配置项，默认值）。更新式：

\[
u \leftarrow (1-\beta)\,u + \beta\,u_{\text{raw}}
\]

只有本节写明的条件下才更新。否则保持上一次的 \(u\)，禁止写成 0。

初值：\(M=0\)，\(A=1\)。\(A\) 取 1 只是日志和再下一版阀门的起点。默认选择不读这个初值，不会因为 \(A=1\) 而多画或少画某个实体。

### 3.1 运动 \(M\)

信号只用数据集轨迹的平移，不用学习到的 `opt_trans`。后者会被优化器自己放大，v1 不把它放进分数。读数路径与 `ActorPose.get_tracking_translation_` 相同：`find_closest_indices` 得到的两帧 `input_trans`。

对候选 actor，在采样到相机之后、`render` 之前：

```
ind1, ind2 = actor_pose.find_closest_indices(track_id, camera.meta['timestamp'])
trans1 = actor_pose.input_trans[ind1]
trans2 = actor_pose.input_trans[ind2]
dt = abs(timestamps[frame2] - timestamps[frame1])  # 秒
if 该 track 的轨迹点少于 2 个，或 dt < 1e-3:
    不更新 M
else:
    M_raw = ||trans2 - trans1|| / dt          # m/s
    M_raw = min(M_raw / motion_norm_mps, 1)   # motion_norm_mps 默认 10
    用 β_m 更新该 track 的 M
```

background 的 \(M\) 恒为 0，不更新。  
track 不在本步候选里：不更新。  
框为空：仍然更新 \(M\)。运动不依赖像素。

### 3.2 外观残差 \(A\)

\(A\) 用这一步已经渲染出来的图，在 `loss.backward()` 之前、`torch.no_grad()` 里计算，并且 `.detach()`。没画出来的实体本步不更新 \(A\)。

禁止用 `guidance['obj_bound']` 当某个 actor 的残差区域。那个张量是 `lib/utils/waymo_utils.py` 里对所有 track 做 `np.logical_or` 之后的并集。单车残差必须用同一套投影，按 track 分开：

- 角点：`bbox_to_corner3d`（`lib/utils/box_utils.py`）加上该 track 在 `obj_info` 里的 `length/width/height`。
- 投影：`get_bound_2d_mask`（同文件）。位姿与 `waymo_utils` 里生成 `obj_bounds` 的循环一致（vehicle 系 box，再投到当前相机）。

```
A_raw(actor) = mean(|render_rgb - gt|) 在该 track 自己的框内
若框内像素数 == 0，或该 actor 本步未入选：不更新 A
否则用 β_a 更新
```

框重叠时，重叠像素可以同时计入多辆车。v1 接受这种重复，不求像素互斥。

background 的 \(A\)，仅当本步 background 入选并且完成了光栅化：

```
static = 图像内、且不落在任一候选 actor 自己的框里的像素
若 static 为空：不更新
否则 A_raw = mean(|render_rgb - gt|) on static，再 EMA
```

background 未入选：不更新 \(A\)，保持原值。不要因为「这步没画」把 \(A\) 清零。

### 3.3 \(G\) 的接口（v2，本版不计算）

```
def geometry_score(name) -> float:
    return 0.0
```

合成时保留 \(w_g \cdot G\)，默认 \(w_g = 0\)。v1 实现不要读深度、点数或尺度来填这个函数。

---

## 4. 合成分数与预算

\[
s_i = w_m M_i + w_a A_i + w_g G_i
\]

默认 \(w_m = w_a = 1\)，\(w_g = 0\)。第一版实现不要搜索这组权重。

代价 \(c_i\) 用该模型当前的高斯数：`get_xyz.shape[0]`。002 初始化时 background 约 82 万，全部车辆约 12.7 万。stride=4 的平均光栅点数约为

\[
L_{\text{ref}} = c_{\text{background}}/4 + \sum_{j \in E_{\text{actor}}} c_j \approx 0.25 \times 8.2\cdot 10^5 + 1.3\cdot 10^5
\]

background 是一整块，\(c_{\text{background}} > L_{\text{ref}}\)。若用「按 \(c_i\) 做背包、装到 \(L_{\text{ref}}\) 就停」，background 永远无法入选。v1 禁止这种背包。

预算拆成两根，长期平均对齐 stride=4，而不是单步硬帽：

**渲染预算 \(B_{\text{render}}\)**

- actor：`actor_budget_ratio` 默认 1。本步允许的车辆点数上限为 \(B_{\text{actor}} = \texttt{actor\_budget\_ratio} \times \sum c_j\)。默认因此放得下全部候选车辆，排序用的 \(s_i\) 不改变名单。第一版不要把这个比例改成小于 1。比例留着，是供再下一版让低分车落选；本版不打开。
- background：信用账户 `bkgd_credit`，初值 0。是否入选与 \(s_{\text{background}}\) 无关。目标入选率 \(1/\texttt{bkgd\_update\_stride}\)，stride 默认 4，即每步 \(+0.25\)。

**更新预算 \(B_{\text{update}}\)**

只约束 Adam，与渲染同一集合。入选者 `Adam.step`；未入选者 `zero_grad` 且不 step。Adam 不设第二套名单。

**densify 与入选 / Adam 解耦**

第 3 路第一次冒烟令 `densify_exclude = update_exclude`。`actor_budget_ratio=1` 时车每步入选，background 未入选的步上车辆仍 densify，结束时 `n_obj=454,858`，超过 20 万。固定 A+B+mask+`freeze_actor_densify_on_skip` 在背景不光栅的步上把车辆放进 densify exclude，点数停在 14.8 万。

因此 densify 不跟「是否入选、是否 Adam」走：

- `raster_exclude = update_exclude = E − selected` 保持不变。未入选者不光栅、不 Adam。ratio=1 时候选车仍入选并 Adam。
- 背景按 stride 被 skip 时，`densify_exclude` 含 `'background'` 以及模型里的全部 `obj_*`（`obj_list`，不是只含本步候选）。用完整名 `obj_{id:03d}`，不要写前缀 `'obj'`。这一步不读 \(s_i\)。这些车不积累半径、不积累梯度、不 `densify_and_prune`，但 `update_exclude` 不含它们，Adam 照常 step。
- 背景已入选时，`densify_exclude` 为空（ratio=1，没有未入选的车）。

这是为了与固定规则做 densify parity，不是按分冻车。分数仍然不改变 `selected`。

**v1 默认不走信用账户**

`score_affects_selection=false` 时，背景是否入选只看 `iteration % bkgd_update_stride == 0`。不要在默认路径里累加 `bkgd_credit`，也不要因为 `force_full_every` 改名单。下面的信用账户只在 `score_affects_selection=true` 时使用。

**信用与强制全选如何一起对齐 1/4（仅后续版本）**

每个训练步：`bkgd_credit += 1/stride`。  
`iteration % force_full_every == 0` 时无条件全选，并且 `bkgd_credit -= 1`（允许变成负数）。  
非强制步：若 `bkgd_credit >= 1`，本步选中 background，然后 `bkgd_credit -= 1`。这里不读 \(M\) 或 \(A\)。信用未满时 background 不入选，也不是因为分数低。延迟上限就是信用从 0 攒到 1 的步数（stride=4 时约 4 步）。强制全选会把账户打成负数，从而把下一次信用入选推后。

每 8 步强制一次且每步 +0.25、强制步再 −1 时，8 步内信用净增 \(8\times 0.25 - 1 = 1\)，再触发一次信用入选。加上那一次强制，8 步里 background 大约入选 2 次，比率约 1/4。实现后用日志里的入选率核对，允许与 0.25 相差 ±0.05。不要为了把单步点数压进 \(L_{\text{ref}}\) 而改掉这根信用。

---

## 5. 逐步算法

插入位置都在 `train.py` 的 `training()` 里，符号如下。不要改 `submodules/diff-gaussian-rasterization`。

1. 采样相机之后，现有代码已经从 `viewpoint_cam.guidance` 取出 `obj_bound` / `sky_mask` / `lidar_depth`。v1 不替换这次采样。
2. 在调用 `StreetGaussianRenderer.render` 之前：
   - 按 §2 建 \(E\)。
   - 按 §3.1 更新候选 actor 的 \(M\)。
   - 用当前 EMA 算 \(s_i\) 并写入日志。分数不改变默认名单。
   - `score_affects_selection=false`：背景入选当且仅当 `iteration % bkgd_update_stride == 0`；候选车全部入选。`raster_exclude = update_exclude` 为 `['background']` 或 `[]`。不读信用账户，不跑 §5.1。
   - `score_affects_selection=true`：现行是 v2.2，不是本节旧稿。分数只写入 `update_exclude`。`raster_exclude`、区域掩码和 `densify_exclude` 不读分数。规则见 `docs/complexity_schedule_v2_design.md`。
   - 调用现有的 `gaussians_renderer.render(..., exclude_list=raster_exclude)`。`render` 内部用 `set_visibility` + `parse_camera`，未列入 include 的模型不会进入 `graph_obj_list`，也不会进 `get_xyz`。
3. 渲染之后、`loss.backward()` 之前，在 `no_grad` 里按 §3.2 更新本步入选且框非空的 \(A\)。这只服务于日志。
4. 背景 skip 用 `guidance['obj_bound']` 做 RGB/LiDAR，关掉 sky；背景入选则全图 + sky。`score_affects_selection=true` 也走这一套，不按 §5.2 把掩码收成入选车。然后才是现有的 `loss.backward()`。空损失时 `loss.requires_grad` 为假则不调用 `backward`。
5. densify 与 Adam 解耦，规则见 §4。默认路径在背景 skip 时把全部 `obj_list` 放进 `densify_exclude`，与 `freeze_actor_densify_on_skip` 一致。不要按分数挑选被冻的车。
6. `StreetGaussianModel.update_optimizer(exclude_list=update_exclude)` 保持现有行为：名单里的模型 `zero_grad` 且不 `Adam.step`。不要把 `'sky'` 或 `'actor_pose'` 写进这个名单，现有循环也管不到它们。

`complexity_enable=false` 时走现有的 `get_raster_exclude_list` / `get_update_exclude_list` / `apply_region_loss`，一行都不要换。

### 5.1 选择集合（仅 `score_affects_selection=true`）

默认实验不执行本节。`score_affects_selection=false` 时用上一节的 stride 规则。下面的信用账户和按点数背包没有被采用。`score_affects_selection=true` 的现行规则见 `docs/complexity_schedule_v2_design.md`：分数只决定车辆 Adam，光栅、监督和 densify 不读分数。

```
force = (iteration % force_full_every == 0)
bkgd_credit += 1 / bkgd_update_stride

if force:
    selected = E 的全部名字
    bkgd_credit -= 1
    n_forced_full += 1
else:
    actors = E 里的 obj_* ，按 s 从高到低
    selected_actors = []
    cost = 0
    for name in actors:
        if cost + c_name <= B_actor or selected_actors 为空:
            # 至少留下分数最高的一辆，避免 B_actor 因取整变成 0 时清空
            selected_actors.append(name)
            cost += c_name
        # actor_budget_ratio 默认 1 时，循环会收进每一辆
    selected = selected_actors
    if 'background' in E and bkgd_credit >= 1:
        selected.append('background')
        bkgd_credit -= 1
```

上面的伪代码保留作记录，现行 true 路径不执行它。背景不读分数。默认实验不进入本节。

### 5.2 监督

下面是 v2 / v2.1 的失败消融记录，训练不再执行。`score_affects_selection=true` 的现行监督与 false 路径相同：背景 skip 用 `guidance['obj_bound']`，背景入选用全图 `mask` 加 sky。

记 `full_scene = ('background' in selected) 且 selected 含全部候选 actor`。

- `full_scene` 为真：RGB（L1+SSIM）、sky loss、LiDAR 与官方 `training()` 相同，掩码用原来的全图 `mask`。
- 否则：RGB 和 LiDAR 只用本步 `selected` 里 actor 的框的并集（§3.2 的逐 track 框，不是 `guidance['obj_bound']`）。sky loss 不加入 `loss`。并集像素数为 0：本步不加入图像、天空、LiDAR，`skipped_image_loss=true`，不反传。
- 未入选的车的框不要出现在掩码里。否则画面上的洞会由别的车或天空解释。

`complexity_enable=false` 时，区域损失仍只由 `mask_loss_on_skip` 打开。`complexity_enable=true` 且 `score_affects_selection=false` 时，复用同一段 `obj_bound` 代码，不要再叠一层逐 track 损失掩码。逐 track 框只更新 \(A\)。

### 5.3 安全退化

| 情况 | 行为 |
|------|------|
| \(E\) 里没有 actor，信用未选中 background | 空监督，不反传，记 `n_empty_supervision` |
| 某 track 框为空 | 不更新该 track 的 \(A\)；掩码里没有它；\(M\) 仍可更新 |
| `find_closest_indices` 不足两帧 | 不更新 \(M\) |
| `loss` 没有 `grad_fn` | 不调用 `backward` |
| `complexity_enable=false` | 完全回到现有 schedule 开关，包括全关时的官方行为 |

禁止 NaN：空掩码不要调用 `l1_loss` / `ssim` 的 `.mean()`。

exclude 名单传给 `startswith_any`（`lib/utils/general_utils.py`）。它是前缀匹配。`obj_100` 会误伤 `obj_1000`。v1 的名单必须是完整的 `obj_{id:03d}` / `'background'`。实现时加一个断言：任意两个模型名互不为前缀。不要把 `'obj'` 放进名单。

---

## 6. 配置项草案

全部加在 `cfg.train.schedule`（`lib/config/config.py`）。默认值必须让 `complexity_enable=false` 时行为与现在的代码一致，包括官方 off。不要改 `configs/example/waymo_train_002.yaml`。

| 字段 | 默认 | 含义 |
|------|------|------|
| `complexity_enable` | `false` | 总开关。false 时忽略本节其余新字段 |
| `bkgd_update_stride` | 已有，默认 4 | 信用每步增加 `1/stride`。复杂度打开时 background 不再由 `iteration % stride` 决定，也不由分数决定 |
| `force_full_every` | 8 | 每隔这么多 iter 选中全部候选。该步不读分数 |
| `actor_budget_ratio` | 1.0 | 相对「全部候选车辆点数」的上限。第一版保持 1 |
| `w_m`, `w_a`, `w_g` | 1, 1, 0 | \(w_g\) 必须为 0 |
| `motion_ema_beta`, `appearance_ema_beta` | 0.1, 0.1 | |
| `motion_norm_mps` | 10 | \(M\) 的截断尺度 |
| `a_init` | 1.0 | \(A\) 初值 |

已有的 `enable`、`skip_bkgd_raster`、`mask_loss_on_skip`、`freeze_actor_densify_on_skip` 在 `complexity_enable=true` 时不再生效，避免两套 exclude 打架。文档和启动日志要打印一句：`complexity_enable` 覆盖 stride 钩子。

示例只放在这里，不要写成仓库里的默认 yaml：

```yaml
train:
  schedule:
    complexity_enable: true
    bkgd_update_stride: 4
    force_full_every: 8
    actor_budget_ratio: 1.0
    w_m: 1
    w_a: 1
    w_g: 0
    motion_ema_beta: 0.1
    appearance_ema_beta: 0.1
    motion_norm_mps: 10
    a_init: 1.0
    log_interval: 50
    log_time: true
```

消融时只改 `force_full_every`：4 用来看强制全选更密时，相对默认 8 的质量；16 用来看两次强制之间信用账户单独运转时的质量。默认实验用 8。这三组都不是「分数已经在挑人」的实验。不要三组一起当成主结果。

---

## 7. 日志与归因

沿用现有 `[schedule]` 启动行和 `[ITER k]` 行（`training()` 里 `schedule_log_interval` 那一段），并增加结束时的汇总。目的是把墙钟拆开，避免把空监督当成少渲染。

每条周期日志至少包含：

- `selected` 名字、`raster_exclude`
- `n_raster`（现有，`parse_camera` 之后的 `gaussians.num_gaussians`）
- `bkgd_credit`
- `forced_full`（0/1）
- `region_frac`、`skipped_image_loss`
- `n_bkgd`、`n_obj`
- `score_affects_selection`：当 `actor_budget_ratio==1` 且背景只由信用账户与 `force_full_every` 决定时，本字段为 `false`。实现版再把它接到日志；设计语义是：本步入选集合与「把 \(s_i\) 全部换成常数」的集合相同。`actor_budget_ratio<1`，或背景入选开始读取 \(s_{\text{background}}\) 时，本字段为 `true`。v1 默认实验必须看到 `false`

结束时 `[DONE]` 增加：

- `n_forced_full`：强制全选的步数
- `n_empty_supervision`：空掩码、未反传的步数
- `bkgd_inclusion_rate`：background 入选步数 / 总步数
- 每个 `obj_*` 的入选率（可以每 50 步打一行累计，结束时再打一次；不要每步打所有 track）
- `cuda_fwd_bwd_ms` 已有，继续用 `log_time`。另记两段墙钟：`score_s`（建 \(E\)、更新 \(M\)、选择）和 `backward_s`（只在真正调用了 `backward` 的步上累加）。空监督步的 `backward_s` 不加

解释加速时只比较这三项，不要只报总 `wall_s`：

1. `n_raster` 相对 off 是否下降（少光栅化）。
2. `n_empty_supervision / iterations` 是否很高（少反传，来自空框，不是复杂度本身）。
3. `bkgd_inclusion_rate` 是否落在约 0.20–0.30（stride=4、force=8 的预期，见 §4）。

---

## 8. 验收冒烟协议

场景 waymo 002，2000 iter，5 张 train_view，同一 `load_pcd_from`，与 `docs/schedule_mvp_notes.md` 现表同一协议。不在本设计稿里开跑。

| # | 设定 | 日志 |
|---|------|------|
| 1 | 调度全关 | 复用 `logs/schedule_smoke_off.log` |
| 2 | 固定 A+B + `mask_loss_on_skip` + `freeze_actor_densify_on_skip`，stride=4 | 复用 `logs/schedule_smoke_mask_freeze.log` |
| 3 | 复杂度 v1：`complexity_enable=true`，`force_full_every=8`，`actor_budget_ratio=1`，权重默认 | 新日志，不要覆盖 1 和 2 |

第 2 路已有数字：PSNR@1000 21.59，PSNR@2000 22.29，墙钟 196.0 s，`n_bkgd` 2,227,898，`n_obj` 147,617，无 NaN。第 1 路：PSNR@2000 25.56，墙钟 313.7 s，`n_obj` 129,852。

第 3 路成功 = 管道正确，并且质量不显著差于第 2 路固定规则（parity）。不声称复杂度调度已经优于 stride，也不把主冒烟写成「按 \(M\)/\(A\) 选出了更好的集合」。

在此之上，第 3 路还要全部满足：

- 无 NaN。
- PSNR@2000 ≥ 21.29（不比第 2 路低过 1 dB），且 PSNR@2000 ≥ PSNR@1000 − 1（不允许再出现 A+B 那种 1000→2000 下滑）。
- 结束时 `n_obj` ≤ 200,000（未再出现「只开 mask、车辆点到 46 万」）。
- `bkgd_inclusion_rate` 在 0.20 与 0.30 之间。
- 日志中 `score_affects_selection=false`（`actor_budget_ratio==1` 且背景未读分数）。
- `n_empty_supervision` 被单独打印。墙钟可以短于 off，但讨论加速时必须同时给出空监督占比。不把墙钟短于第 2 路写成复杂度已优。

失败也要留下：入选率、`region_frac`、空监督步数、以及未入选实体是否仍出现在损失掩码里。失败时不要改权重再跑一串；回到 §10 的下一版。

`force_full_every` 的 4 和 16 不在这次成功判据里。主冒烟通过之后才各跑一次，只看入选率和 PSNR 的方向，不作为 v1 是否成立的条件。

---

## 9. 非目标

- 不改 CUDA，不改 `diff-gaussian-rasterization`，不做 tile skip。
- 不做 TurboGS 式的按像素误差采相机。相机采样保持 `training()` 里现有的随机 pop。
- 不做 DriveSplat 式的按深度 LOD densify。
- 本版不加 \(G\)，不把 \(w_g\) 设成非 0。
- 不把 background 切成空间块。
- 不按分数关闭 sky 或 ActorPose 的 Adam。
- 不新训复杂度网络，不用光流或语义分割。
- 不把 `complexity_enable` 的默认值改成 true，不改官方 example yaml。
- 第一版实现不扫描 \(w_m\)、\(w_a\)、`actor_budget_ratio`。

---

## 10. 实现任务拆分

下一版 Cursor 只做「接上选择集合」，做完就停，用来打磨：

1. 按 §6 加配置，默认 `complexity_enable=false`。
2. 在 `training()` 里按 §5 从 \(E\) 得到 `raster_exclude` / `update_exclude` / `densify_exclude`，接到现有 `render(..., exclude_list=)` 和 `update_optimizer`。
3. 实现 §3 的 \(M\)、\(A\) 和 EMA。\(G\) 只留返回 0 的函数。
4. 监督按 §5.2。逐 track 框走 `get_bound_2d_mask`，不要用并集 `obj_bound` 给单车打分。
5. 日志按 §7。
6. 跑 §8 的第 3 路一次。不过线就停，把失败表写回 `docs/schedule_mvp_notes.md`。

第一次第 3 路把 densify 和 Adam 绑在同一份名单上，`n_obj` 超过 20 万。densify 已按 §4 与入选解耦，重跑仍不降低 `actor_budget_ratio`，分数仍不改变 `selected`。再下一版才让分数影响选择：降低 `actor_budget_ratio`，和/或让背景入选读取分数，并补 `force_full_every` 为 4 和 16 的消融。那一版之前，非强制步应仍然保留全部候选车辆，背景仍只由信用和强制全选决定。

# Entity-level update schedule MVP（钩子 B）

**当前推荐默认（`complexity_enable=true` 且 `score_affects_selection=true`）：v2.2 接法。** 分数只决定车辆 Adam。全部候选车光栅；densify 与区域监督跟第 2 路，不读分数。v2 / v2.1 只是失败消融。总表和口径在文末「锁定口径」。

固定 iteration 预算下，降低 **background** 的 Adam / densify 频率；所有 `obj_*`、`actor_pose`、`sky` 每步照常更新。本 MVP **不改** `diff-gaussian-rasterization`，也不做像素难度采样或深度 LOD densify。

**预期限制（不是 bug）：** rasterize + backward 仍然覆盖全体高斯，墙钟加速主要来自少做 background 的 `optimizer.step` / densify，幅度通常不大。要真正少渲染背景需要钩子 A（`parse_camera` / `exclude_list` 进 rasterizer）。

---

## 开关（默认关闭 = 原官方行为）

在 `lib/config/config.py`：

| 字段 | 默认 | 含义 |
|------|------|------|
| `train.schedule.enable` | `false` | `false` 时 `exclude_list=[]`，optimizer / densify 调用集合与改前一致 |
| `train.schedule.bkgd_update_stride` | `4` | 仅 `enable=true` 时生效：`iteration % stride == 0` 才更新 background |
| `train.schedule.always_update` | `["actors", "actor_pose", "sky"]` | 语义说明，训练代码不读此列表 |
| `train.schedule.log_interval` | `100` | 周期性打印 `exclude_list` / 是否更新 background |
| `train.schedule.log_time` | `false` | 打印 CUDA Event 的 fwd+bwd 毫秒；默认关，不影响数值 |

yaml 示例：

```yaml
train:
  schedule:
    enable: true          # 一键关：false 或整段省略
    bkgd_update_stride: 4
    always_update: ["actors", "actor_pose", "sky"]
    log_interval: 50
    log_time: false
```

命令行覆盖也可以：`train.schedule.enable True train.schedule.bkgd_update_stride 4`。

---

## 改了哪些函数

| 文件 | 改动 |
|------|------|
| `lib/config/config.py` | 增加 `cfg.train.schedule.*`，默认 `enable=False` |
| `train.py` | `get_update_exclude_list(iteration)`；把 `exclude_list` 传入 densify 统计 / prune / `update_optimizer`；周期性日志；结束时打印墙钟与高斯数 |
| `lib/models/street_gaussian_model.py` | `update_optimizer`：被 exclude 的模型 `zero_grad` **不** `Adam.step`（避免梯度跨步累积）；`set_max_radii2D` / `add_densification_stats` / `densify_and_prune` 跳过 exclude 模型，避免 skip densify 时污染 accum |

未改：`submodules/diff-gaussian-rasterization`、loss、example 数据协议、`waymo_train_002.yaml`。

调度规则（`enable=true`）：

- background：仅当 `iteration % bkgd_update_stride == 0` 才进入本 step 的 `optimizer.step` 与 `densify_and_prune`；其它 step 进 `exclude_list=['background']`，同时跳过该模型的 densify 统计。
- 所有 `obj_*`、`actor_pose`、`sky`（以及 color/pose correction 若开启）：每步更新。某 actor 本帧不在 graph 中仍走原 `parse_camera`。

---

## 复现实验

环境：`conda activate street_gaussians`。复用已跑通的 002 COLMAP / 初始化点云，**不覆盖** `configs/example/waymo_train_002.yaml`。

```bash
# 1) 基线：schedule off，2000 iter 冒烟
python -u train.py --config configs/example/waymo_train_002_schedule_smoke.yaml \
  2>&1 | tee logs/schedule_smoke_off.log

# 2) 实验：schedule on，bkgd stride=4，同样 2000 iter
python -u train.py --config configs/example/waymo_train_002_schedule_on_smoke.yaml \
  2>&1 | tee logs/schedule_smoke_on.log
```

输出目录：

- `output/waymo_full_exp/waymo_train_002_sched_off_smoke/`
- `output/waymo_full_exp/waymo_train_002_sched_on_smoke/`

验收日志关键字：`[schedule] enable=...`、`[ITER ...] exclude_list=`、`update_background=`、`[DONE] wall_s=`。开启后应看到 `exclude_list=['background']` 与 `update_background=False` 周期性出现。

example 的 `split_train: 1` 使测试相机为空；表中 PSNR 来自 `training_report` 的 **train_view** 抽样（5 张），不是 held-out test。

---

## 冒烟结果（002，2000 iter，RTX 4090）

初始化点云复用 `output/waymo_full_exp/waymo_train_002`（`load_pcd_from`）。完整日志：`logs/schedule_smoke_{off,on}.log`。

| 设定 | enable | stride | train_view PSNR @1000 | train_view PSNR @2000 | wall_s（训练环） | NaN | n_bkgd 初→终 | n_obj 初→终 | cuda fwd+bwd median |
|------|--------|--------|------------------------|------------------------|------------------|-----|--------------|-------------|---------------------|
| off  | false  | n/a    | 24.30                  | 25.56                  | 313.7            | 无  | 820876 → 1,648,001 | 126530 → 129852 | 119.4 ms |
| on   | true   | 4      | 22.24                  | 23.35                  | 330.5            | 无  | 820876 → 2,194,170 | 126530 → 128313 | 123.8 ms |

开启后日志样例（stride=4）：

```
[schedule] enable=True bkgd_update_stride=4 log_interval=50 log_time=True
[ITER 1] exclude_list=['background'] update_background=False ...
[ITER 4] exclude_list=[] update_background=True ...
[ITER 5] exclude_list=['background'] update_background=False ...
[ITER 100] exclude_list=[] update_background=True ...
[ITER 101] exclude_list=['background'] update_background=False ...
```

关闭时全程 `exclude_list=[]`、`update_background=True`。

### 读数时要注意

- **墙钟没有变快（略慢 ~5%）是预期限制，不是 bug。** 本 MVP 只降 background 的 Adam / densify-stats 频率；rasterize+backward 仍覆盖全体高斯。`log_time` 的 CUDA Event 只包 fwd+bwd，两边都在 ~120 ms。on 跑终局 background 点更多，光栅更重。要拉墙钟需要钩子 A（真正不渲染背景）。
- PSNR 在 2000 step 上掉约 2.2 dB（train_view 抽样，很初步）。actor 点数两边接近（~12.9 万 vs ~12.8 万）。
- `densification_interval=100` 能被 stride=4 整除，因此 background 的 `densify_and_prune` **触发步仍是 600/700/…**，与基线相同；被 skip 的是这些步之间的 Adam 和 densify accum。accum 样本变少、估计更噪，on 跑 background 点数反而更多（220 万 vs 165 万）。若以后要连 densify 一起降频，需要单独旋钮或让 stride 与 densify interval 不对齐。
- 默认 `enable=false` 时 optimizer / densify 的调用集合与改前一致（`exclude_list=[]`）。原 example yaml 未覆盖，50k 基线可复现。

下一步候选：用更长 iteration / 真正 test split 再采论文指标。钩子 A 的开关和冒烟命令见下一节。

---

## 钩子 A（少渲染 background）

在钩子 B 的同一 stride 上，off-stride 步不把 `background` 送进 `StreetGaussianRenderer.render(..., exclude_list=)`。`parse_camera` 已有的 `get_visibility` 因此不会把背景点拼进这次 rasterize，fwd+bwd 都不覆盖背景。`obj_*`、ActorPose、sky cubemap 仍每步。**不改** `diff-gaussian-rasterization`。

| 字段 | 默认 | 含义 |
|------|------|------|
| `train.schedule.skip_bkgd_raster` | `false` | `false` 时训练调用 `render(exclude_list=[])`，与改前一致 |
| `train.schedule.enable` | `false` | 钩子 B。钩子 A 冒烟同时打开，避免背景没进图却仍 `Adam.step` |
| `train.schedule.bkgd_update_stride` | `4` | 两个钩子共用。`iteration % stride == 0` 时背景照常渲染并更新 |

`skip_bkgd_raster=false` 时，已有的 off/on 冒烟 yaml 行为不变（on 仍只跳 Adam/densify）。

```bash
python -u train.py --config configs/example/waymo_train_002_schedule_raster_smoke.yaml \
  2>&1 | tee logs/schedule_smoke_raster.log
```

输出目录：`output/waymo_full_exp/waymo_train_002_sched_raster_smoke/`。  
验收：`[schedule] ... skip_bkgd_raster=True`；stride=4 时 `[ITER 1] raster_exclude=['background'] raster_background=False`，`[ITER 4] raster_exclude=[] raster_background=True`。关闭时全程 `raster_exclude=[]`。

已知限制：跳过的背景该步没有梯度，也没有 densify 统计。天空合成用的是前景 `acc`，背景不进图时天空会填进本该由背景覆盖的像素。

### 钩子 A 冒烟（002，2000 iter，RTX 4090，2026-09-24）

日志：`logs/schedule_smoke_raster.log`。输出：`output/waymo_full_exp/waymo_train_002_sched_raster_smoke/`。无 NaN。PSNR 仍是 train_view 5 张。

| 设定 | PSNR@1000 | PSNR@2000 | wall_s | n_bkgd 终 | n_obj 终 | fwd+bwd |
|------|-----------|-----------|--------|-----------|----------|---------|
| off | 24.30 | 25.56 | 313.7 | 1,648,001 | 129,852 | median 119.4 ms |
| 钩子 B | 22.24 | 23.35 | 330.5 | 2,194,170 | 128,313 | median 123.8 ms |
| 钩子 A+B | 21.12 | 18.90 | 250.3 | 2,272,216 | 198,360 | 跳过步 median 88.8 ms；整步 median 114.7 ms |

跳过步 `n_raster` median 118,530（只有车辆），整步 median 1,466,639。墙钟比 off 短约 20%。PSNR 在 1000→2000 继续下降，车辆点数明显涨过 off / 钩子 B。

---

## skip 步区域损失（堵住车补背景）

诊断：钩子 A 的 skip 步不渲染背景，但 L1 / SSIM / 天空 loss / LiDAR 仍对整张图计算，梯度和 densify 统计都打在车上和天空上。

插入点：`train.py` 里算出 `raster_bkgd` 之后、`loss.backward()` 之前。只在 `mask_loss_on_skip=true` 且本步 `raster_exclude` 含 `background` 时生效。`enable=false` 时背景每步都进光栅化，这条分支不进入，官方全图损失不变。裸 A+B（`mask_loss_on_skip=false`）也不变。

| 字段 | 默认 | skip 步行为 |
|------|------|-------------|
| `train.schedule.mask_loss_on_skip` | `false` | RGB（L1+SSIM）和 LiDAR 只在 `guidance['obj_bound']` 上计算。天空 loss 整项不算，避免天空扛背景误差。`obj_bound` 缺失或全空：本步跳过图像 / 天空 / LiDAR，不反传，打日志，不产生 NaN |
| `train.schedule.freeze_actor_densify_on_skip` | `false` | 本步 densify 统计和 `densify_and_prune` 跳过全部 `obj_*`。背景本来就在 exclude 里 |

`obj_bound` 是数据加载时已有的 3D box 投影并集，不是新网络。

```bash
python -u train.py --config configs/example/waymo_train_002_schedule_mask_smoke.yaml \
  2>&1 | tee logs/schedule_smoke_mask.log
python -u train.py --config configs/example/waymo_train_002_schedule_mask_freeze_smoke.yaml \
  2>&1 | tee logs/schedule_smoke_mask_freeze.log
```

### 冒烟（002，2000 iter，5 train views，RTX 4090，2026-09-24）

off 与裸 A+B 复用已有日志。无 NaN。

| 设定 | PSNR@1000 | PSNR@2000 | wall_s | n_bkgd 终 | n_obj 终 | 相对 off 墙钟 |
|------|-----------|-----------|--------|-----------|----------|----------------|
| off | 24.30 | 25.56 | 313.7 | 1,648,001 | 129,852 | — |
| 钩子 B | 22.24 | 23.35 | 330.5 | 2,194,170 | 128,313 | 慢约 5% |
| A+B | 21.12 | 18.90 | 250.3 | 2,272,216 | 198,360 | 快约 20% |
| A+B + mask | 21.41 | 22.28 | 195.5 | 2,246,246 | 462,067 | 快约 38% |
| A+B + mask + 冻结车辆 densify | 21.59 | 22.29 | 196.0 | 2,227,898 | 147,617 | 快约 38% |

mask 日志里非空 `obj_bound` 的 `region_frac` 中位数约 **0.043**（大约 4% 像素）。2000 步里有 **793** 步框为空，这些步跳过了图像损失和反传。所以墙钟比裸 A+B 更短，有一部分是空步没做 backward，不只是少渲染背景。

读数：相对裸 A+B，PSNR@2000 从 18.90 回到 22.3，而且 1000→2000 不再下降。只开 mask 时车辆点涨到 46 万，densify 仍在吃框内梯度。加上 `freeze_actor_densify_on_skip` 后车辆点 14.8 万，接近 off 的 13.0 万，PSNR 与只开 mask 几乎相同。两路都还低于 off 的 25.56，这是 2000 步、训练视角上的阶段性对照。

下一步的设计是把管道接上，不是这一版就按分数挑人。空框比例高，说明监督必须跟着入选集合走；若仍用整帧全图损失，会重复「少渲染、全图反传」的错配。

复杂度 v1 的规格在 `docs/complexity_schedule_v1_design.md`（方案 A）。实现已落地：`lib/config/config.py` 的默认开关仍是关；`lib/utils/complexity_schedule.py` 算 \(M\)/\(A\)（\(G\) 返回 0）并按 §5.1 选人；`train.py` 在采样相机之后、`render` 之前接入。`actor_budget_ratio=1` 时分数不改变名单，background 只由信用账户和 `force_full_every=8` 决定。监督用逐 track 的 `get_bound_2d_mask`，不用 `guidance['obj_bound']` 并集。

### 第 3 路（未过线，未改权重重跑）

命令（只跑这一次，不覆盖 off / mask_freeze 日志）：

```bash
python -u train.py --config configs/example/waymo_train_002_schedule_complexity_smoke.yaml \
  2>&1 | tee logs/schedule_smoke_complexity.log
```

002，2000 iter，5 train views，同一 `load_pcd_from`，RTX 4090。无 NaN。日志里抽查的 87 条周期记录中，出现在 `scores` 里的 `obj_*` 全部在 `selected` 中（`dropped=0`），`score_affects_selection=false`。非空且非整图的 `region_frac` 中位数约 0.043。

| 设定 | PSNR@1000 | PSNR@2000 | wall_s | n_bkgd | n_obj | n_empty_supervision | bkgd_inclusion_rate | score_affects_selection |
|------|-----------|-----------|--------|--------|-------|---------------------|---------------------|-------------------------|
| 第 2 路 固定 A+B+mask+冻 densify | 21.59 | 22.29 | 196.0 | 2,227,898 | 147,617 | 793 | — | — |
| 第 3 路 复杂度 v1（ratio=1） | 21.56 | 22.32 | 222.3 | 2,238,478 | 454,858 | 793 | 0.250 | false |

判据：PSNR@2000 22.32 ≥ 21.29，且高于 PSNR@1000；入选率 0.250 落在 [0.20, 0.30]；`score_affects_selection=false`；`n_empty_supervision=793` 已打印。**未过线的是 `n_obj=454,858` > 200,000。** 这和「只开 mask、不冻车辆 densify」的 462,067 同一现象：ratio=1 时每步候选车都入选，跳过背景的步上车辆仍在 densify。第 2 路把这些步的车辆 densify 冻住，所以点停在 14.8 万。当时规格把 densify 和 Adam 绑在同一份名单上，没有改权重再跑。墙钟 222.3 s 长于第 2 路，不能写成复杂度已优。

### 第 3 路补丁：densify 与 Adam 解耦（过线）

规格改为：`raster_exclude` / `update_exclude` 仍是候选集减去入选集，车可以入选并 Adam；background 未入选时，`densify_exclude` 再并上本步全部候选 `obj_*`。这是对齐固定规则的 densify，不是按分冻车。`actor_budget_ratio` 仍为 1。

```bash
python -u train.py --config configs/example/waymo_train_002_schedule_complexity_smoke.yaml \
  2>&1 | tee logs/schedule_smoke_complexity_v1_densifyfix.log
```

同一协议重跑一次。无 NaN。周期日志 87 条里，`scores` 中的 `obj_*` 都在 `selected` 中；background 未入选时这些车都在 `densify_exclude` 里，background 入选时 densify 名单不含车。

| 设定 | PSNR@1000 | PSNR@2000 | wall_s | n_bkgd | n_obj | n_empty_supervision | bkgd_inclusion_rate | score_affects_selection |
|------|-----------|-----------|--------|--------|-------|---------------------|---------------------|-------------------------|
| 第 2 路 固定 A+B+mask+冻 densify | 21.59 | 22.29 | 196.0 | 2,227,898 | 147,617 | 793 | — | — |
| 第 3 路 解耦前 | 21.56 | 22.32 | 222.3 | 2,238,478 | 454,858 | 793 | 0.250 | false |
| 第 3 路 densify 解耦 | 21.50 | 21.89 | 219.0 | 2,253,714 | 148,212 | 793 | 0.250 | false |

`n_obj=148,212` ≤ 200,000。PSNR@2000 21.89 ≥ 21.29，且不低于 PSNR@1000 减 1。入选率 0.250，`score_affects_selection=false`，`n_empty_supervision=793`。这一路过线。PSNR 比第 2 路低约 0.4 dB，墙钟 219.0 s 仍长于 196.0 s，不写成复杂度已优。空监督仍占 793/2000。

### 诊断：0.4 dB 不是背景入选相位

训练循环是 iter 1..2000（没有第 0 步）。第 2 路背景入选条件是 `iteration % 4 == 0`。第 3 路 densify 解耦用的是信用账户加 `force_full_every=8`。把这套信用规则在 1..2000 上回放，入选集合与 `% 4 == 0` **完全相同**：两边都是 500 步，对称差 **0**。相位都落在 `iter % 8 ∈ {0, 4}`，各 250 步。

两份日志各有 87 条带 `raster_background` 的周期记录，共同步上背景是否入选 **0 处不一致**，`region_loss` / `skipped_image_loss` / `freeze_actor_densify` 也 0 处不一致。因此 0.4 dB 不是「差了多少步、差在哪个相位」。

仍对不齐的是实现路径，不是入选集合：

- 默认路径用信用账户**碰巧**等于 stride=4 且 force=8 的集合，并没有调用第 2 路的 `% stride`。
- 背景 skip 时，第 2 路 `densify_exclude` 是全部 `obj_list`；第 3 路只并上本步候选车。87 步里 `n_raster` 有 56 步不同，后段大约差几百个点。
- 区域监督：第 2 路用 `guidance['obj_bound']`，第 3 路用逐 track 框。87 步里 `region_frac` 有 13 步差在 0.0001 量级，空监督步数都是 793。

下一步把 `score_affects_selection=false` 改成直接复用第 2 路：stride 决定背景，skip 时走 `obj_bound` 区域损失，并冻全部 `obj_*` 的 densify。信用账户留给 `score_affects_selection=true`。分数仍只记录。

### 第 3 路：默认路径改成固定 stride（过线）

`score_affects_selection=false` 时不再累加信用。背景入选就是 `iteration % 4 == 0`。skip 步的损失用 `obj_bound`，`densify_exclude` 含 `background` 和全部 `obj_list`。\(M\)/\(A\) 仍写入周期日志。

```bash
python -u train.py --config configs/example/waymo_train_002_schedule_complexity_smoke.yaml \
  2>&1 | tee logs/schedule_smoke_complexity_v1_parity.log
```

iter 1..2000 上，这套 stride 与第 2 路的背景入选集合对称差为 0（都是 500 步，相位 `iter % 8 ∈ {0, 4}`）。87 条共同周期记录里 `raster_background`、`region_frac`、`skipped_image_loss`、`freeze_actor_densify` 全部一致。`n_raster` 仍有 56 步不同，结束点数差约 600。无 NaN。`score_affects_selection=false` 出现 88 次，分数行 87 条，`n_empty_supervision=793`。

| 设定 | 背景入选集合 | PSNR@1000 | PSNR@2000 | wall_s | n_bkgd | n_obj | n_empty_supervision | score_affects_selection |
|------|----------------|-----------|-----------|--------|--------|-------|---------------------|-------------------------|
| 第 2 路 固定 A+B+mask+冻 densify | `iter % 4 == 0`，500 步 | 21.59 | 22.29 | 196.0 | 2,227,898 | 147,617 | 793 | — |
| 第 3 路 densify 解耦（信用） | 与上表对称差 0，但是信用碰巧算出 | 21.50 | 21.89 | 219.0 | 2,253,714 | 148,212 | 793 | false |
| 第 3 路 固定 stride + 分数遥测 | 与第 2 路对称差 0 | 21.43 | 22.36 | 215.2 | 2,265,229 | 148,180 | 793 | false |

`|22.36 − 22.29| = 0.07` ≤ 0.15 dB。`n_obj=148,180` ≤ 200,000，与第 2 路 147,617 同量级。这一路过线。0.07 dB 和墙钟 215.2 s 对 196.0 s 都不写成复杂度已优。

### 第 4 路：v2 少渲车（失败消融，不是默认）

规则见 `docs/complexity_schedule_v2_design.md`。`score_affects_selection=true`，`actor_budget_ratio=0.5`。背景仍是 `iteration % 4 == 0`。非 force 步按 \(s=M+A\) 取 \(\lceil 0.5 n \rceil\) 辆，并列时 `track_id` 小的在前。`force_full_every=8` 只把候选车补全，不改背景。背景 skip 时全部候选车进入 `densify_exclude`。监督用入选车自己的框。

```bash
python -u train.py --config configs/example/waymo_train_002_schedule_complexity_v2_smoke.yaml \
  2>&1 | tee logs/schedule_smoke_complexity_v2.log
```

87 条周期记录里，背景是否入选与 `iter % 4 == 0` 的不符次数是 0，入选率 0.250。76 条非 force 记录有 `dropped`，被丢掉的车分数不高于入选车；11 条 force 记录里写进 `scores` 的车全部入选。非 force 的 `nonforce_actor_keep=0.569`（奇数 \(n\) 时 \(\lceil 0.5n\rceil/n\) 高于 0.5）。入选率：obj_000 0.989、obj_012 0.947、obj_006 0.705、obj_001 0.209、obj_004 0.154。无 NaN。`n_empty_supervision=936`。

| 设定 | 背景入选 | PSNR@1000 | PSNR@2000 | wall_s | n_bkgd | n_obj | n_empty_supervision | score_affects_selection |
|------|----------|-----------|-----------|--------|--------|-------|---------------------|-------------------------|
| 第 2 路 固定规则 | `iter % 4 == 0` | 21.59 | 22.29 | 196.0 | 2,227,898 | 147,617 | 793 | — |
| v1 parity | 与第 2 路对称差 0 | 21.43 | 22.36 | 215.2 | 2,265,229 | 148,180 | 793 | false |
| v2 ratio=0.5 | 与第 2 路对称差 0 | 20.18 | 21.00 | 178.1 | 1,763,777 | 313,764 | 936 | true |

**未过线。** PSNR@2000 为 21.00，低于 21.29。`n_obj=313,764`，超过 20 万。1000→2000 没有再掉超过 1 dB（20.18→21.00）。分数确实改了名单，但这不是过线。停，不扫权重。墙钟 178.1 s 更短，来自少画了车，不写成优化目标。

### 诊断：v2 把未入选车从光栅里拿掉了

87 条周期记录里，76 步的 `raster_exclude` 含未入选 `obj_*`（其余 11 步是 `force_full`，候选车全部入选）。`raster_exclude` 与 `update_exclude` 是同一份名单。例如 iter 1：`raster_exclude=['background','obj_001','obj_004']`，入选的是 `obj_000,obj_006,obj_012`。

背景 skip 的 65 条记录里，`densify_exclude` 都盖住了本步全部候选车，包括已经入选的高分车，不符次数是 0。高分车没有在 bg-skip 步 densify。点数涨到 31 万，对得上「光栅空洞由入选车在背景入选步补洞」：那些步上未入选车不在光栅里，入选车仍可 densify。v2.1 只把车辆从 `raster_exclude` 拿掉，Adam 与 densify 口径不动，选取公式也不改。

### v2.1：车辆仍光栅，监督和 densify 仍跟 Adam（失败消融，不是默认）

```bash
python -u train.py --config configs/example/waymo_train_002_schedule_complexity_v2_smoke.yaml \
  2>&1 | tee logs/schedule_smoke_complexity_v2_1_adamonly.log
```

87 条记录里 `raster_exclude` 不再含 `obj_*`。每步 `n_raster_actors == n_candidates`，全程 `n_raster_actor_mismatch=0`。非 force 的 76 步都有 `dropped`，`n_adam_actors / n_candidates` 在 0.50 到 0.60，均值 0.570。背景 skip 的 65 步里，入选高分车仍在 `densify_exclude`，不符次数 0。入选率：obj_000 0.989、obj_012 0.943、obj_001 0.199、obj_004 0.155。无 NaN。`n_empty_supervision=937`。背景入选率 0.250，与 `iter % 4 == 0` 一致。

| 设定 | PSNR@1000 | PSNR@2000 | wall_s | n_bkgd | n_obj | n_empty_supervision | score_affects_selection |
|------|-----------|-----------|--------|--------|-------|---------------------|-------------------------|
| 第 2 路 固定规则 | 21.59 | 22.29 | 196.0 | 2,227,898 | 147,617 | 793 | — |
| v1 parity | 21.43 | 22.36 | 215.2 | 2,265,229 | 148,180 | 793 | false |
| v2 低分车不光栅 | 20.18 | 21.00 | 178.1 | 1,763,777 | 313,764 | 936 | true |
| v2.1 低分车仍光栅 | 20.27 | 20.99 | 202.5 | 1,752,542 | 296,643 | 937 | true |

**未过线。** PSNR@2000 为 20.99，低于 21.29。`n_obj=296,643`，仍超过 20 万。把低分车放回光栅没有消掉补洞：背景入选且非 force 的步上，入选车仍可 densify，损失又只盖住入选车的框。停，不扫权重，不把车再从光栅里拿掉。

### v2.2：分数只闸 Adam（过线）

v2.2 不改选取公式、权重、\(G\)、CUDA。光栅仍是全部候选车。Adam 仍只更新按分入选的车。densify 不再读分数：背景未入选时 `densify_exclude` 含 `'background'` 和全部 `obj_list`（含高分已入选车）；背景入选时名单里没有车。监督与第 2 路相同：背景未入选用 `guidance['obj_bound']`，背景入选用整图加 sky。

```bash
python -u train.py --config configs/example/waymo_train_002_schedule_complexity_v2_smoke.yaml \
  2>&1 | tee logs/schedule_smoke_complexity_v2_2_adamonly_parity_supervise.log
```

87 条周期记录：`raster_exclude` 不含 `obj_*`，每步 `n_raster_actors == n_candidates`，`n_raster_actor_mismatch=0`。背景入选与 `iter % 4 == 0` 不符次数 0，入选率 0.250。背景 skip 的 65 步里，`densify_exclude` 盖住全部 `obj_list`，含已入选高分车，不符次数 0；背景入选的 22 步里 `densify_exclude` 不含车。skip 步 `region_loss=True`，入选步整图 `region_frac=1`。非 force 的 76 步都有 `dropped`，`n_adam_actors / n_candidates` 在 0.50 到 0.60，均值 0.570，`nonforce_actor_keep=0.5693`。入选率：obj_000 0.989、obj_012 0.946、obj_006 0.721、obj_001 0.192、obj_004 0.156。无 NaN。`n_empty_supervision=793`，与第 2 路相同。

| 设定 | PSNR@1000 | PSNR@2000 | wall_s | n_bkgd | n_obj | n_empty_supervision | score_affects_selection |
|------|-----------|-----------|--------|--------|-------|---------------------|-------------------------|
| 第 2 路 固定规则 | 21.59 | 22.29 | 196.0 | 2,227,898 | 147,617 | 793 | — |
| v1 parity | 21.43 | 22.36 | 215.2 | 2,265,229 | 148,180 | 793 | false |
| v2 低分车不光栅 | 20.18 | 21.00 | 178.1 | 1,763,777 | 313,764 | 936 | true |
| v2.1 低分车仍光栅 | 20.27 | 20.99 | 202.5 | 1,752,542 | 296,643 | 937 | true |
| v2.2 分数只闸 Adam | 21.61 | 22.35 | 215.7 | 2,265,762 | 141,515 | 793 | true |

**过线。** PSNR@2000 为 22.35，不低于 21.29。`n_obj=141,515`，不超过 20 万。空监督回到 793。分数仍只改变 Adam 名单：高分 track 入选率明显高于低分 track。墙钟 215.7 s 与 v1 parity 的 215.2 s 同量级，不写成复杂度已优。

## 锁定口径（接力用）

当前推荐默认是 **v2.2 接法**（`score_affects_selection=true`）。仓库里 `complexity_enable` 和该标志的默认值仍是 false，官方训练不改。打开该标志时：分数只决定车辆是否 Adam；全部候选车光栅；densify 与第 2 路相同（背景 skip 冻全部 `obj_*`）；背景 skip 的区域监督是全部候选车框；背景入选是 `iter % 4 == 0`。

v2、v2.1 是失败消融，不要当默认。少渲车、按分数缩小 region mask、按分数偏置 densify，会改变监督覆盖或补洞动力学并诱发炸点。这不说明按分数选择 Adam 本身不可行。

日志：`logs/schedule_smoke_complexity_v2_2_adamonly_parity_supervise.log`。`score_affects_selection=true`。非 force 步 Adam 保留比 `nonforce_actor_keep=0.5693`（取整偏高）。高入选率：obj_000 0.989，obj_012 0.946。低入选率：obj_001 0.192，obj_004 0.156。87 条周期记录里每步 `n_raster_actors == n_candidates`，`n_raster_actor_mismatch=0`。背景 skip 的 65 步里 `densify_exclude` 含全部 `obj_*`。

| 设定 | PSNR@1000 | PSNR@2000 | wall_s | n_bkgd | n_obj | n_empty_supervision | score_affects_selection |
|------|-----------|-----------|--------|--------|-------|---------------------|-------------------------|
| 第 2 路固定规则 | 21.59 | 22.29 | 196.0 | 2,227,898 | 147,617 | 793 | — |
| v1 parity | 21.43 | 22.36 | 215.2 | 2,265,229 | 148,180 | 793 | false |
| v2 少渲车（失败消融） | 20.18 | 21.00 | 178.1 | 1,763,777 | 313,764 | 936 | true |
| v2.1 监督/densify 跟 Adam（失败消融） | 20.27 | 20.99 | 202.5 | 1,752,542 | 296,643 | 937 | true |
| v2.2 分数只闸 Adam | 21.61 | 22.35 | 215.7 | 2,265,762 | 141,515 | 793 | true |

**当前结论：** v2.2 过线。PSNR@2000 = 22.35，`n_obj` ≈ 141515，`empty_supervision` = 793，墙钟约 215.7 s。分数分化还在，光栅和空监督已经回到第 2 路。墙钟没有优于第 2 路。更软的分配、更长训练、ratio 单点实验都还没做。规则见 `docs/complexity_schedule_v2_design.md`。

## v3：10k 收益探针（开跑前）

`v3 control = configs/example/waymo_train_002_schedule_mask_freeze_smoke.yaml`

这是已经跑过的第 2 路（A+B + `obj_bound` + 背景 skip 冻全部车辆 densify）。没有 `actor_budget_ratio`，也没有 `score_affects_selection`。v1 parity 那份 yaml 里已经写了 `actor_budget_ratio: 1.0`，所以不拿它当 control。

control 复制为 `configs/example/waymo_train_002_schedule_v3_10k_off.yaml`。相对源文件，除 `iterations: 10000` 外，只加了 `resume: false` 和新的 `exp_name`。源实验目录里已经有 `iteration_2000.pth`，默认 `resume: true` 会从 2000 接着训，而不是从同一初始化跑满 10000 步。treatment 用 `configs/example/waymo_train_002_schedule_v3_10k_v22.yaml`，同样 `resume: false`、`iterations: 10000`。两边 `exp_name` 不同，只为第二跑不删第一跑的输出目录。

`test_iterations` 仍是源文件的 `[1000, 2000]`。不新增 5000 / 10000 评估点。共有 PSNR 只有这两个点；10000 步结束时的 PSNR 不会单独打印。

开跑前 diff（`v3_10k_off.yaml` vs `v3_10k_v22.yaml`）通过。两边 `iterations: 10000`。control 没有 `actor_budget_ratio`，也没有 `score_affects_selection: true`。数值差异只有：

- `exp_name`：`waymo_train_002_sched_v3_10k_off` / `waymo_train_002_sched_v3_10k_v22`（输出路径，不改调度）
- treatment 多出 `complexity_enable: true`、`score_affects_selection: true`、`actor_budget_ratio: 0.5`

`complexity_enable` 必须写在 treatment 上，否则分数开关不会进入 v2.2 路径。其余字段与 control 相同，包括钩子开关；`complexity_enable=true` 时这些钩子被覆盖，走 v2.2。

运行顺序：先 off，再 v22。同一 GPU、同一环境，中间不清缓存。off 开始时间 2026-09-24 20:51:15 +0800，日志写完 21:06:37 +0800。v22 紧接其后，日志写完 21:24:02 +0800。两边都是 `Starting from 0`。

## v3：10k 结果

日志：`logs/schedule_v3_10k_off.log`，`logs/schedule_v3_10k_v22.log`。无 NaN。`test_iterations` 只有 1000 和 2000，所以没有 5000 / 10000 的 PSNR，也不把最终 tqdm 的训练 PSNR 当成评估。

off 仍是第 2 路：`complexity_enable=False`，skip 步 `raster_exclude=['background']`，`region_loss=True`，`freeze_actor_densify=True`。空监督没有 `n_empty_supervision` 字段，按 `empty obj_bound` 行计数为 3948。

v22 仍是 v2.2。`score_affects_selection=True`，`actor_budget_ratio=0.500`，`nonforce_actor_keep=0.5694`。407 条周期记录里 `n_raster_actors == n_candidates`，`n_raster_actor_mismatch=0`，`raster_exclude` 不含 `obj_*`。背景 skip 的 305 步里 `densify_exclude` 含全部 `obj_*`（含已入选高分车），不符次数 0；背景入选的 102 步里名单不含车，且 `region_loss=False`。高入选率：obj_000 0.998，obj_012 0.958。低入选率：obj_001 0.162，obj_004 0.131。`n_empty_supervision=3948`。抽到的周期行里，含 force 步的 Adam/候选比约 0.620；非 force 行在 0.50–0.60。

| 设定 | PSNR@1000 | PSNR@2000 | PSNR@5000 | PSNR@10000 | wall_s | n_bkgd | n_obj | n_empty |
|------|-----------|-----------|-----------|------------|--------|--------|-------|---------|
| v3 off（第 2 路） | 21.54 | 22.81 | — | — | 878.8 | 3,543,128 | 127,265 | 3948 |
| v3 v2.2 | 21.62 | 22.72 | — | — | 999.3 | 3,636,419 | 123,908 | 3948 |

两路 `n_obj` 都低于 400000。空监督步数相同。共有 PSNR：1000 步 v2.2 高 0.07 dB，2000 步低 0.09 dB。墙钟 v2.2 多约 120 s，只作辅助，不单独当成收益。

**判读：10k same-iteration no benefit, but connection remains healthy。** 不包装成 win。不因此加 G、不扫 ratio、不给背景按分。

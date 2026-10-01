# Windows 重连交接（钩子 B 已完成 → 下一步钩子 A）

写给：从 AutoDL Linux 切到 Windows Cursor Remote-SSH 后的下一个 Agent / 自己。  
日期：2026-09-23。当前这台机是 **无卡模式**，不能训、不能渲；改代码可以。要跑钩子 A 实验必须 **有卡开机**。

**先交接，再上钩子 A。不要在没读完本文和 `docs/schedule_mvp_notes.md` 的情况下改 rasterizer。**

默认使用 v2.2 接法。  
不要开启少渲车路径。  
下一版候选为更软的分配方式 / 更长训练 / ratio 单点实验，目前均尚未开工。

---

## 1. 机器 / 项目 / 环境 / example

| 项 | 值 |
|----|----|
| 机器 | AutoDL 容器（上次 hostname 形如 `autodl-container-...`） |
| 项目根 | `/root/autodl-tmp/street_gaussians` |
| 仓库身份 | **ZJU3DV 官方 Street Gaussians**（ECCV 2024），不是 4DGS fork |
| conda | `source /root/miniconda3/etc/profile.d/conda.sh && conda activate street_gaussians` |
| 环境路径 | `/root/miniconda3/envs/street_gaussians` |
| 另有（勿混） | `/root/autodl-tmp/colmap-env`（COLMAP 用，不是训练 env） |

已跑通的 example（50k，原 yaml **未改**）：

```bash
python train.py --config configs/example/waymo_train_002.yaml
python train.py --config configs/example/waymo_train_031.yaml
python render.py --config configs/example/waymo_train_002.yaml mode trajectory
python render.py --config configs/example/waymo_train_031.yaml mode trajectory
```

| 场景 | 数据 | 50k ckpt | 训练日志 | 备注 |
|------|------|----------|----------|------|
| 002 | `data/waymo/training/002`，帧 `[98,198]` | `output/waymo_full_exp/waymo_train_002/trained_model/iteration_50000.pth` | `train_002.log` | train_view PSNR@50000 ≈ **29.94** |
| 031 | `data/waymo/training/031`，帧 `[0,198]` | `output/waymo_full_exp/waymo_train_031/trained_model/iteration_50000.pth` | `train_031.log` | PSNR@50000 ≈ **28.80**；COLMAP 曾修过 `database.db` |

管线摸底（只读、插入点）：`docs/streetgs_training_profile.md`。  
§5 里 CUDA Event「从未 elapsed_time」已过时：钩子 B 在 `train.schedule.log_time=true` 时会打印 fwd+bwd。

---

## 2. Git 远程、pull 习惯、数据路径

### 远程（重要：没有私有 remote）

```
origin  https://github.com/zju3dv/street_gaussians.git
```

本机 **没有** 配置私有 Git 远程。钩子 B / 工程修补 **全部未 commit、未 push**，只活在这块盘上。

当前 `HEAD`：`0924b0c` *Update README.md*（2025-07-04）。  
`origin/main` 比本地 **超前 1**：`c343613 Adopt Project Registration License v1.0`。

### `git pull` 习惯（先别 pull）

此前工作流是：**不 pull、不 commit**，直接在官方 clone 上改。  
Windows 上 **不要** 一上来 `git pull`：会碰到那 1 个 license commit，且可能把未提交改动搞乱。钩子 B 不在 origin 上。

若要备份再 pull：

```bash
cd /root/autodl-tmp/street_gaussians
git status -sb
git stash push -u -m "hookB-before-anything"   # 或先自己 commit 到新分支
# 有私有远程再：git remote add private <url> && git push private HEAD
```

需要私有远程的话，由你另加；本文不编造地址。

### 数据集：在工作区里，但不在 git 里

`.gitignore` 第 4 行是 `data`。物理路径：

```
/root/autodl-tmp/street_gaussians/data/waymo/training/002
/root/autodl-tmp/street_gaussians/data/waymo/training/031
```

yaml 里写 `source_path: ./data/waymo/training/{002,031}`。  
**不是** 仓库外另一块盘。停实例一般保留 `autodl-tmp`；**销毁数据盘会丢数据、ckpt、未提交代码**。

`output/` 同样被 ignore。冒烟复用 COLMAP / 初始化点：

```yaml
data.load_pcd_from: output/waymo_full_exp/waymo_train_002
```

---

## 3. 钩子 B 已做完（最小实体级更新调度）

主张：固定 iteration 预算，降 **background** 的 Adam / densify 频率；`obj_*`、ActorPose、sky 每步 full。  
加速是观察结果，**没改 CUDA kernel**。

### 改了哪些文件

| 文件 | 作用 |
|------|------|
| `lib/config/config.py` | `cfg.train.schedule.*`，默认 `enable=False` |
| `train.py` | `get_update_exclude_list`；exclude 传入 densify / optimizer；日志 |
| `lib/models/street_gaussian_model.py` | exclude 时 `zero_grad` 不 `step`；skip densify accum |
| `configs/example/waymo_train_002_schedule_smoke.yaml` | 冒烟 off |
| `configs/example/waymo_train_002_schedule_on_smoke.yaml` | 冒烟 on，stride=4 |
| `docs/schedule_mvp_notes.md` | 开关、命令、结果表 |

**未改：** `submodules/diff-gaussian-rasterization`、loss、`waymo_train_{002,031}.yaml`。

同工作区还有 **非钩子 B** 的本地修补（训练能跑需要它们）：

- `lib/utils/cfg_utils.py`：`args.mode` 写入 `cfg.mode`（否则 `render.py ... mode trajectory` 不生效）
- `script/waymo/colmap_waymo_full.py`：SIFT GPU 开关

### 开关

```yaml
train:
  schedule:
    enable: false                 # 默认关 = 原官方 optimizer/densify 集合
    bkgd_update_stride: 4         # 仅 enable 时：iteration % stride == 0 才更新 background
    always_update: ["actors", "actor_pose", "sky"]  # 语义说明，代码不读
    log_interval: 100
    log_time: false               # 默认关；不影响数值
```

`enable=true` 且 `iteration % stride != 0` → `exclude_list=['background']`。

### 冒烟命令与结果表

详情：**`docs/schedule_mvp_notes.md`**（以该文件为准）。

```bash
conda activate street_gaussians
python -u train.py --config configs/example/waymo_train_002_schedule_smoke.yaml \
  2>&1 | tee logs/schedule_smoke_off.log
python -u train.py --config configs/example/waymo_train_002_schedule_on_smoke.yaml \
  2>&1 | tee logs/schedule_smoke_on.log
```

002，2000 iter，RTX 4090；PSNR 是 **train_view 5 张**（`split_train: 1` 无 held-out test）：

| 设定 | PSNR@1000 | PSNR@2000 | wall_s | NaN | n_bkgd | n_obj |
|------|-----------|-----------|--------|-----|--------|-------|
| off | 24.30 | **25.56** | 313.7 | 无 | 82万→165万 | 12.7万→13.0万 |
| on, stride=4 | 22.24 | **23.35** | 330.5 | 无 | 82万→219万 | 12.7万→12.8万 |

**预期限制（不是 bug）：** 钩子 B 几乎不降 rasterize 成本，墙钟没有变快（还略慢）。on 跑 background 点更多：`densify_interval=100` 能被 stride=4 整除，densify 触发步没变，accum 样本变少更噪。

---

## 4. 钩子 A 冒烟已跑完

`train.schedule.skip_bkgd_raster` 默认 `false`。为 `true` 且 `iteration % bkgd_update_stride != 0` 时，训练把 `exclude_list=['background']` 传给 `StreetGaussianRenderer.render`，背景不进 rasterize；`obj_*` / ActorPose / sky 仍每步。`enable=false` 且 `skip_bkgd_raster=false` 时回到官方集合。未改 CUDA，未做 TurboGS / DriveSplat。

冒烟 yaml：`configs/example/waymo_train_002_schedule_raster_smoke.yaml`（`enable` 与 `skip_bkgd_raster` 都开，stride=4）。002 / 2000 iter / RTX 4090 已跑完：`wall_s=250.3`，train_view PSNR@2000 **18.90**，无 NaN。跳过步 `n_raster` 约 12 万，fwd+bwd median 88.8 ms；整步 median 114.7 ms。细节在 `docs/schedule_mvp_notes.md`。日志：`logs/schedule_smoke_raster.log`。

```bash
python -u train.py --config configs/example/waymo_train_002_schedule_raster_smoke.yaml \
  2>&1 | tee logs/schedule_smoke_raster.log
```

插入点（见 profile §5）：

- `StreetGaussianModel.parse_camera`（约 `street_gaussian_model.py:219`）
- `StreetGaussianRenderer.render(..., exclude_list=)`（`street_gaussian_renderer.py:89-103`）。训练在 `skip_bkgd_raster=true` 时按 stride 传入 `background`
- 风险：跳过的高斯无梯度、无 densify 统计；天空合成依赖前景 `acc`

论文指标：现在只有短冒烟 + train_view；要写论文需更长 iteration / 真正 test split。

禁止清单（与上次 MVP 相同）：改 CUDA / tile skip、像素误差采相机、深度 LOD densify、新训复杂度网、依赖光流/语义、大规模改 loss、默认开启导致基线不可复现。

---

## 5. 关键日志 / 输出（不要删）

```
docs/streetgs_training_profile.md
docs/schedule_mvp_notes.md
docs/handoff_windows.md          # 本文件

logs/schedule_smoke_off.log
logs/schedule_smoke_on.log
train_002.log
train_031.log
render_002.log
render_031.log

output/waymo_full_exp/waymo_train_002/                 # 50k 基线
output/waymo_full_exp/waymo_train_031/
output/waymo_full_exp/waymo_train_002_sched_off_smoke/
output/waymo_full_exp/waymo_train_002_sched_on_smoke/

StreetGaussians_002_031_结果.zip   # 以前打包的视频/指标，非全 ckpt
```

未跟踪杂项（与调度无关）：`nvdiffrast/`、`submodules/simple-waymo-open-dataset-reader/*.egg-info/`。

---

## 6. Windows 重连后先做这几步（再开新 Agent）

1. SSH 连 **同一台** AutoDL（数据盘还在），**有卡开机** 才能训钩子 A。
2. Cursor 打开 **同一路径**：`/root/autodl-tmp/street_gaussians`（不要重新 clone）。
3. 终端核对应是这份未提交工作：

```bash
cd /root/autodl-tmp/street_gaussians
git status -sb
git log -1 --oneline
# 期望：HEAD = 0924b0c，且能看到 train.py / street_gaussian_model.py / config.py 已修改
# 期望：有 docs/schedule_mvp_notes.md 与 logs/schedule_smoke_*.log
conda activate street_gaussians
# 无卡时 nvidia-smi 会失败，属正常
```

4. 钩子 A 的代码已经在这棵树上（`skip_bkgd_raster`，默认关）。有卡后跑 `configs/example/waymo_train_002_schedule_raster_smoke.yaml`，不要再重写一版。
5. 不要 `git pull`、不要覆盖 `waymo_train_002.yaml`、不要改 `diff-gaussian-rasterization`。

新 Agent 建议开场（可复制）：

> 仓库在 `/root/autodl-tmp/street_gaussians`，conda 是 `street_gaussians`。先读 `docs/handoff_windows.md` 和 `docs/schedule_mvp_notes.md`。钩子 B 已落地（`train.schedule.enable` 默认 false）。现在做钩子 A：stride 跳过 background rasterize，actor/pose/sky 每步进图；默认关闭必须回到原行为。不要改 CUDA，不要做成 TurboGS / DriveSplat。

---

## 7. `git status` 对照（2026-09-23）

已修改：`lib/config/config.py`、`lib/models/street_gaussian_model.py`、`train.py`，以及工程修补 `lib/utils/cfg_utils.py`、`script/waymo/colmap_waymo_full.py`。

未跟踪：上述 smoke yaml、`docs/`、`logs/`、zip、render/train 日志等。

若 Windows 上 `git status` 是干净的 `0924b0c` 且没有 `docs/schedule_mvp_notes.md`，说明开错目录或盘被清过，**不要开始钩子 A**。

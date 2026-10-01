# 复杂度调度 v2.2 锁定口径

仓库开关默认仍是 `complexity_enable=false`、`score_affects_selection=false`，官方训练不变。v1 的 false 路径也不变：选人、光栅、Adam、densify 与固定 A+B+mask+冻 densify 同构，分数只遥测。

当前推荐实验接法是 **v2.2**。它历史上属于 v2 的 score-based selection 路径，但分数的作用已经收窄。打开

```text
score_affects_selection=true
```

时，锁定行为如下。`score_affects_selection=true` 不表示光栅、区域掩码或 densify 可以再读分数。

## 锁定行为

1. \(s = M + A\)（\(w_g = 0\)，\(G\) 固定返回 0）。分数只决定车辆是否进入 Adam，也就是 `update_exclude`。
2. 所有候选车都参与 raster / render。
3. densify 完全不看分数，规则与第 2 路 / v1 相同。
4. 背景未入选时，`densify_exclude` 冻结全部 `obj_*`。
5. 背景未入选时，区域监督覆盖全部候选车框（`guidance['obj_bound']`），不能缩成仅 Adam 入选车。
6. 背景仍固定为 `iteration % 4 == 0`。本版不按分数给背景分配预算。

## 分数的作用边界

记本步候选车为 \(E_{\text{actor}}\)（`start_frame <= frame <= end_frame` 的 `obj_*`），\(n = |E_{\text{actor}}|\)。

\[
s = w_m M + w_a A + w_g G
\]

\(M\) 来自数据集 tracklet 速度，\(A\) 是入选且框非空时的外观残差 EMA。\(G\) 不读深度、点数或尺度。不新训复杂度网络。

分数唯一改变的名单是车辆 Adam：

- `iteration % force_full_every == 0`（默认 8）：本步全部候选车进入 Adam。背景仍只看 stride。
- 其他步：按 \(s\) 从高到低取 \(k\) 辆，未入选的车写入 `update_exclude`。

\[
k = \min\bigl(n,\ \max(1,\ \lceil r \cdot n \rceil)\bigr)
\]

\(r =\) `actor_budget_ratio`。smoke 用 \(r = 0.5\)，不扫这个比例。奇数 \(n\) 时 \(\lceil 0.5 n\rceil / n\) 略高于 0.5，002 上 `nonforce_actor_keep=0.5693`，这是取整，不是要改公式。并列时 `track_id` 小的在前，排序键是 `(-s, track_id, name)`。

sky 与 ActorPose 不进选择集合，每步照常更新。

## 五条规则

| 环节 | `score_affects_selection=true`（v2.2） |
|--|--|
| Adam | `update_exclude = E − selected`。非 force 步只更新前 \(k\) 辆 |
| raster | 候选 `obj_*` 全部光栅。`raster_exclude` 只在背景 skip 时含 `'background'` |
| 监督 | 背景 skip：RGB/LiDAR 用 `guidance['obj_bound']`（全部车框并集），关掉 sky。背景入选：整图 + sky，与第 2 路相同。空掩码不 mean、不反传 |
| densify | 不读分数，也不读车辆是否入选。背景 skip：`densify_exclude` 含 `'background'` 和全部 `obj_list`。背景入选：名单里没有车 |
| 背景 | `iteration % bkgd_update_stride == 0`，smoke 的 stride 是 4。不用信用账户 |

名单用完整模型名，不用前缀 `'obj'`。

## 与 v1 / 第 2 路的关系

第 2 路是固定 A+B + `obj_bound` + 背景 skip 时冻全部车辆 densify。v1 在 `score_affects_selection=false` 时与它对齐（002 上 PSNR@2000 差 0.07 dB）。

v2.2 与这两条的差别只有车辆 Adam：非 force 步按分数留下约一半候选车。光栅、监督覆盖、densify、背景 stride 与它们相同。

## 失败消融（不是默认）

少渲车、按分数缩小 region mask、按分数偏置 densification，会改变监督覆盖或表示补洞动力学，并诱发补洞/炸点。这并不说明「按分数选择 Adam 更新」本身不可行。

### v2：低分车进入 `raster_exclude`

未入选车不进光栅。002，2000 iter：PSNR@2000 = 21.00，`n_obj` = 313,764，`n_empty_supervision` = 936，墙钟 178.1 s。墙钟更短来自少画了车，不是优化目标。日志：`logs/schedule_smoke_complexity_v2.log`。

### v2.1：车辆仍光栅，监督和 densify 仍跟着 Adam

候选车都光栅，但背景入选步上只有未入选车被冻 densify，损失掩码只用入选车的框。PSNR@2000 = 20.99，`n_obj` = 296,643，`n_empty_supervision` = 937，墙钟 202.5 s。日志：`logs/schedule_smoke_complexity_v2_1_adamonly.log`。

`ComplexityState.selection_mask` 是这条消融留下的函数，`train.py` 不调用它。不要把它接回损失。

## v2.2 过线结果

日志：`logs/schedule_smoke_complexity_v2_2_adamonly_parity_supervise.log`。002，2000 iter，5 train views，`actor_budget_ratio=0.5`，`resume=false`。无 NaN。

| 设定 | PSNR@1000 | PSNR@2000 | wall_s | n_bkgd | n_obj | n_empty_supervision | score_affects_selection |
|------|-----------|-----------|--------|--------|-------|---------------------|-------------------------|
| 第 2 路固定规则 | 21.59 | 22.29 | 196.0 | 2,227,898 | 147,617 | 793 | — |
| v1 parity | 21.43 | 22.36 | 215.2 | 2,265,229 | 148,180 | 793 | false |
| v2 少渲车（失败消融） | 20.18 | 21.00 | 178.1 | 1,763,777 | 313,764 | 936 | true |
| v2.1 监督/densify 跟 Adam（失败消融） | 20.27 | 20.99 | 202.5 | 1,752,542 | 296,643 | 937 | true |
| v2.2 分数只闸 Adam | 21.61 | 22.35 | 215.7 | 2,265,762 | 141,515 | 793 | true |

机制证据（87 条周期记录）：

- `score_affects_selection=true`，`n_raster_actor_mismatch=0`，每步 `n_raster_actors == n_candidates`，`raster_exclude` 不含 `obj_*`。
- 非 force 的 76 步都有 `dropped`。`n_adam_actors / n_candidates` 在 0.50 到 0.60，均值 0.570，`nonforce_actor_keep=0.5693`。
- 高入选率：obj_000 0.989，obj_012 0.946。低入选率：obj_001 0.192，obj_004 0.156。
- 背景入选率 0.250，与 `iter % 4 == 0` 不符次数为 0。背景 skip 的 65 步里，`densify_exclude` 含全部 `obj_list`（含已入选高分车）；背景入选的 22 步里名单不含车。
- `n_empty_supervision=793`，与第 2 路相同。

## 仍未解决

- 墙钟 215.7 s，与 v1 parity 同量级，长于第 2 路的 196.0 s。不写成复杂度已经更快。
- 空监督仍有 793/2000 步。这些步跳过图像损失和反传，不能算成少渲染的收益。
- 几何分 \(G\) 仍是 0。背景不按分数分配预算。
- `force_full_every=8` 的步上全部候选车仍会 Adam。
- 更软的分配、更长训练、`actor_budget_ratio` 单点实验都还没做。

## 日志

周期行要能证明分数改了 Adam 名单、但没有改车辆光栅：`score_affects_selection`，`actor_budget_ratio`，`n_candidates`，`n_selected_actors`，`n_raster_actors`，`n_adam_actors`，`dropped`，`densify_exclude`。每步 `n_raster_actors` 应等于 `n_candidates`。结束时打印 `n_empty_supervision`、`bkgd_inclusion_rate`、`nonforce_actor_keep`、`n_raster_actor_mismatch`（应为 0）和各 track 入选率。

## 本版不做

- 不让分数决定背景，不恢复信用账户。
- 不把低分车放进 `raster_exclude`，不把损失掩码收成入选框，不按分数挑选 densify。
- 不计算几何分，不把 \(w_g\) 设成非 0。
- 不改 CUDA，不为墙钟做优化，不扫 \(w_m\)、\(w_a\) 或 `actor_budget_ratio`。
- 不改 `complexity_enable` 的默认 false，不改 `configs/example/waymo_train_002.yaml`。

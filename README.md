# RHPE 手骨骨龄预测工程框架

这是一个融合 `SIMBA` 与 `BoNet` 思想的多模态骨龄预测工程框架，重点是工程化训练、可解释配置和可复现实验，不是两篇论文原始代码的直接复现版本。

## 1. 项目定义

- 任务：预测 RHPE 数据集中的 `Boneage`
- 默认训练目标：`relative age = Boneage - Chronological`
- 最终输出：`final_boneage = predicted_relative_age + Chronological`
- 默认 metadata 模式：`mlp`
- 可选 metadata 模式：`mlp`、`simba_multiplier`、`simba_hybrid`
- 默认运行模式声明：`experiment.mode=enhanced`

默认行为的设计原则是“先保证工程稳定，再提供论文相关变体用于对照实验”。因此默认配置不是论文忠实复现，而是当前工程增强版。

## 2. 两篇论文与参考文件夹

- `SIMBA/` 对应论文：`SIMBA: Specific Identity Markers for Bone Age Assessment`
- `Bonet-master/` 对应论文：`Hand Pose Estimation for Pediatric Bone Age Assessment`

这两个目录保留为参考代码与方法来源说明，不会被当前主训练入口直接 import 或执行。

## 3. Method Relationship

### SIMBA

当前工程真实使用了以下思想：

- metadata 输入：`Male + Chronological`
- relative target
- 可选 `SIMBA` 风格 multiplier / hybrid metadata 分支

当前工程没有直接使用以下内容：

- 原论文完整网络结构
- 原参考工程的原始训练脚本与代码路径

因此更准确的表述是：

> 当前项目受 `SIMBA` 启发并进行了工程实现，而不是 `SIMBA` 原始代码复现。

### BoNet

当前工程真实使用了以下思想：

- ROI
- keypoints
- heatmap
- local patches
- 基于 ROI 的全局裁剪

当前工程没有直接使用以下内容：

- 原始 two-channel Inception backbone
- 原始训练流程与工程组织

因此更准确的表述是：

> 当前项目受 `BoNet` 启发并进行了工程实现，而不是 `BoNet` 原始代码复现。

## 4. 当前默认训练行为

当前默认配置定义在 [configs/default.yaml](/home/lqw/DLBBAP/configs/default.yaml)。

默认主流程是：

- `experiment.mode=enhanced`
- `model.ensemble_mode=ensemble`
- `model.branch_mode=global_local`
- `model.target_mode=relative`
- `model.relative_target_direction=boneage_minus_chronological`
- `model.metadata.mode=mlp`

默认训练目标为：

```text
relative_age = Boneage - Chronological
predicted_relative_age = model(...)
final_boneage = predicted_relative_age + Chronological
```

训练与验证日志会同时输出：

- `relative_mae / relative_mad`
- `final_mae / final_mad`

其中：

- `relative_*` 用来描述模型对相对骨龄偏差项的学习效果
- `final_*` 才是最终骨龄还原后的真实评估指标

兼容性说明：

- 历史字段 `mae / mad` 仍然保留，但它们现在等价于 `final_mae / final_mad`

## 5. 三种模式说明

### `experiment.mode=enhanced`

默认工程增强模式。

- 目标：稳定训练、清晰对照、适合作为论文主方法实现
- 默认 metadata：`mlp`
- `SIMBA` 状态：partial
- `BoNet` 状态：partial

### `experiment.mode=simba`

SIMBA 导向声明模式。

- 用于明确当前实验希望更贴近 `SIMBA` 思想
- 推荐同时使用 `model.target_mode=relative`
- 推荐同时使用 `model.metadata.mode=simba_multiplier` 或 `simba_hybrid`
- 框架仍然是当前工程实现，不会切换成原论文代码

### `experiment.mode=bonet_like`

BoNet 导向声明模式。

- 用于明确当前实验希望更强调 ROI / keypoints / local branch
- 推荐保持 `model.branch_mode=global_local`
- 推荐启用 heatmap 与 local branch
- 框架仍然是当前工程实现，不会切换成原论文代码

启动训练时会打印：

```text
Running mode: enhanced (default)
- SIMBA: partial
- BoNet: partial
```

如果声明模式与实际配置不一致，例如 `mode=simba` 但 metadata 仍是 `mlp`，日志会主动告警。

## 6. 输入与模型结构

当前主模型是一个多模态回归框架：

- 全局灰度图分支
- 全局 ROI heatmap 引导
- 局部 patch / local heatmap / ROI geometry 分支
- metadata 分支
- 最终融合回归头

默认 ensemble 下包含两个子模型：

- `resnet18`
- `efficientnet_b0`

两者分别输出预测后做均值融合。

### 输入模态

当前主流程真实接入的输入包括：

- 灰度图像
- 全局 ROI heatmap
- 局部 patch
- 局部 heatmap
- ROI 几何向量
- `Male`
- `Chronological`

说明：

- 数据层读取的是单通道灰度图
- torchvision backbone 前向时会把单通道复制为 3 通道，以兼容标准主干

## 7. 数据读取与一致性检查

数据默认从 `dataset/` 自动发现：

- `RHPE_train`
- `RHPE_val`
- `RHPE_test`
- `RHPE_Annotations`

工程会建立严格映射：

```text
ID -> image_path -> csv_row -> roi_annotation
```

当前检查项包括：

- 缺失图像
- 缺失 CSV
- 缺失 ROI
- 重复 CSV ID
- 重复图像 ID
- 重复 ROI ID
- 图像不可读

其中重复 ROI 现在会做真实统计；一旦发现重复标注，将直接报错，避免训练落在不可信数据上。

单独检查数据可运行：

```bash
python scripts/inspect_dataset.py --dataset-root dataset
python scripts/inspect_dataset.py --dataset-root dataset --verify-images
```

## 8. 图像归一化

默认不再固定使用 `0.5 / 0.5`。

当前默认行为：

- `data.normalization.source=auto_train_stats`
- 自动统计 train split 的灰度图 `mean/std`
- 默认缓存到 `dataset/train_mean_std.json`
- 运行时把实际使用的归一化信息写入：
  - `image_normalization.json`
  - `config.json`
  - `run_config.json`
  - checkpoint

如果你要手动指定，可以在配置中写：

```yaml
data:
  normalization:
    source: manual
    mean: 0.42
    std: 0.21
```

也可以单独运行统计脚本：

```bash
python compute_train_mean_std.py --image-dir dataset/RHPE_train
```

## 9. 训练入口与主流程

主训练入口：

```bash
python scripts/train.py --config configs/default.yaml
```

真实调用链：

```text
scripts/train.py
-> src/rhpe_boneage/training/runner.py:train_main
-> 数据发现 / 数据集构建 / 模型构建 / 训练 / best checkpoint / 最终 val/test / 报告生成
```

训练开始时会打印 `CONFIG SUMMARY`，至少包含：

- model type
- metadata mode
- input modalities
- target type
- dataset size
- device

## 10. 常用命令

### 标准训练

```bash
python scripts/train.py --config configs/default.yaml
```

### 更贴近 SIMBA 的配置声明

```bash
python scripts/train.py \
  --config configs/default.yaml \
  --set experiment.mode=simba \
  --set model.metadata.mode=simba_hybrid
```

### 更强调 ROI / local branch 的配置声明

```bash
python scripts/train.py \
  --config configs/default.yaml \
  --set experiment.mode=bonet_like
```

### 断点续训

```bash
python scripts/train.py \
  --config configs/default.yaml \
  --set training.resume_checkpoint=outputs/xxx/model/last_checkpoint.pt
```

续训时会在构建 dataset 之前先恢复 checkpoint 中保存的配置与归一化信息，避免续训前后数据处理不一致。

### 验证

```bash
python scripts/validate.py --checkpoint outputs/xxx/model/best_model.pt
```

### 测试

```bash
python scripts/test.py --checkpoint outputs/xxx/model/best_model.pt
```

### 推理

```bash
python scripts/infer.py --checkpoint outputs/xxx/model/best_model.pt
```

## 11. 输出目录结构

每次训练会生成一个独立目录，结构如下：

```text
outputs/
  exp_xxx/
    config.yaml
    config.json
    run_config.json
    effective_params.json
    config_summary.json
    runtime.json
    dataset_report.json
    dataset_summary.json
    dataloader.json
    image_normalization.json
    history.csv
    val_predictions.csv
    val_metrics.json
    test_predictions.csv
    test_metrics.json
    metrics.json
    best_metrics.json
    metrics_summary.csv
    run.log
    model/
      best_model.pt
      last_checkpoint.pt
    plots/
      curves.png
      loss_curve.png
      mae_curve.png
      mad_curve.png
      val_scatter.png
      val_residual.png
      error_histogram_val.png
      ...
```

`metrics.json` / `best_metrics.json` 中会汇总：

- `loss`
- `final_mae`
- `final_mad`
- `relative_mae`
- `relative_mad`
- `relative_age_error_corr`
- `relative_age_error_slope`

如果 test 集没有 `Boneage` 真值，会自动退化为只导出预测和预测分布图。

## 12. 配置与文档一致性约定

当前仓库采用以下约定：

- 文档描述必须以默认配置真实行为为准
- `experiment.mode` 用于声明方法关系和实验定位，不隐藏地改模型
- `model.metadata.mode` 真实决定 metadata 分支实现
- `model.target_mode` 真实决定训练目标
- `run_config.json` 与 `effective_params.json` 记录当次实验的实际运行配置

因此当前默认结论是：

- 默认 metadata 模式是 `mlp`
- `simba_multiplier` / `simba_hybrid` 是可选模式
- 默认目标是 `relative bone age`
- 当前项目是“受论文启发并工程实现”的版本，不是论文原始代码复现

## 13. 目录职责

主流程文件：

- [configs/default.yaml](/home/lqw/DLBBAP/configs/default.yaml)
- [scripts/train.py](/home/lqw/DLBBAP/scripts/train.py)
- [scripts/validate.py](/home/lqw/DLBBAP/scripts/validate.py)
- [scripts/test.py](/home/lqw/DLBBAP/scripts/test.py)
- [scripts/infer.py](/home/lqw/DLBBAP/scripts/infer.py)
- [src/rhpe_boneage/data](/home/lqw/DLBBAP/src/rhpe_boneage/data)
- [src/rhpe_boneage/models](/home/lqw/DLBBAP/src/rhpe_boneage/models)
- [src/rhpe_boneage/training](/home/lqw/DLBBAP/src/rhpe_boneage/training)

辅助工具：

- [scripts/train_ui.py](/home/lqw/DLBBAP/scripts/train_ui.py)
- [scripts/inspect_dataset.py](/home/lqw/DLBBAP/scripts/inspect_dataset.py)
- [scripts/tune.py](/home/lqw/DLBBAP/scripts/tune.py)
- [compute_train_mean_std.py](/home/lqw/DLBBAP/compute_train_mean_std.py)

参考代码：

- [SIMBA](/home/lqw/DLBBAP/SIMBA)
- [Bonet-master](/home/lqw/DLBBAP/Bonet-master)

## 14. 环境与运行信息

依赖清单见 [requirements.txt](/home/lqw/DLBBAP/requirements.txt)。

具体运行环境不要以 README 中的静态文字为准，应以每次运行自动写出的：

- `runtime.json`
- `run.log`

为准。

## 15. 论文写作建议

如果要直接用于论文写作，推荐这样表述：

- 方法章节：描述这是一个融合 `SIMBA` 与 `BoNet` 思想的工程化多模态骨龄回归框架
- 实验章节：把 `experiment.mode`、`model.metadata.mode`、`model.target_mode`、`branch_mode` 作为消融轴
- 复现实验：以 `run_config.json` 和 `effective_params.json` 为准

这样可以避免“我以为在做 A，实际跑的是 B”的描述偏差。

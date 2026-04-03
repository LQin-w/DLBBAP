# RHPE 手骨骨龄预测新工程

一个从头重写的、可直接运行的 PyTorch 多模态骨龄预测项目，面向当前目录下的 RHPE 数据集。项目不兼容旧代码，也不依赖 `Bonet-master` 或 `SIMBA` 的原始工程结构，但充分吸收了两篇论文的核心思想：

- `SIMBA`：显式融合 `Male` 与 `Chronological` 两类 identity markers，并默认采用 `relative bone age` 预测。
- `BoNet`：显式使用 ROI / keypoints 生成的局部信息，不再只依赖整图。

在你补发论文原文后，工程又按论文内容做了一轮复核与优化，新增了几项更贴近论文的方法实现：

- 全局分支加入 `full-image heatmap guidance`
- metadata 支持 `SIMBA` 论文式 `learnable multipliers`
- `relative target` 支持论文原式与工程等价式两种方向
- 评估阶段新增 `relative age vs abs_error` 偏差分析指标

## 1. 项目简介

本项目实现了一个新的多模态骨龄回归框架，联合使用以下信息预测 `Boneage`：

- 手骨 X 光灰度图
- 性别 `Male`
- 真实年龄 `Chronological`
- ROI / keypoints `json`

核心结构是：

- `Global + Local` 双分支
- `ResNet + EfficientNet` 双子模型平均预测
- `CBAM attention`
- metadata fusion
- `direct` / `relative` 两种预测模式
- 标签标准化、AMP、checkpoint、断点续训、Optuna 搜索

## 2. 任务目标

给定 RHPE 数据集中的手骨图像、CSV 元数据、ROI JSON，预测骨龄 `Boneage`。默认推荐 `relative` 模式：

```text
relative_target = Boneage - Chronological
pred_boneage = pred_relative + Chronological
```

这样更符合 `SIMBA` 中“真实年龄作为先验、模型学习偏差项”的思想。

## 3. 数据说明

当前项目默认读取根目录下的 `dataset/`。项目已经针对当前目录实际数据完成过检查，现有结构为：

- `dataset/RHPE_train`
- `dataset/RHPE_val`
- `dataset/RHPE_test`
- `dataset/RHPE_Annotations`

其中：

- `train`: 5491 张图像，CSV 含 `ID, Male, Boneage, Chronological`
- `val`: 713 张图像，CSV 含 `ID, Male, Boneage, Chronological`
- `test`: 79 张图像，CSV 含 `ID, Male, Chronological`
- ROI JSON 为 COCO 风格，包含 `images / annotations / bbox / keypoints / num_keypoints`

## 4. 自动读取逻辑

项目不会手写死你的本地绝对路径，而是通过以下规则自动发现数据：

- 自动寻找包含图像的 split 文件夹，按文件夹名中的 `train / val / test` 关键词识别。
- 自动寻找标注目录中包含 `train / val / test` 关键词的 CSV 与 ROI JSON。
- 自动解析 ROI JSON 中 `images.file_name` 与 `annotations.image_id` 的对应关系。
- 自动根据实际文件名长度确定 `ID` 的零填充宽度，确保 `csv ID` 与图像名严格一致。

可以单独执行：

```bash
python scripts/inspect_dataset.py --dataset-root dataset
```

## 5. 图像 / CSV / ROI 对应关系

工程内部建立的是严格映射：

```text
ID -> image_path -> csv_row -> roi_annotation
```

启动训练、验证、测试、推理时都会自动执行检查并输出日志，检查项包括：

- 缺失图像
- 缺失 CSV 记录
- 缺失 ROI JSON
- 重复 ID
- 无法读取的图像
- 图像名与 ID 不匹配

当前目录实测结果是三套 split 都严格匹配，无缺失、无重复、无错配。

## 6. 模型整体结构

整体由两个并行子模型组成：

- `ResNet` 子模型
- `EfficientNet` 子模型

每个子模型都独立接收完整输入信息：

- 全局图像 `global image`
- 全局 ROI 热图 `global heatmap`
- 局部 ROI 信息 `local patches + heatmaps + ROI geometry`
- metadata `Male + Chronological`

两个子模型分别输出骨龄预测，最终取平均值作为最终输出：

```text
pred_final = (pred_resnet + pred_efficientnet) / 2
```

同时也支持：

- `ensemble_mode=resnet`
- `ensemble_mode=efficientnet`
- `ensemble_mode=ensemble`

方便做消融实验。

## 7. Global + Local

### Global 分支

输入为基于手部 `bbox` 裁剪后的全局灰度图，以及对应的全局 anatomical heatmap。这里和论文更一致：先用 ROI 定位手，再让全局特征提取受到 heatmap 引导，而不是只看原始整图。

由于 `ResNet` / `EfficientNet` 的 torchvision 实现默认使用 3 通道输入，代码会在模型内部把单通道灰度图安全复制为 3 通道。

这样做的原因是：

- 兼容标准 backbone 结构
- 不破坏灰度图像的强度信息
- 比直接修改 backbone 第一层更稳，工程风险更低

### Local 分支

Local 分支不会把 ROI 只读不算，而是实际参与建模。当前实现支持三种模式：

- `patch`：只使用 keypoint-centered 局部 patch
- `heatmap`：只使用局部热图 patch
- `patch_heatmap`：将局部图像 patch 与对应热图 patch 拼接后编码

Local 分支流程：

1. 从 ROI JSON 中读取 `bbox` 与 `17` 个 keypoints
2. 根据 keypoints 生成高斯 heatmap
3. 默认先根据 `bbox` 做全局裁剪并统一 resize
4. 在增强后的图像上，以 keypoint 为中心裁剪多个局部 patch
5. 使用局部 CNN 编码每个 patch
6. 使用 attention pooling 聚合多个局部 patch
7. 再与 ROI 几何向量编码结果融合

这部分对应 `BoNet` 的本质思想：局部骨骼细节对 RHPE 骨龄预测有效，不能只看整图。

## 8. CBAM

工程内实现了标准 `CBAM`，并支持配置开关：

- `model.cbam.enabled`
- `model.cbam.global_branch`
- `model.cbam.local_branch`

当前默认：

- Global 分支启用 CBAM
- Local patch 编码器启用 CBAM

这样 CBAM 不是形式性插入，而是真正用于增强全局与局部视觉特征。

## 9. Metadata Fusion

`Male` 与 `Chronological` 被显式输入模型，而不是只作为日志：

- 支持 `SIMBA` 论文式独立可学习 multiplier
- 同时支持 MLP 投影
- 默认使用 `simba_hybrid`，即 `multiplier + MLP` 联合融合
- metadata 用于调制视觉特征并参与最终回归头

这对应 `SIMBA` 的 identity markers 思想。

## 10. Relative Bone Age

支持两种模式：

- `direct`：直接预测 `Boneage`
- `relative`：预测 `Boneage - Chronological`

默认使用 `relative`。原因：

- 更符合 `SIMBA` 的核心方法
- `Chronological` 作为先验直接进入模型
- 模型学习“生理年龄偏差”通常更稳定

切换方式：

```yaml
model:
  target_mode: relative   # or direct
```

论文中 `SIMBA` 写法是：

```text
rb = Chronological - Boneage
```

而很多工程实现会写成：

```text
rb = Boneage - Chronological
```

两者只差一个符号，只要训练和反标准化保持一致即可。本工程支持两种方向：

```yaml
model:
  relative_target_direction: boneage_minus_chronological
  # 或 chronological_minus_boneage
```

## 11. 标签标准化

项目对训练目标做标准化，并且严格只使用训练集统计量：

- `direct` 模式标准化 `Boneage`
- `relative` 模式标准化 `Boneage - Chronological`

验证集、测试集、推理阶段都复用训练集保存的 mean/std。checkpoint 中会保存：

- target normalizer
- chronological normalizer

## 12. 数据增强

增强策略面向灰度 X 光，不做颜色增强。默认使用的都是保守增强：

- affine
- small rotation
- mild scale / translation
- optional blur
- optional noise
- optional horizontal flip

默认不启用 `horizontal flip`，因为手骨影像左右镜像是否合理取决于实验设定。若要做方向鲁棒性实验，可以在配置中打开。

## 13. 损失函数

支持：

- `smoothl1`，默认
- `l1`
- `mse`

切换方式：

```yaml
training:
  loss: smoothl1
```

## 14. 自动超参数搜索

项目提供 `Optuna` 搜索入口：

```bash
python scripts/tune.py --config configs/default.yaml
```

默认会搜索：

- learning rate
- batch size
- optimizer
- weight decay
- scheduler
- dropout
- CBAM 开关
- `direct / relative`
- metadata hidden dim
- local feature dim
- blur / noise / flip 开关

结果自动保存到：

- `optuna_trials.csv`
- `optuna_best.json`

## 15. 环境依赖

核心依赖见 [requirements.txt](/home/lqw/DLBBAP/requirements.txt)。

当前环境中已检查到：

- Python 3.12.12
- torch 2.10.0+cu130
- torchvision 0.25.0+cu130
- torchaudio 2.10.0+cu130
- numpy 2.4.2
- pandas 3.0.0
- scikit-learn 1.8.0
- pillow 12.1.0
- opencv-python 4.13.0
- albumentations 2.0.8
- matplotlib 3.10.8
- optuna 4.8.0
- tqdm 4.67.3
- pyyaml 6.0.3

## 16. CUDA 与设备说明

代码当前默认固定请求 `cuda:0`：

- 默认配置为 `runtime.device: cuda:0`
- 若当前环境无法访问该 GPU，会直接报错，避免训练静默跑到 CPU
- 如果你确实要改回 CPU，可显式设置 `--set runtime.device=cpu`

当前这台机器的实测情况是：

- torch 构建自 `CUDA 13.0`
- 当前运行时 `cuda available = False`
- `nvidia-smi` 被操作系统阻止，当前会话无法直接访问 GPU

因此本工程代码路径以 CUDA 为主；如果当前会话看不到 GPU，需要先修复运行环境，再启动训练。

## 17. 安装步骤

```bash
python -m pip install -r requirements.txt
```

如果你已经在现有虚拟环境中安装好依赖，也可以直接运行脚本。

## 18. 训练命令

标准训练：

```bash
python scripts/train.py --config configs/default.yaml
```

显式指定设备：

```bash
python scripts/train.py --config configs/default.yaml --set runtime.device=cuda:0
```

常见覆盖示例：

```bash
python scripts/train.py \
  --config configs/default.yaml \
  --set model.target_mode=direct \
  --set model.ensemble_mode=resnet \
  --set model.branch_mode=global_only
```

断点续训：

```bash
python scripts/train.py \
  --config configs/default.yaml \
  --set training.resume_checkpoint=outputs/xxx/last_checkpoint.pt
```

## 19. 验证命令

```bash
python scripts/validate.py --checkpoint outputs/xxx/best_model.pt
```

## 20. 测试命令

```bash
python scripts/test.py --checkpoint outputs/xxx/best_model.pt
```

注意：当前 RHPE `test` CSV 没有 `Boneage`，因此默认只导出预测，`MAE / MAD` 会是 `None`。

## 21. 推理命令

对自动发现的测试集推理：

```bash
python scripts/infer.py --checkpoint outputs/xxx/best_model.pt
```

对任意 RHPE 风格数据推理：

```bash
python scripts/infer.py \
  --checkpoint outputs/xxx/best_model.pt \
  --image-dir your_images \
  --csv-path your_meta.csv \
  --roi-json-path your_roi.json
```

## 22. 输出文件

每次运行都会在 `outputs/` 下生成一个独立目录，典型内容包括：

- `config.yaml`
- `runtime.json`
- `dataset_report.json`
- `dataloader.json`
- `run.log`
- `history.csv`
- `curves.png`
- `best_model.pt`
- `last_checkpoint.pt`
- `val_predictions.csv`
- `val_metrics.json`
- `test_predictions.csv`
- `test_metrics.json`

预测 CSV 至少包含：

- `ID`
- `gt_boneage`
- `pred_boneage`
- `abs_error`
- `sex`
- `chronological`
- `gt_relative_boneage`
- `pred_relative_boneage`

若 split 有真值，指标 JSON 还会额外保存论文式验证信息：

- `relative_age_error_corr`
- `relative_age_error_slope`

用于检查模型是否对 relative age 有明显偏置。

## 23. 如何复现实验

### 相同设置复现

```bash
python scripts/train.py --config configs/default.yaml
```

### 对照实验

只需要覆盖配置即可：

- `direct vs relative`

```bash
--set model.target_mode=direct
--set model.target_mode=relative
```

- `global only vs local only vs global+local`

```bash
--set model.branch_mode=global_only
--set model.branch_mode=local_only
--set model.branch_mode=global_local
```

- `metadata off vs on`

```bash
--set model.metadata.enabled=false
--set model.metadata.enabled=true
```

- `CBAM off vs on`

```bash
--set model.cbam.enabled=false
--set model.cbam.enabled=true
```

- `ResNet only vs EfficientNet only vs ensemble`

```bash
--set model.ensemble_mode=resnet
--set model.ensemble_mode=efficientnet
--set model.ensemble_mode=ensemble
```

## 24. 与论文思想的对应关系

### 吸收自 SIMBA

- `Male` 与 `Chronological` 作为 identity markers 显式入模
- metadata 与视觉特征进行融合，而不是只做后处理
- 支持论文原式的 `learnable multipliers`
- 默认采用 `relative bone age` 预测

### 吸收自 BoNet

- ROI / anatomical local information 真正参与建模
- 使用 keypoints 生成 heatmap
- 使用 `bbox` 对全局手区做裁剪
- 用全局 heatmap 引导全局特征提取
- 使用局部 patch 对细粒度骨骼区域进行编码
- 将全局图像特征与局部 ROI 特征联合回归骨龄

### 工程化替代

本工程没有机械照抄原仓库的 Inception + Horovod 结构，而是做了更适合当前环境的新实现：

- 用 `ResNet + EfficientNet` 替代旧版 backbone
- 用 `heatmap-guided backbone + attention pooling + ROI geometry encoding` 聚合局部信息
- 用统一配置与单机训练流程替代旧工程中的硬编码路径与分布式脚本

## 25. 常见错误排查

### 1. CUDA 不可用

现象：

- 日志显示 `cuda_available=False`

原因：

- 当前机器没有可用 GPU
- 驱动不可见
- WSL / 容器环境屏蔽了 GPU

处理：

- 现在默认会直接报错，不再静默回退 CPU
- 若要真正使用 GPU，请先确认 `torch.cuda.is_available()` 为 `True`
- 若临时只想在 CPU 跑，请显式传入 `--set runtime.device=cpu`

### 2. 测试集没有指标

原因：

- 当前 RHPE `test` CSV 没有 `Boneage`

处理：

- 这是数据本身决定的
- 工程会导出预测 CSV，但不会伪造 `MAE / MAD`

### 3. 预训练权重下载失败

原因：

- 当前环境网络受限

处理：

- 默认 `model.pretrained=false`
- 若开启后下载失败，代码会回退到随机初始化

### 4. matplotlib 缓存目录报错

工程已在运行时自动写入本地可写缓存目录；若外部环境还强制设置了不可写的 `MPLCONFIGDIR`，请手动清理该环境变量。

### 5. Albumentations 联网提示

工程已关闭其自动在线版本检查。若仍看到旧进程残留提示，可以忽略，不影响训练。

## 26. 目录结构

```text
configs/
  default.yaml
scripts/
  inspect_dataset.py
  train.py
  validate.py
  test.py
  infer.py
  tune.py
src/rhpe_boneage/
  config.py
  data/
  models/
  training/
  utils/
outputs/
README.md
requirements.txt
```

## 27. 已完成的本地自检

本工程已经在当前机器上完成以下检查：

- `import` 与 `py_compile` 通过
- `scripts/inspect_dataset.py` 可读出 train/val/test 严格映射
- 单样本前向通过
- 极小规模 `train.py` smoke test 通过
- `validate.py` 通过
- `test.py` 通过
- `infer.py` 通过
- `tune.py` 1 trial smoke test 通过

如果你直接开始训练，建议优先使用默认配置，然后根据显存和实验需求再调 `input_size / batch_size / ensemble_mode`。

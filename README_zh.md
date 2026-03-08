# 视网膜图像分析探索项目（Retinal Image Analysis Exploratory Project）

本仓库是一个**探索型 demo 项目**，聚焦于视网膜图像分析。  
包含两条独立流程：

1. **视网膜图像分类**（眼底图像类别预测）
2. **视网膜血管分割**（血管掩膜预测 + 基础生物标志物提取）

项目目标是提供从数据到训练、测试与可视化的完整 pipeline 示例，而非追求 SOTA 指标。

---

## 1）仓库结构说明

```text
Retinal_Image_Analysis/
├── classification/
│   ├── train_model.py                    # 分类模型训练（SqueezeNet 迁移学习）
│   ├── predict.py                        # 单张图像推理
│   ├── sample_data.py                    # 从原始分类数据集中随机抽样构建小样本数据
│   ├── utils.py                          # 预处理与类别名称工具函数
│   ├── retinal_classification_demo.ipynb # 分类可视化 notebook
│   └── outputs/                          # 训练权重与 TensorBoard 日志
│
├── segmentation/
│   ├── train_model.py                    # 分割模型训练（U-Net）
│   ├── evaluate.py                       # 测试评估 + 可视化 + 生物标志物计算
│   ├── model.py                          # U-Net 模型构建
│   ├── data_loader.py                    # DRIVE 数据读取与预处理
│   ├── utils.py                          # Dice/IoU、生物标志物、绘图工具
│   ├── retinal_seg_demo.ipynb            # 分割可视化 notebook
│   └── outputs/                          # 训练结果与测试可视化输出
│
├── requirements.txt                      # Python 依赖
└── README.md                             # 英文说明
```

---

## 2）环境与安装

### 建议环境
- Python 3.9+
- PyTorch + torchvision
- segmentation_models_pytorch
- scikit-image
- matplotlib
- jupyter / notebook

安装方式：

```bash
pip install -r requirements.txt
```

> 说明：`requirements.txt` 为完整环境导出（包含较多 notebook 相关包）。若仅运行主流程，可按核心依赖创建精简环境。

---

## 3）数据集与数据处理

### 分类任务数据（FundusImage1000）
- **来源**：Kaggle 数据集页面  
  https://www.kaggle.com/datasets/linchundan/fundusimage1000
- **规模**：1000 张眼底图像，39 个疾病/状态类别。
- **采集背景**：数据来自中国广东汕头 Joint Shantou International Eye Centre（JSIEC）；该 1000 张数据是总计 209,494 张眼底图像中的子集。
- **项目中的使用方式**：`sample_data.py` 会在 `classification/datasets/sampled_images` 中构建小规模 demo 子集（默认每类 3 张）。
- **引用信息**：
  - "Automatic detection of 39 fundus diseases and conditions in retinal photographs using deep neural networks"  
    https://www.nature.com/articles/s41467-021-25138-w

### 分割任务数据（DRIVE: Digital Retinal Images for Vessel Extraction）
- **来源**：
  - Kaggle 镜像：https://www.kaggle.com/datasets/zionfuo/drive2004
  - 官方网站：https://drive.grand-challenge.org/
- **规模**：
  - 总计 40 张彩色眼底图像。
  - 划分为 20 张训练集 + 20 张测试集。
  - 临床构成：33 张未见明显糖网病征象，7 张为轻度早期糖网病征象。
- **图像属性**：
  - 采集设备/视野：Canon CR5 非散瞳 3CCD 相机，45° 视场角（FOV）。
  - 分辨率：768×584，每个颜色通道 8 bit。
  - 圆形 FOV（直径约 540 像素），每张图都提供 FOV mask。
- **标注与评估相关信息**：
  - 训练集提供人工血管标注（`1st_manual`）和 mask。
  - 本项目按 DRIVE 目录组织：`training/images`, `training/1st_manual`, `training/mask`, `test/images`, `test/1st_manual`, `test/mask`。
  - 在本仓库中，`train_model.py` 还会把训练集进一步切分为 train/validation。
- **使用该数据集的原因**：DRIVE 是经典的视网膜血管分割基准数据集，支持血管形态学分析（长度、宽度、迂曲度、分叉模式等），常用于眼科与心血管相关研究。

> 该项目强调“可运行流程演示”，数据与训练设置均偏轻量。

---

## 4）模型与方法简介

### A. 分类流程
- **模型**：`torchvision.models.squeezenet1_1(pretrained=True)`
- **训练策略**：
  - 冻结预训练特征层
  - 替换分类头以适配实际类别数
  - 使用交叉熵，仅训练分类头参数
- **预处理**：
  - Train：随机裁剪 + 随机翻转 + ImageNet 归一化
  - Val/Test：Resize + CenterCrop + ImageNet 归一化

### B. 分割流程
- **模型**：`segmentation_models_pytorch` 的 U-Net
- **Encoder**：ResNet18（ImageNet 预训练）
- **输出**：单通道血管概率图（sigmoid）
- **损失函数**：Dice Loss + BCE Loss
- **评估指标**：Dice、IoU
- **额外特征提取**：
  - Vessel density（FOV 内血管像素占比）
  - Skeleton length（骨架化后中心线长度）

---

## 5）执行流程（How to Run）

### 分类（classification）
1. （可选）先抽样构建 demo 数据：
   ```bash
   cd classification
   python sample_data.py
   ```
2. 训练：
   ```bash
   python train_model.py
   ```
   - 在 `classification/outputs/时间戳/` 下保存模型与日志
3. 单图推理：
   ```bash
   python predict.py --image path/to/image.jpg
   ```
   - 不指定 `--model` 时，自动使用最新模型
4. 可视化：
   - 打开 `classification/retinal_classification_demo.ipynb`

### 分割（segmentation）
1. 训练：
   ```bash
   cd segmentation
   python train_model.py
   ```
   - 按验证集 Dice 保存最佳模型
2. 测试评估：
   ```bash
   python evaluate.py
   ```
   - 不传模型路径时，自动使用最新 checkpoint
   - 输出 Dice/IoU、血管密度、骨架长度
   - 并在 `test_results/` 保存可视化对比图
3. 可视化：
   - 打开 `segmentation/retinal_seg_demo.ipynb`

---

## 6）项目亮点与学习收获

- 完成了分类 + 分割两条视网膜分析 pipeline 的搭建。
- 覆盖了从数据处理、训练、评估到可视化的完整工程流程。
- 在分割任务中加入了基础微血管特征提取（density / skeleton length）。
- 通过两个 notebook 提供结果可视化，便于直观查看分类与分割结果。

---

## 7）免责声明（Disclaimer）

- 本仓库为学习用途的 exploratory demo。
- 训练轮次和数据规模较小，结果不代表模型上限或临床结论。
- 项目重点在于流程与方法实践，而非刷指标。

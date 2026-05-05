# CIFAR-10 图像分类课程项目

本项目使用 CIFAR-10 数据集完成 10 类彩色图像分类任务，最终只保留一个经过优化的 `CNN` 模型用于提交。项目覆盖数据预处理、模型训练、测试评估、结果可视化和报告导出。

数据集官网：
`https://www.cs.toronto.edu/~kriz/cifar.html`

## 1. 项目特点

- 只保留单一 `CNN` 模型，结构清晰，适合课程作业提交。
- CNN 进行了增强，加入残差卷积块、SE 通道注意力和更稳定的训练配置。
- 支持 CIFAR-10 自动下载，也兼容本地已有原始批文件。
- 自动保存最佳权重、分类报告、混淆矩阵、训练曲线和实验总结。
- 提供 Markdown 报告模板和 PDF 导出脚本。

## 2. 目录结构

```text
PyCharm/
├─ main.py                 # 主训练入口
├─ data_loader.py          # 数据读取、预处理、划分
├─ models.py               # 优化版 CNN 模型
├─ trainer.py              # 训练与评估逻辑
├─ utils.py                # 绘图与结果保存工具
├─ advanced_analysis.py    # 生成实验文字分析
├─ export_report_pdf.py    # 导出 PDF 报告
├─ test_project.py         # 冒烟测试
├─ requirements.txt        # 依赖列表
├─ 项目报告.md              # 报告模板
├─ 项目报告.pdf             # 已导出的 PDF 报告
├─ data/                   # CIFAR-10 数据目录
└─ outputs/                # 训练输出目录
```

## 3. 环境安装

```powershell
.\.venv\Scripts\pip.exe install -r requirements.txt
```

## 4. 运行方式

正式训练：

```powershell
.\.venv\Scripts\python.exe main.py --epochs 30 --batch-size 128
```

快速测试：

```powershell
.\.venv\Scripts\python.exe test_project.py
```

小规模演示：

```powershell
.\.venv\Scripts\python.exe main.py --epochs 2 --subset-size 2000
```

导出 PDF 报告：

```powershell
.\.venv\Scripts\python.exe export_report_pdf.py
```

## 5. 优化版 CNN 模型说明

本项目最终采用一个强化后的卷积神经网络作为唯一模型，核心设计如下：

- 使用卷积 stem 提取初始低层特征。
- 使用 3 个残差卷积块增强特征表达能力。
- 每个残差块内部加入 `SEBlock` 做通道注意力建模。
- 激活函数使用 `GELU`，比普通 `ReLU` 更平滑。
- 使用 `AdaptiveAvgPool` 压缩空间特征并连接分类头。
- 使用 `Dropout`、`Weight Decay`、`Label Smoothing` 抑制过拟合。

## 6. 训练策略

- 训练集与验证集按 `9:1` 划分。
- 数据增强使用随机裁剪、随机水平翻转和 `AutoAugment(CIFAR10)`。
- 优化器使用 `AdamW`。
- 学习率调度使用 `Cosine Annealing`。
- 启用梯度裁剪和早停策略。

## 7. 输出结果

每次运行会在 `outputs/时间戳/` 下生成：

- `results_summary.csv`
- `results_summary.json`
- `checkpoints/cnn_best.pt`
- `figures/training_curves.png`
- `figures/cnn_confusion_matrix.png`
- `reports/cnn_classification_report.txt`
- `reports/experiment_summary.txt`
- `reports/advanced_analysis.txt`

## 8. 报告建议

报告里重点写这几部分即可：

- 数据预处理流程
- 优化版 CNN 的结构设计
- 训练参数与调参思路
- 测试准确率、混淆矩阵和分类报告
- 易混淆类别及原因分析

## 9. 提交建议

建议最终提交：

- 项目代码
- 本 `README.md`
- `项目报告.pdf`
- 一次正式训练得到的 `outputs/` 结果图表

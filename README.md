# JPEG 压缩与频段保留对图像分类准确率的影响

本项目用固定的预训练模型比较三种条件下的图像分类准确率。归档结果包括完整测试集的统计、符合条件的样本记录和报告用图。

| 实验 | 测试数据 | 模型 | 主要结果 |
| --- | --- | --- | --- |
| 分块 DCT 频段保留 | CIFAR‑10 测试集，10,000 张 | CIFAR‑10 ResNet‑18 | 当 `r` 从 2 增至 6，低频重建准确率从 25.42% 升至 90.12%，高频重建准确率从 73.60% 降至 15.16%。 |
| 标准 JPEG 退化 | CIFAR‑10‑C JPEG 子集，每级 10,000 张 | 同一 CIFAR‑10 ResNet‑18 | 准确率从原图的 94.98% 降至退化等级 5 的 70.87%。 |
| JPEG 重编码 | Imagenette v2 320 px 验证集，3,925 张 | ImageNet‑1K ResNet‑18 和 ViT‑B/16 | `Q=10` 时，两个模型分别从 78.17% 和 91.41% 降至 63.64% 和 82.52%。 |

准确率指 Top‑1 准确率，即模型最高分的预测与真实标签一致的样本比例。三组实验均只做推理。

## 本地环境

在仓库根目录运行。`requirements.txt` 记录本地实验使用的依赖版本，Python 版本为 3.10.11。建议先创建并激活独立虚拟环境，再安装依赖。

```sh
python -m pip install -r requirements.txt
```

前两组实验支持 CPU 和 CUDA，可用 `--device cpu` 或 `--device cuda` 指定设备。Imagenette 评估脚本需要可用的 CUDA，使用 Kaggle 时沿用其 GPU 环境自带的 PyTorch 和 TorchVision。

## 1 CIFAR‑10 分块 DCT

脚本在 RGB 三通道上分别执行 8×8 分块 DCT。令 `u`、`v` 为块内的垂直与水平频率索引，低频组保留 `sqrt(u² + v²) <= r` 的系数，高频组保留其余系数。两组各自逆变换后输入同一个分类模型。

```sh
python cifar10_dct_preview.py --r 3 --output outputs/cifar10_dct/comparison_r3.png
python cifar10_dct_eval.py --r 3 --output outputs/cifar10_dct_eval/r3
```

完整评估分别运行 `r=2、3、4、5、6`。每次评估的 `--output` 应为新目录或空目录。首次运行会下载 [CIFAR‑10](https://www.cs.toronto.edu/~kriz/cifar.html) 和 [CIFAR‑10 ResNet‑18 权重](https://huggingface.co/edadaltocg/resnet18_cifar10)。默认缓存位置分别为 `datasets/` 和 `models/resnet18_cifar10.bin`，脚本会校验权重的 SHA‑256。

[归档结果](results/cifar10_dct_eval)含各半径的 `summary.json`、`matches.csv` 和报告主图。`matches.csv` 记录原图和高频图均预测正确且低频图预测错误的样本。主图下半部分比较低频与高频重建中只有一组预测正确的比例，分母为全部 10,000 张测试图。`reconstructions/` 收录 `r=0` 到 `6` 的重建图，`r6/` 收录一页预测示例。

## 2 CIFAR‑10‑C 标准 JPEG 退化

先运行实验 1 或将原始 CIFAR‑10 解包到 `datasets/cifar-10-batches-py/`。从 [CIFAR‑10‑C 发布页](https://zenodo.org/records/2535967) 下载数据，并将 `jpeg_compression.npy` 和 `labels.npy` 放到 `datasets/CIFAR-10-C/`。脚本按基准数据定义的五个退化等级读取同一批图片，并校验标签和顺序。

```sh
python cifar10c_jpeg_eval.py --output outputs/cifar10c_jpeg_eval
```

[归档结果](results/cifar10c_jpeg_eval)含 `summary.json`、`flipped_samples.csv`、准确率曲线和样例对照图。明细记录原图预测正确且至少一个退化等级预测错误的图片编号，以及对应的退化等级。

## 3 Imagenette JPEG 重编码

本实验使用 [Imagenette v2 320 px](https://github.com/fastai/imagenette) 验证集，以及 TorchVision 的 [ResNet‑18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html) 和 [ViT‑B/16](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vit_b_16.html) 的 `IMAGENET1K_V1` 权重。基线使用数据集提供的图像解码后的像素。各质量档 `Q=95、75、50、25、10` 都从同一份解码图像独立重编码，色度采样固定为 4:2:0。随后统一缩放和裁剪至 224×224 输入。两个模型保留 ImageNet 的 1,000 类输出，验证集的真实标签来自其中 10 类。

在 Kaggle 启用 GPU 和 Internet，把 `imagenette_jpeg_eval.py` 放到可执行路径后运行。数据和权重会在云端加载。已有挂载数据时，可加 `--val-dir` 指向 Imagenette 320 px 的 `val` 目录。

```sh
python imagenette_jpeg_eval.py --output /kaggle/working/imagenette_jpeg_top1.csv
```

在 Notebook 单元格中运行时，可用 `!python`，并将脚本名换成实际挂载路径。只预览图像差异时，可单独运行 `python imagenette_jpeg_preview.py`，预览脚本仅需 CPU。

[归档结果](results/imagenette_jpeg_eval)含以下文件：

- `imagenette_jpeg_top1.csv` 包含两个模型在六种条件下的准确率、正确数和相对基线的百分点变化。
- `imagenette_jpeg_top1_flips.csv` 按模型和质量档记录所有由对变错的图片。单张图片可对应多条记录。
- `imagenette_jpeg_top1.png` 是报告所用的准确率折线图。`flip_previews/` 包含 10 张错判对照图。评估脚本默认每个模型和质量档选至多 5 个样本，将对照 PNG 打包成 ZIP。加 `--examples-per-condition 0` 可为所有记录生成对照图。

Imagenette v2 的验证集与 ImageNet‑1K 的训练集有图像来源重叠。这里的结果衡量额外 JPEG 重编码对这批图的影响。DCT 半径 `r`、CIFAR‑10‑C 退化等级和 JPEG 质量 `Q` 分别控制不同操作，应按各自的实验条件解读。

## 结果与运行目录

`results/` 存放报告所用的归档结果。重新运行前两组实验时，建议写入 `outputs/`，方便与归档结果比较。数据、下载的权重和运行时文件由各脚本按需获取或在本地生成。

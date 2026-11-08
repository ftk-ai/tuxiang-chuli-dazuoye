"""固定 ResNet-18，比较 CIFAR-10 原图与 CIFAR-10-C 的五个 JPEG 退化等级。

先从 https://zenodo.org/records/2535967 下载 CIFAR-10-C.tar。
把其中 jpeg_compression.npy、labels.npy 放进 datasets/CIFAR-10-C。

全量评估：python cifar10c_jpeg_eval.py
快速看指定样本：python cifar10c_jpeg_eval.py --preview-only --indices 4 5 6
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torchvision.datasets import CIFAR10

from cifar10_dct_eval import MEAN, MODEL_ID, ROOT, STD, WEIGHTS_SHA256, load_model

matplotlib.use("Agg")
from matplotlib import pyplot as plt


SEVERITIES = range(1, 6)
VIEW_NAMES = ("Original", *(f"JPEG severity {level}" for level in SEVERITIES))


def load_jpeg_levels(directory, targets):
    """按官方存储顺序切成五组，并核对各组标签与原测试集一致。"""
    n = len(targets)
    pixels = np.load(directory / "jpeg_compression.npy", mmap_mode="r", allow_pickle=False)
    labels = np.load(directory / "labels.npy", allow_pickle=False)
    expected_shape = (len(SEVERITIES) * n, 32, 32, 3)
    if pixels.shape != expected_shape or pixels.dtype != np.uint8:
        raise ValueError(f"JPEG 数组应为 uint8 {expected_shape}，实际为 {pixels.dtype} {pixels.shape}")
    if labels.shape not in ((n,), (len(SEVERITIES) * n,)):
        raise ValueError(f"labels.npy 的形状不正确：{labels.shape}")
    if not np.all(labels.reshape(-1, n) == targets):
        raise ValueError("CIFAR-10-C 标签与 CIFAR-10 测试集的顺序不一致")
    return tuple(pixels[(level - 1) * n:level * n] for level in SEVERITIES)


@torch.inference_mode()
def evaluate(model, images, indices, batch_size, device):
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    predictions = np.empty(len(indices), dtype=np.int64)
    confidence = np.empty(len(indices), dtype=np.float32)
    for start in range(0, len(indices), batch_size):
        stop = min(start + batch_size, len(indices))
        # 高级索引生成可写的小批量数组，原始 .npy 保持内存映射。
        batch = torch.from_numpy(images[indices[start:stop]]).permute(0, 3, 1, 2)
        pixels = batch.to(device, dtype=torch.float32).div_(255)
        logits = model((pixels - mean) / std)
        best = logits.argmax(dim=1, keepdim=True)
        predictions[start:stop] = best.squeeze(1).cpu().numpy()
        confidence[start:stop] = logits.softmax(dim=1).gather(1, best).squeeze(1).cpu().numpy()
    return predictions, confidence


def save_accuracy(output, accuracy):
    fig, ax = plt.subplots(figsize=(7.6, 4.4), layout="constrained")
    x = np.arange(len(VIEW_NAMES))
    ax.plot(x, accuracy, marker="o", color="#44776d", linewidth=2)
    ax.set_xticks(x, ("Original", "1", "2", "3", "4", "5"))
    ax.set(xlabel="JPEG corruption severity", ylabel="Top-1 accuracy [%]",
           title="CIFAR-10-C JPEG robustness · fixed ResNet-18",
           ylim=(max(0, min(accuracy) - 8), 105), xlim=(-.4, 5.4))
    ax.grid(axis="y", color="#dddddd")
    for position, value in zip(x, accuracy):
        ax.annotate(f"{value:.2f}%", (position, value), xytext=(0, 7),
                    textcoords="offset points", ha="center")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_examples(output, views, indices, targets, classes, predictions, confidence):
    fig, axes = plt.subplots(len(indices), len(views), squeeze=False, layout="constrained",
                             figsize=(12, 1.85 * len(indices) + .55))
    for row, index in enumerate(indices):
        target = targets[index]
        for column, (ax, images) in enumerate(zip(axes[row], views)):
            ax.imshow(images[index], interpolation="nearest")
            ax.set_axis_off()
            if row == 0:
                ax.set_title(VIEW_NAMES[column], fontsize=11)
            prediction = predictions[row, column]
            label = f"{classes[prediction]} · {confidence[row, column]:.1%}"
            if column == 0:
                label = f"#{index}  True: {classes[target]}\n{label}"
            ax.text(.5, -.045, label, transform=ax.transAxes, ha="center", va="top",
                    fontsize=9, color="#17694f" if prediction == target else "#a73828")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_results(output, targets, classes, predictions, device):
    correct = predictions == targets[:, None]
    correct_counts = correct.sum(axis=0)
    accuracy = 100 * correct_counts / len(targets)
    flips = correct[:, :1] & ~correct[:, 1:]
    candidates = np.flatnonzero(flips.any(axis=1))
    failures = flips[candidates].sum(axis=1)
    candidates = candidates[np.argsort(-failures, kind="stable")]
    with (output / "flipped_samples.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["index", "true_class", "wrong_severities", *VIEW_NAMES])
        for index in candidates:
            levels = ";".join(str(level) for level in SEVERITIES if flips[index, level - 1])
            writer.writerow([index, classes[targets[index]], levels,
                             *(classes[label] for label in predictions[index])])

    summary = {
        "model": MODEL_ID,
        "weights_sha256": WEIGHTS_SHA256,
        "device": str(device),
        "samples_per_level": len(targets),
        "top1_percent": dict(zip(VIEW_NAMES, accuracy.tolist())),
        "original_correct_jpeg_wrong": dict(zip(VIEW_NAMES[1:], flips.sum(axis=0).tolist())),
        "original_correct_any_jpeg_wrong": len(candidates),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    save_accuracy(output / "accuracy.png", accuracy)
    for name, value in zip(VIEW_NAMES, accuracy):
        print(f"{name}: {value:.2f}%")
    print(f"原图正确、至少一级 JPEG 错误：{len(candidates)} 张")
    return candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "datasets", help="原始 CIFAR-10 目录")
    parser.add_argument("--corruption-dir", type=Path, default=ROOT / "datasets" / "CIFAR-10-C")
    parser.add_argument("--weights", type=Path, default=ROOT / "models" / "resnet18_cifar10.bin")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--indices", type=int, nargs="+", help="指定要展示的测试图编号，从 0 开始")
    parser.add_argument("--examples", type=int, default=6, help="自动选图数量，默认 6")
    parser.add_argument("--preview-only", action="store_true", help="只评估和展示 --indices 指定的图")
    parser.add_argument("-o", "--output", type=Path, default=ROOT / "outputs" / "cifar10c_jpeg_eval")
    args = parser.parse_args()
    if args.batch_size < 1 or args.examples < 1:
        parser.error("--batch-size 和 --examples 都必须为正整数")
    if args.preview_only and args.indices is None:
        parser.error("--preview-only 需要指定 --indices")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("当前 PyTorch 环境没有可用的 CUDA")

    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    if device.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    try:
        dataset = CIFAR10(root=str(args.data_dir), train=False, download=False)
        targets = np.asarray(dataset.targets)
        views = (dataset.data, *load_jpeg_levels(args.corruption_dir, targets))
        if args.indices is not None and any(not 0 <= i < len(targets) for i in args.indices):
            parser.error(f"--indices 必须在 0 到 {len(targets) - 1} 之间")
        model = load_model(args.weights, device)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"加载失败：{exc}\n"
                    f"请确认 {args.corruption_dir} 包含 jpeg_compression.npy 和 labels.npy。\n"
                    "数据下载：https://zenodo.org/records/2535967\n")

    indices = np.asarray(args.indices) if args.preview_only else np.arange(len(targets))
    results = []
    for name, images in zip(VIEW_NAMES, views):
        print(f"正在评估 {name}：{len(indices)} 张（{device}）", flush=True)
        results.append(evaluate(model, images, indices, args.batch_size, device))
    predictions = np.column_stack([result[0] for result in results])
    confidence = np.column_stack([result[1] for result in results])

    args.output.mkdir(parents=True, exist_ok=True)
    if args.preview_only:
        sample_rows = np.arange(len(indices))
    else:
        candidates = save_results(args.output, targets, dataset.classes, predictions, device)
        sample_rows = np.asarray(args.indices) if args.indices is not None else candidates[:args.examples]
    if len(sample_rows):
        name = "selected_examples.png" if args.indices is not None else "examples.png"
        save_examples(args.output / name, views, indices[sample_rows], targets,
                      dataset.classes, predictions[sample_rows], confidence[sample_rows])
        print(f"展示图像编号：{indices[sample_rows].tolist()}")
    else:
        print("没有找到原图正确但 JPEG 版本错误的样本")
    print(f"输出目录：{args.output.resolve()}")


if __name__ == "__main__":
    main()

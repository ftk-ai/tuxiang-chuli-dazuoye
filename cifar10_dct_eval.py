"""比较 CIFAR-10 原图、低频图和高频图的 ResNet-18 准确率。

运行：python cifar10_dct_eval.py --r 3
使用完整测试集。首次运行会下载公开的 CIFAR-10 ResNet-18 权重。
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18

from cifar10_dct_preview import ROOT, frequency_mask, save_preview, split_frequency
from jpeg_dct_demo import to_uint8


MODEL_ID = "edadaltocg/resnet18_cifar10"
WEIGHTS_URL = f"https://huggingface.co/{MODEL_ID}/resolve/main/pytorch_model.bin?download=true"
WEIGHTS_SHA256 = "f656242526b567dbd46857276acbcfcb4ee74a7f8a7ae5e56557b7e8fa5d914a"
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)
VIEWS = ("original", "low", "high")


def load_model(weights, device):
    if not weights.is_file():
        weights.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(WEIGHTS_URL, str(weights), hash_prefix=WEIGHTS_SHA256)
    if hashlib.sha256(weights.read_bytes()).hexdigest() != WEIGHTS_SHA256:
        raise ValueError(f"权重校验失败：{weights}。请使用 {MODEL_ID} 发布的原始权重。")

    # CIFAR 版本从 32x32 图像开始，使用 3x3 卷积且不做初始池化。
    model = resnet18(weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    return model.to(device).eval()


@torch.inference_mode()
def predict_all(model, dataset, low_mask, batch_size, device):
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    predictions = np.empty((len(dataset), len(VIEWS)), dtype=np.int64)
    probabilities = np.empty((len(dataset), len(VIEWS), len(dataset.classes)), dtype=np.float32)
    for start in range(0, len(dataset), batch_size):
        stop = min(start + batch_size, len(dataset))
        originals = dataset.data[start:stop]
        low = np.empty_like(originals)
        high = np.empty_like(originals)
        for i, rgb in enumerate(originals):
            low_rgb, high_rgb = split_frequency(rgb, low_mask)
            # 和 PNG 使用同一组像素值，避免增强预览改变模型输入。
            low[i] = to_uint8(low_rgb)
            high[i] = to_uint8(high_rgb)

        for column, images in enumerate((originals, low, high)):
            pixels = torch.from_numpy(images).permute(0, 3, 1, 2).to(device, dtype=torch.float32)
            normalized = (pixels.div_(255) - mean) / std
            logits = model(normalized)
            predictions[start:stop, column] = logits.argmax(dim=1).cpu().numpy()
            probabilities[start:stop, column] = logits.softmax(dim=1).cpu().numpy()
        if (start // batch_size + 1) % 10 == 0 or stop == len(dataset):
            print(f"已评估 {stop}/{len(dataset)} 张测试图（三种输入）", flush=True)
    return predictions, probabilities


def save_results(dataset, predictions, probabilities, radius, low_mask, device, output):
    targets = np.asarray(dataset.targets)
    correct = predictions == targets[:, None]
    matches = np.flatnonzero(correct[:, 0] & ~correct[:, 1] & correct[:, 2]).tolist()
    low_correct_high_wrong = correct[:, 1] & ~correct[:, 2]
    output.mkdir(parents=True, exist_ok=True)

    with (output / "matches.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "index", "true_class", *(f"{view}_prediction" for view in VIEWS),
            *(f"{view}_true_class_probability" for view in VIEWS),
        ])
        for index in matches:
            target = targets[index]
            labels = [target, *predictions[index]]
            writer.writerow([
                index, *(dataset.classes[label] for label in labels),
                *(f"{score:.8g}" for score in probabilities[index, :, target]),
            ])

    page_size = 8
    for start in range(0, len(matches), page_size):
        page = start // page_size + 1
        save_preview(dataset, matches[start:start + page_size], radius,
                     output / f"examples_{page:03d}.png",
                     predictions=predictions, probabilities=probabilities)

    summary = {
        "model": MODEL_ID,
        "weights_sha256": WEIGHTS_SHA256,
        "device": str(device),
        "dataset": "CIFAR-10 test",
        "samples": len(dataset),
        "radius": radius,
        "low_coefficients_per_block": int(low_mask.sum()),
        "high_coefficients_per_block": int((~low_mask).sum()),
        "correct": dict(zip(VIEWS, correct.sum(axis=0).tolist())),
        "top1_percent": dict(zip(VIEWS, (100 * correct.mean(axis=0)).tolist())),
        "match_rule": "original correct AND low incorrect AND high correct",
        "matches": len(matches),
        "match_percent": 100 * len(matches) / len(dataset),
        "low_correct_high_wrong": int(low_correct_high_wrong.sum()),
        "original_and_low_correct_high_wrong": int((correct[:, 0] & low_correct_high_wrong).sum()),
        "example_pages": (len(matches) + page_size - 1) // page_size,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for view in VIEWS:
        print(f"{view:>8}: {summary['top1_percent'][view]:.2f}%"
              f" ({summary['correct'][view]}/{len(dataset)})")
    print(f"符合条件：{len(matches)}/{len(dataset)}，结果保存在 {output.resolve()}")
    print(f"低频正确、高频错误：{summary['low_correct_high_wrong']}/{len(dataset)}")
    print(f"其中原图也正确：{summary['original_and_low_correct_high_wrong']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--r", type=float, default=3.0, help="低频半径，默认 3")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "datasets")
    parser.add_argument("--weights", type=Path, default=ROOT / "models" / "resnet18_cifar10.bin",
                        help="公开权重的缓存路径")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("-o", "--output", type=Path, help="输出目录，必须为空或尚未创建")
    args = parser.parse_args()
    if not np.isfinite(args.r) or args.r < 0:
        parser.error("--r 必须是有限的非负数")
    if args.batch_size < 1:
        parser.error("--batch-size 必须是正整数")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("当前 PyTorch 环境没有可用的 CUDA")

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    radius_label = str(args.r).removesuffix(".0")
    output = args.output or ROOT / "outputs" / "cifar10_dct_eval" / f"r{radius_label}"
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f"输出目录已存在且非空，请用 --output 指定新目录：{output}")

    try:
        dataset = CIFAR10(root=str(args.data_dir), train=False, download=True)
        model = load_model(args.weights, device)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"加载数据或模型失败：{exc}\n权重来源：{WEIGHTS_URL}\n")

    low_mask = frequency_mask(args.r)
    print(f"设备：{device}，测试图：{len(dataset)}，r={args.r:g}", flush=True)
    predictions, probabilities = predict_all(model, dataset, low_mask, args.batch_size, device)
    save_results(dataset, predictions, probabilities, args.r, low_mask, device, output)


if __name__ == "__main__":
    main()

"""预览 CIFAR-10 测试图的 RGB 分块 DCT 低频和高频重建。

用法：python cifar10_dct_preview.py --r 3 --indices 0 1 2 3 4 5
依赖：torchvision、NumPy、SciPy、Pillow。首次运行会下载 CIFAR-10。
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision.datasets import CIFAR10

from jpeg_dct_demo import block_dct, block_idct, to_uint8


ROOT = Path(__file__).resolve().parent


def frequency_mask(radius):
    frequencies = np.arange(8)
    return np.hypot(frequencies[:, None], frequencies[None, :]) <= radius


def split_frequency(rgb, low_mask):
    """返回两张浮点 RGB 重建图。仅在导出预览时裁剪、取整。"""
    low_channels = []
    high_channels = []
    for channel in rgb.transpose(2, 0, 1).astype(np.float64):
        coefficients = block_dct(channel)
        low_channels.append(block_idct(coefficients, channel.shape, low_mask))
        high_channels.append(block_idct(coefficients, channel.shape, ~low_mask))
    return np.stack(low_channels, axis=-1), np.stack(high_channels, axis=-1)


def draw_probabilities(draw, x, y, classes, scores, prediction, target, font):
    """所有条形图使用相同的 0–100% 比例。"""
    for class_id, (name, score) in enumerate(zip(classes, scores)):
        bar_y = y + class_id * 13
        color = "#8294a8"
        if class_id == target:
            color = "#167d70"
        elif class_id == prediction:
            color = "#a53a28"
        draw.text((x, bar_y), name, font=font, fill=color)
        draw.rectangle((x + 65, bar_y + 3, x + 159, bar_y + 8), fill="#dfe5eb")
        width = round(float(score) * 95)
        if width:
            draw.rectangle((x + 65, bar_y + 3, x + 64 + width, bar_y + 8), fill=color)
        draw.text((x + 164, bar_y), f"{score:.1%}", font=font, fill="#182433")


def save_preview(dataset, indices, radius, output, predictions=None, probabilities=None):
    low_mask = frequency_mask(radius)
    low_count = int(low_mask.sum())
    tile = 128
    column_x = (20, 172, 324)
    row_height = tile + (51 if predictions is not None else 29)
    mask_x, gallery_width = 493, 678
    if probabilities is not None:
        column_x = (20, 240, 460)
        row_height += 140
        mask_x, gallery_width = 680, 865
    gallery = Image.new("RGB", (gallery_width, max(290, 81 + row_height * len(indices))), "#f4f6f8")
    draw = ImageDraw.Draw(gallery)
    font = ImageFont.load_default()
    ink = "#182433"
    low_color, high_color = "#167d70", "#b45424"

    draw.text((20, 14), f"CIFAR-10 test set  |  RGB block DCT  |  r = {radius:g}", font=font, fill=ink)
    for x, label in zip(column_x, ("Original", f"Low ({low_count}/64)", f"High ({64 - low_count}/64)")):
        draw.text((x, 43), label, font=font, fill=ink)

    mask_y, cell = 82, 19
    draw.text((mask_x, 43), "8 x 8 frequency mask", font=font, fill=ink)
    for k in range(8):
        draw.text((mask_x + cell * k + 6, mask_y - 15), str(k), font=font, fill=ink)
        draw.text((mask_x - 13, mask_y + cell * k + 3), str(k), font=font, fill=ink)
    for u in range(8):
        for v in range(8):
            x, y = mask_x + v * cell, mask_y + u * cell
            draw.rectangle((x, y, x + cell - 2, y + cell - 2),
                           fill=low_color if low_mask[u, v] else high_color)
    draw.text((mask_x, 244), "Low: inside radius", font=font, fill=low_color)
    draw.text((mask_x, 262), "High: outside radius", font=font, fill=high_color)

    for row, index in enumerate(indices):
        image, label = dataset[index]
        original = np.asarray(image)
        low, high = split_frequency(original, low_mask)
        y = 80 + row * row_height
        for x, pixels in zip(column_x, (original, to_uint8(low), to_uint8(high))):
            thumbnail = Image.fromarray(pixels).resize((tile, tile), Image.Resampling.NEAREST)
            gallery.paste(thumbnail, (x, y))
        if predictions is not None:
            for column, (x, prediction) in enumerate(zip(column_x, predictions[index])):
                color = low_color if prediction == label else "#a53a28"
                draw.text((x, y + tile + 4), f"Pred: {dataset.classes[prediction]}",
                          font=font, fill=color)
                if probabilities is not None:
                    draw_probabilities(draw, x, y + tile + 41, dataset.classes,
                                       probabilities[index, column], prediction, label, font)
            draw.text((20, y + tile + 23), f"#{index}  True: {dataset.classes[label]}",
                      font=font, fill=ink)
        else:
            draw.text((20, y + tile + 4), f"#{index}  {dataset.classes[label]}", font=font, fill=ink)

    output.parent.mkdir(parents=True, exist_ok=True)
    gallery.save(output)
    return low_count


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--r", type=float, default=3.0, help="低频半径，默认 3")
    parser.add_argument("--indices", type=int, nargs="+", default=list(range(6)),
                        help="测试集图像编号，从 0 开始，默认 0 到 5")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "datasets",
                        help="CIFAR-10 数据目录，默认脚本旁的 datasets")
    parser.add_argument("-o", "--output", type=Path, help="输出 PNG 路径")
    args = parser.parse_args()
    if not np.isfinite(args.r) or args.r < 0:
        parser.error("--r 必须是有限的非负数")
    if any(not 0 <= index < 10000 for index in args.indices):
        parser.error("--indices 必须在 0 到 9999 之间")

    output = args.output or ROOT / "outputs" / "cifar10_dct" / f"comparison_r{args.r:g}.png"
    if output.suffix.lower() != ".png":
        parser.error("--output 必须以 .png 结尾")
    try:
        dataset = CIFAR10(root=str(args.data_dir), train=False, download=True)
    except (OSError, RuntimeError) as exc:
        parser.exit(1, f"读取 CIFAR-10 失败：{exc}\n"
                    f"可把官方的 cifar-10-python.tar.gz 放到 {args.data_dir.resolve()} 后重试。\n")
    low_count = save_preview(dataset, args.indices, args.r, output)
    print(f"r={args.r:g}: 低频 {low_count}/64，高频 {64 - low_count}/64")
    print(f"已保存：{output.resolve()}")


if __name__ == "__main__":
    main()

"""8x8 分块 DCT、交互选择基底与逆变换（不进行量化）。

运行：python jpeg_dct_demo.py Y.png Cb.png Cr.png
色度也可以输入上一阶段的 Cb_420.png、Cr_420.png。
无窗口：python jpeg_dct_demo.py Y.png Cb.png Cr.png --no-gui --keep "0,0;0,1;1,0"
依赖：numpy Pillow scipy；窗口使用 Python 自带 tkinter。

所有块、三个分量共用一张基底选择掩码。索引从 0 开始，先行后列。
输入必须是上一阶段保存的单通道 8 位 PNG。边缘复制补齐到 8 的倍数。
先减 128 再 DCT；逆变换后加回 128，并裁掉补齐部分。
DCT 可视化是 log(1+abs(系数)) 灰度图，仅供观察，不可用于后续量化。
下一步应读取 dct_coefficients.npz 中的浮点系数，而不是读取可视化 PNG。
"""

# python jpeg_dct_demo.py outputs/paizhao_ycbcr/Y.png outputs/paizhao_ycbcr/Cb_420.png outputs/paizhao_ycbcr/Cr_420.png
# python jpeg_dct_demo.py outputs/ttg_semi_final_ycbcr/Y.png outputs/ttg_semi_final_ycbcr/Cb_420.png outputs/ttg_semi_final_ycbcr/Cr_420.png
# python jpeg_dct_demo.py outputs/骑行_ycbcr/Y.png outputs/骑行_ycbcr/Cb_420.png outputs/骑行_ycbcr/Cr_420.png
# python jpeg_dct_demo.py outputs/sunlight_ease_ycbcr/Y.png outputs/sunlight_ease_ycbcr/Cb_420.png outputs/sunlight_ease_ycbcr/Cr_420.png

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.fft import dctn, idctn


def to_uint8(array):
    return np.clip(np.floor(array + 0.5), 0, 255).astype(np.uint8)


def load_channel(path):
    with Image.open(path) as image:
        if image.format != "PNG" or image.mode != "L":
            raise ValueError(f"需要单通道 8 位 PNG（模式 L）：{path}")
        return np.array(image, dtype=np.float64)


def block_dct(channel):
    h, w = channel.shape
    padded = np.pad(channel, ((0, (-h) % 8), (0, (-w) % 8)), mode="edge")
    # 维度：[块行, 块列, 块内行频率, 块内列频率]。
    blocks = padded.reshape(padded.shape[0] // 8, 8, padded.shape[1] // 8, 8).transpose(0, 2, 1, 3)
    return dctn(blocks - 128 , axes=(-2, -1), norm="ortho")
# 

def block_idct(coefficients, shape, mask):
    blocks = idctn(coefficients * mask, axes=(-2, -1), norm="ortho") + 128
    image = blocks.transpose(0, 2, 1, 3).reshape(blocks.shape[0] * 8, blocks.shape[1] * 8)
    return image[:shape[0], :shape[1]]
# 

def coefficient_view(coefficients, reference_max):
    values = np.log1p(np.abs(coefficients))
    # values = abs(coefficients)
    pixels = to_uint8(values * (255 / reference_max)) if reference_max else np.zeros(values.shape, dtype=np.uint8)
    return pixels.transpose(0, 2, 1, 3).reshape(pixels.shape[0] * 8, pixels.shape[1] * 8)


def compose_rgb(channels):
    y = channels["Y"]
    h, w = y.shape
    chroma = []
    for name in ("Cb", "Cr"):
        array = channels[name]
        if array.shape != y.shape:
            array = np.repeat(np.repeat(array, 2, axis=0), 2, axis=1)[:h, :w]
        chroma.append(array - 128)
    cb, cr = chroma
    return to_uint8(np.stack((y + 1.402 * cr,
                             y - 0.344136 * cb - 0.714136 * cr,
                             y + 1.772 * cb), axis=-1))


def basis_atlas(mask, tile=44):
    """每个格子是 c_i.T @ c_j；同一灰度尺度，灰=0，黑=负，白=正。"""
    n = np.arange(8)
    c = np.sqrt(2 / 8) * np.cos(np.pi * (2 * n[None, :] + 1) * n[:, None] / 16)
    c[0] /= np.sqrt(2)
    basis = np.einsum("ir,js->ijrs", c, c)
    peak = np.abs(basis).max()
    atlas = Image.new("RGB", (8 * tile, 8 * tile), "#222222")
    draw = ImageDraw.Draw(atlas)
    for i in range(8):
        for j in range(8):
            x, y = j * tile, i * tile
            patch = Image.fromarray(to_uint8(127.5 + 127.5 * basis[i, j] / peak))
            patch = patch.resize((tile - 4, tile - 16), Image.Resampling.NEAREST).convert("RGB")
            atlas.paste(patch, (x + 2, y + 2))
            color = "#34d399" if mask[i, j] else "#666666"
            draw.rectangle((x, y, x + tile - 1, y + tile - 1), outline=color, width=2)
            draw.text((x + 4, y + tile - 13), f"{i},{j}", fill=color)
    return atlas


class Experiment:
    def __init__(self, channels, output, mask):
        yshape = channels["Y"].shape
        cshape = channels["Cb"].shape
        if channels["Cr"].shape != cshape or cshape not in (yshape, tuple((s + 1) // 2 for s in yshape)):
            raise ValueError("Cb、Cr 尺寸必须相同，且等于 Y 尺寸或 Y 宽高各减半（向上取整）")
        self.channels = channels
        self.coeff = {name: block_dct(array) for name, array in channels.items()}
        self.output = output
        self.mask = mask.copy()
        self.reference_max = float(np.log1p(np.abs(self.coeff["Y"])).max())
        self.original_rgb = compose_rgb(channels)

    def reconstruct(self):
        return {name: block_idct(self.coeff[name], array.shape, self.mask)
                for name, array in self.channels.items()}

    def save(self):
        self.output.mkdir(parents=True, exist_ok=True)
        reconstructed = self.reconstruct()
        for name, array in reconstructed.items():
            Image.fromarray(to_uint8(array)).save(self.output / f"{name}_reconstructed.png")
        Image.fromarray(self.original_rgb).save(self.output / "rgb_before_dct.png")
        Image.fromarray(compose_rgb(reconstructed)).save(self.output / "rgb_reconstructed.png")
        for filename, coeff in (("Y_dct_log.png", self.coeff["Y"]),
                                ("Y_dct_selected_log.png", self.coeff["Y"] * self.mask)):
            Image.fromarray(coefficient_view(coeff, self.reference_max)).save(self.output / filename)
        basis_atlas(self.mask).save(self.output / "basis_selection.png")
        data = {name: array for name, array in self.coeff.items()}
        data.update({f"{name}_selected": array * self.mask for name, array in self.coeff.items()})
        data.update({f"{name}_shape": np.array(array.shape) for name, array in self.channels.items()})
        data["mask"] = self.mask
        np.savez_compressed(self.output / "dct_coefficients.npz", **data)
        info = {
            "selected_basis_row_col": np.argwhere(self.mask).tolist(),
            "mask_applies_to": "All blocks of Y, Cb and Cr",
            "coefficient_axes": ["block_row", "block_col", "vertical_frequency", "horizontal_frequency"],
            "transform": "DCT-II, orthonormal, level shift -128, edge padding",
            "coefficient_view": "log(1+abs(coefficient)); both PNGs use the same scale; signs are not shown",
            "quantization": "None. Deselected coefficients are zeroed, which is a lossy experiment.",
            "npz": "Y/Cb/Cr are full coefficients; *_selected use mask; *_shape are unpadded (height,width)",
        }
        (self.output / "selection.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved {int(self.mask.sum())}/64 bases to {self.output.resolve()}")


class DemoWindow:
    def __init__(self, root, experiment):
        import tkinter as tk
        from PIL import ImageTk

        self.tk, self.ImageTk = tk, ImageTk
        self.root, self.experiment = root, experiment
        self.pending = None
        root.title("JPEG：8×8 二维 DCT 基底选择")
        left = tk.Frame(root, padx=12, pady=12)
        left.pack(side="left", fill="y")
        tk.Label(left, text="点击切换基底：绿框=保留；灰框=移除\n向右：水平频率增加；向下：垂直频率增加").pack()
        self.canvas = tk.Canvas(left, width=352, height=352, highlightthickness=0)
        self.canvas.pack(pady=8)
        self.canvas.bind("<Button-1>", self.toggle)
        for label, action in (("全部保留", "all"), ("全部移除", "none"),
                              ("仅 DC (0,0)", "dc"), ("左上角 4×4（16 个基底）", "low")):
            tk.Button(left, text=label, command=lambda a=action: self.preset(a)).pack(fill="x", pady=2)
        tk.Button(left, text="保存当前选择和重建结果", command=self.save).pack(fill="x", pady=12)
        self.status = tk.Label(left, text="", wraplength=350, justify="left")
        self.status.pack()
        right = tk.Frame(root, padx=8, pady=8)
        right.pack(side="left", fill="both", expand=True)
        self.labels = []
        for index, title in enumerate(("输入分量合成的 RGB", "所选基底重建 RGB")):#, "Y 的 DCT 幅度（对数增强）", "所选基底重建 Y"
            frame = tk.LabelFrame(right, text=title)
            frame.grid(row=index // 2, column=index % 2, padx=4, pady=4)
            label = tk.Label(frame)
            label.pack()
            self.labels.append(label)
        tk.Label(right, text="所有块和三个分量共用所选基底；PNG 是预览，下一步量化请读取 NPZ 浮点系数。").grid(row=2, column=0, columnspan=2)
        self.refresh()

    def toggle(self, event):
        i, j = event.y // 44, event.x // 44
        if 0 <= i < 8 and 0 <= j < 8:
            self.experiment.mask[i, j] = not self.experiment.mask[i, j]
            self.schedule()

    def preset(self, name):
        mask = self.experiment.mask
        mask[:] = name == "all"
        if name == "dc":
            mask[0, 0] = True
        elif name == "low":
            mask[:4, :4] = True
        self.schedule()

    def schedule(self):
        if self.pending is not None:
            self.root.after_cancel(self.pending)
        self.pending = self.root.after(120, self.refresh)

    def refresh(self):
        self.pending = None
        exp = self.experiment
        self.atlas_photo = self.ImageTk.PhotoImage(basis_atlas(exp.mask))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.atlas_photo)
        recon = exp.reconstruct()
        arrays = (exp.original_rgb, compose_rgb(recon),
                  )#coefficient_view(exp.coeff["Y"], exp.reference_max), to_uint8(recon["Y"])
        for label, array in zip(self.labels, arrays):
            preview = Image.fromarray(array)
            # 最近邻显示，避免预览插值掩盖块状细节。
            # ----
            preview.thumbnail((500, 500), Image.Resampling.NEAREST)
            # ----
            photo = self.ImageTk.PhotoImage(preview)
            label.configure(image=photo)
            label.image = photo
        count = int(exp.mask.sum())
        mae = float(np.abs(recon["Y"] - exp.channels["Y"]).mean())
        self.status.configure(text=f"已保留 {count}/64 个基底\n Y那张图的误差：{mae:.6f}\n点击保存以导出当前结果。全部移除时重建值为 128。")

    def save(self):
        from tkinter import messagebox
        try:
            self.experiment.save()
        except OSError as exc:
            messagebox.showerror("保存失败", str(exc))
            return
        self.status.configure(text=f"已保存 {int(self.experiment.mask.sum())}/64 个基底\n{self.experiment.output.resolve()}")


def parse_mask(value):
    mask = np.zeros((8, 8), dtype=bool)
    if value == "all":
        mask[:] = True
    elif value != "none":
        for pair in value.split(";"):
            i, j = map(int, pair.split(","))
            if not (0 <= i < 8 and 0 <= j < 8):
                raise ValueError("基底行列编号范围为 0~7")
            mask[i, j] = True
    return mask


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("Y", "Cb", "Cr"):
        parser.add_argument(name, type=Path)
    parser.add_argument("-o", "--output-dir", type=Path)
    parser.add_argument("--keep", default="all", help='all、none 或 "0,0;0,1;1,0"，默认全部保留')
    parser.add_argument("--no-gui", action="store_true", help="只导出结果，不打开交互窗口")
    args = parser.parse_args()
    try:
        channels = {name: load_channel(getattr(args, name)) for name in ("Y", "Cb", "Cr")}
        experiment = Experiment(channels, args.output_dir or args.Y.parent / "dct_demo", parse_mask(args.keep))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    experiment.save()
    if not args.no_gui:
        import tkinter as tk
        root = tk.Tk()
        DemoWindow(root, experiment)
        root.mainloop()


if __name__ == "__main__":
    main()

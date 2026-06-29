from pathlib import Path
import math

from PIL import Image, ImageDraw, ImageFont, ImageFilter


HERE = Path(__file__).resolve().parent
OUT = HERE / "06_advanced_architecture_presentation.png"

W, H = 3840, 2160
BG = "#f4f7fb"
WHITE = "#ffffff"
INK = "#102033"
MUTED = "#5c6b7c"
LINE = "#d6e0ee"
BLUE = "#2563eb"
CYAN = "#0891b2"
PURPLE = "#7c3aed"
GREEN = "#059669"
ORANGE = "#d97706"
RED = "#dc2626"
DARK = "#0f172a"


def load_font(size, bold=False):
    names = ["seguisb.ttf", "segoeuib.ttf", "arialbd.ttf"] if bold else ["segoeui.ttf", "arial.ttf"]
    for name in names:
        p = Path("C:/Windows/Fonts") / name
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


F_TITLE = load_font(64, True)
F_SUB = load_font(31)
F_H = load_font(38, True)
F_M = load_font(29, True)
F_BODY = load_font(24)
F_SMALL = load_font(21)
F_TINY = load_font(18)
F_CODE = load_font(22)


img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)


def shadow(box, radius=28, alpha=30):
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(layer)
    sd.rounded_rectangle((box[0] + 8, box[1] + 12, box[2] + 8, box[3] + 12), radius=radius, fill=(15, 23, 42, alpha))
    layer = layer.filter(ImageFilter.GaussianBlur(14))
    img.paste(layer, (0, 0), layer)


def card(box, fill=WHITE, outline=LINE, width=3, r=26, sh=True):
    if sh:
        shadow(box, r)
    d.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=width)


def text(x, y, value, font=F_BODY, fill=INK, anchor=None):
    d.text((x, y), value, font=font, fill=fill, anchor=anchor)


def fit_text_center(box, value, font, fill=INK):
    text((box[0] + box[2]) / 2, (box[1] + box[3]) / 2, value, font, fill, "mm")


def arrow(x1, y1, x2, y2, color, width=5):
    d.line((x1, y1, x2, y2), fill=color, width=width)
    ang = math.atan2(y2 - y1, x2 - x1)
    size = 18
    p1 = (x2 - size * math.cos(ang - 0.45), y2 - size * math.sin(ang - 0.45))
    p2 = (x2 - size * math.cos(ang + 0.45), y2 - size * math.sin(ang + 0.45))
    d.polygon([(x2, y2), p1, p2], fill=color)


def poly_arrow(points, color, width=5):
    d.line(points, fill=color, width=width, joint="curve")
    x1, y1 = points[-2]
    x2, y2 = points[-1]
    ang = math.atan2(y2 - y1, x2 - x1)
    size = 18
    p1 = (x2 - size * math.cos(ang - 0.45), y2 - size * math.sin(ang - 0.45))
    p2 = (x2 - size * math.cos(ang + 0.45), y2 - size * math.sin(ang + 0.45))
    d.polygon([(x2, y2), p1, p2], fill=color)


def pill(box, label, color, fill=WHITE, font=F_SMALL):
    d.rounded_rectangle(box, radius=(box[3] - box[1]) // 2, fill=fill, outline=color, width=3)
    fit_text_center(box, label, font, color)


def code_box(box, lines, color, fill="#fbfdff"):
    card(box, fill=fill, outline=color, width=3, r=18, sh=False)
    x0, y0, x1, y1 = box
    yy = y0 + 18
    for line in lines:
        text(x0 + 20, yy, line, F_CODE, INK)
        yy += 34


def waveform(box, color=BLUE):
    card(box, fill=WHITE, outline="#cdd9ea", width=2, r=18, sh=False)
    x0, y0, x1, y1 = box
    mid = (y0 + y1) / 2
    pts = []
    for i in range(230):
        t = i / 229
        env = 0.25 + 0.75 * abs(math.sin(3.2 * math.pi * t))
        y = mid + (math.sin(i * 0.23) + 0.38 * math.sin(i * 0.77)) * env * (y1 - y0) * 0.18
        x = x0 + 18 + t * (x1 - x0 - 36)
        pts.append((x, y))
    d.line((x0 + 16, mid, x1 - 16, mid), fill="#cbd5e1", width=2)
    d.line(pts, fill=color, width=5)


def matrix(box, palette, rows=20, cols=34):
    card(box, fill=WHITE, outline="#cdd9ea", width=2, r=18, sh=False)
    x0, y0, x1, y1 = box
    pad = 14
    cw = (x1 - x0 - 2 * pad) / cols
    ch = (y1 - y0 - 2 * pad) / rows
    for r in range(rows):
        for c in range(cols):
            val = (math.sin(c * 0.42 + r * 0.22) + math.cos(c * 0.16 - r * 0.70) + 2) / 4
            if c % 10 in (0, 1):
                val = min(1, val + 0.25)
            idx = max(0, min(len(palette) - 1, int(val * (len(palette) - 1))))
            d.rectangle((x0 + pad + c * cw + 1, y0 + pad + r * ch + 1, x0 + pad + (c + 1) * cw - 1, y0 + pad + (r + 1) * ch - 1), fill=palette[idx])


def bar_vector(box):
    card(box, fill=WHITE, outline="#cdd9ea", width=2, r=18, sh=False)
    x0, y0, x1, y1 = box
    colors = [BLUE, CYAN, PURPLE, GREEN, ORANGE, RED]
    for i in range(34):
        bw = (x1 - x0 - 38) / 34
        h = (0.20 + 0.72 * ((math.sin(i * 1.31) + 1) / 2)) * (y1 - y0 - 36)
        x = x0 + 19 + i * bw
        d.rounded_rectangle((x + 2, y1 - 18 - h, x + bw - 2, y1 - 18), radius=4, fill=colors[i % len(colors)])


def paste_image(src, box):
    p = HERE / "assets" / src
    if not p.exists():
        return False
    card(box, fill=WHITE, outline="#cdd9ea", width=2, r=18, sh=False)
    im = Image.open(p).convert("RGB")
    x0, y0, x1, y1 = box
    im.thumbnail((x1 - x0 - 18, y1 - y0 - 18))
    img.paste(im, (int(x0 + (x1 - x0 - im.width) / 2), int(y0 + (y1 - y0 - im.height) / 2)))
    return True


def conv_icon(box):
    x0, y0, x1, y1 = box
    card(box, fill="#f8fbff", outline=BLUE, width=3, r=18, sh=False)
    for row in range(4):
        pts = []
        yy = y0 + 28 + row * 24
        for i in range(54):
            pts.append((x0 + 18 + i * 3.0, yy + math.sin(i * 0.55 + row) * 7))
        d.line(pts, fill="#60a5fa", width=3)
    for i in range(3):
        d.rounded_rectangle((x0 + 185 + i * 16, y0 + 25 + i * 7, x0 + 235 + i * 16, y0 + 98 + i * 5), radius=9, fill="#eff6ff", outline=BLUE, width=3)
    text((x0 + x1) / 2, y1 - 18, "1D-CNN", F_TINY, BLUE, "mm")


def recurrent_icon(box):
    x0, y0, x1, y1 = box
    card(box, fill="#f8fbff", outline=BLUE, width=3, r=18, sh=False)
    xs = [x0 + 42 + i * 44 for i in range(6)]
    for i, xx in enumerate(xs):
        d.rounded_rectangle((xx - 15, y0 + 48, xx + 15, y0 + 78), radius=8, fill="#eff6ff", outline=BLUE, width=2)
        d.rounded_rectangle((xx - 15, y0 + 94, xx + 15, y0 + 124), radius=8, fill="#ecfdf5", outline=GREEN, width=2)
        if i < len(xs) - 1:
            arrow(xx + 17, y0 + 63, xs[i + 1] - 17, y0 + 63, BLUE, 2)
        if i > 0:
            arrow(xx - 17, y0 + 109, xs[i - 1] + 17, y0 + 109, GREEN, 2)
    text((x0 + x1) / 2, y0 + 22, "BiLSTM", F_SMALL, BLUE, "mm")


def attention_icon(box):
    x0, y0, x1, y1 = box
    card(box, fill="#fbf8ff", outline=PURPLE, width=3, r=18, sh=False)
    vals = [18, 34, 26, 60, 42, 74, 50, 30, 44]
    for i, h in enumerate(vals):
        x = x0 + 38 + i * 24
        d.rounded_rectangle((x, y1 - 34 - h, x + 13, y1 - 34), radius=5, fill=PURPLE)
    d.arc((x0 + 26, y0 + 24, x1 - 26, y1 - 20), 198, 342, fill="#bba7ff", width=4)
    text((x0 + x1) / 2, y0 + 22, "Attention", F_SMALL, PURPLE, "mm")


def transformer_icon(box):
    x0, y0, x1, y1 = box
    card(box, fill="#fbf8ff", outline=PURPLE, width=3, r=18, sh=False)
    waveform((x0 + 16, y0 + 43, x0 + 150, y0 + 112), PURPLE)
    arrow(x0 + 160, y0 + 78, x0 + 194, y0 + 78, PURPLE, 3)
    for i in range(4):
        xx = x0 + 204 + i * 58
        d.rounded_rectangle((xx, y0 + 45, xx + 44, y0 + 109), radius=9, fill="#f5f3ff", outline=PURPLE, width=3)
        text(xx + 22, y0 + 77, "Tr", F_TINY, PURPLE, "mm")
    text((x0 + x1) / 2, y1 - 17, "Frozen WavLM", F_TINY, MUTED, "mm")


def cnn2d_icon(box):
    x0, y0, x1, y1 = box
    card(box, fill="#f8feff", outline=CYAN, width=3, r=18, sh=False)
    labels = ["48", "64", "96", "128"]
    for i, lab in enumerate(labels):
        xx = x0 + 30 + i * 68
        d.rounded_rectangle((xx, y0 + 45, xx + 52, y0 + 98), radius=9, fill="#ecfeff", outline=CYAN, width=3)
        text(xx + 26, y0 + 71, lab, F_TINY, CYAN, "mm")
        if i < len(labels) - 1:
            arrow(xx + 54, y0 + 71, xx + 67, y0 + 71, CYAN, 2)
    d.rounded_rectangle((x1 - 78, y0 + 38, x1 - 22, y0 + 105), radius=11, fill="#fff7ed", outline=ORANGE, width=3)
    text(x1 - 50, y0 + 64, "SE", F_TINY, ORANGE, "mm")
    text(x1 - 50, y0 + 86, "gate", F_TINY, ORANGE, "mm")


def svm_icon(box):
    x0, y0, x1, y1 = box
    card(box, fill="#f8fffb", outline=GREEN, width=3, r=18, sh=False)
    for i in range(17):
        x = x0 + 32 + i * 16
        y = y0 + 42 + 28 * math.sin(i * 0.6)
        d.ellipse((x - 5, y - 5, x + 5, y + 5), fill=GREEN)
    for i in range(17):
        x = x0 + 32 + i * 16
        y = y0 + 108 + 24 * math.cos(i * 0.58)
        d.ellipse((x - 5, y - 5, x + 5, y + 5), fill=ORANGE)
    pts = []
    for i in range(170):
        t = i / 169
        x = x0 + 26 + t * (x1 - x0 - 52)
        y = y0 + 82 + 18 * math.sin(t * math.pi * 3.0)
        pts.append((x, y))
    d.line(pts, fill=DARK, width=3)


def block(box, title, subtitle, color, fill=WHITE):
    card(box, fill=fill, outline=color, width=4, r=24)
    x0, y0, x1, y1 = box
    text(x0 + 24, y0 + 20, title, F_M, color)
    text(x0 + 24, y0 + 58, subtitle, F_SMALL, MUTED)


# background grid
for x in range(0, W, 120):
    d.line((x, 0, x, H), fill="#edf2f8", width=1)
for y in range(0, H, 120):
    d.line((0, y, W, y), fill="#edf2f8", width=1)

# header
card((70, 48, 3770, 172), fill=WHITE, outline="#dce6f2", width=2, r=24)
text(120, 85, "06 Advanced Model Architecture", F_TITLE, INK)
text(122, 142, "visual-first layout: larger feature images, larger modules, separate Fusion and Output blocks", F_SUB, MUTED)

# Main container
card((70, 220, 3770, 2070), fill=WHITE, outline="#b8c7da", width=4, r=34)
text(120, 275, "Unified Speech Emotion Recognition Pipeline", F_H, INK)
text(122, 318, "One audio sample is transformed into four representations, fused into one final prediction.", F_BODY, MUTED)

# Left input pipeline: one connected data path.
card((120, 390, 470, 1475), fill="#f8fafc", outline="#94a3b8", width=4, r=28)
text(158, 435, "Input Pipeline", F_H, INK)
text(160, 478, "one audio sample", F_SMALL, MUTED)

card((155, 535, 435, 710), fill="#eff6ff", outline=BLUE, width=3, r=20, sh=False)
text(180, 565, "1. Input", F_M, BLUE)
text(182, 598, "16 kHz waveform", F_SMALL, MUTED)
waveform((190, 625, 405, 690), BLUE)
arrow(295, 710, 295, 765, BLUE, 4)

card((155, 765, 435, 940), fill="#ffffff", outline=CYAN, width=3, r=20, sh=False)
text(180, 795, "2. Preprocess", F_M, CYAN)
text(182, 828, "crop 3s + normalize", F_SMALL, MUTED)
pill((185, 865, 405, 910), "no aug before split", ORANGE, font=F_TINY)
arrow(295, 940, 295, 995, CYAN, 4)

card((155, 995, 435, 1400), fill="#ffffff", outline=DARK, width=3, r=20, sh=False)
text(180, 1025, "3. Feature Splitter", F_M, DARK)
text(182, 1058, "same audio -> 4 views", F_SMALL, MUTED)
for i, (lab, col) in enumerate([("A Temporal", BLUE), ("B Spectral", CYAN), ("C WavLM", PURPLE), ("D Stats", GREEN)]):
    pill((185, 1110 + i * 62, 405, 1154 + i * 62), lab, col, font=F_TINY)

card((120, 1540, 470, 1910), "Protocol", GREEN, width=0) if False else None
block((120, 1540, 470, 1910), "Protocol Note", "training/evaluation only", GREEN, "#f0fdf4")
pill((160, 1635, 430, 1685), "train-only aug", ORANGE)
pill((160, 1710, 430, 1760), "SpecAugment + mixup", PURPLE)
pill((160, 1785, 430, 1835), "random / strict / single", GREEN)

# Branch cards
bx, bw, bh = 610, 2010, 360
ys = [390, 795, 1200, 1605]
branches = [
    ("A", "Temporal Acoustic Branch", "[B,T,132]", BLUE, "#eff6ff"),
    ("B", "Spectrogram CNN-SE Branch", "[B,3,96,T]", CYAN, "#ecfeff"),
    ("C", "Frozen WavLM Speech Branch", "[B,48000]", PURPLE, "#f5f3ff"),
    ("D", "Statistical + RBF-SVM Branch", "~2937-D vector", GREEN, "#ecfdf5"),
]

for idx, (tag, title, shape, col, fill) in enumerate(branches):
    y = ys[idx]
    card((bx, y, bx + bw, y + bh), fill=fill, outline=col, width=4, r=28)
    d.rounded_rectangle((bx + 24, y + 24, bx + 92, y + 92), radius=20, fill=col)
    text(bx + 58, y + 58, tag, F_M, WHITE, "mm")
    text(bx + 115, y + 28, title, F_M, col)
    text(bx + 115, y + 67, shape, F_SMALL, MUTED)
    source_y = 1132 + idx * 62
    poly_arrow([(435, source_y), (540, source_y), (540, y + bh // 2), (bx, y + bh // 2)], col, 4)

# Branch A
y = ys[0]
matrix((680, y + 118, 945, y + 305), ["#dbeafe", "#93c5fd", "#22c55e", "#14b8a6"], 17, 26)
text(812, y + 325, "MFCC + delta + LLDs", F_TINY, MUTED, "mm")
conv_icon((1005, y + 120, 1325, y + 300))
recurrent_icon((1385, y + 120, 1705, y + 300))
attention_icon((1765, y + 120, 2045, y + 300))
code_box((2105, y + 118, 2545, y + 305), ["Conv1d -> BiLSTM", "AttentionPool", "z_temporal[192]"], BLUE)
for x0, x1, col in [(945, 1005, BLUE), (1325, 1385, BLUE), (1705, 1765, PURPLE), (2045, 2105, PURPLE)]:
    arrow(x0, y + 210, x1, y + 210, col, 3)

# Branch B
y = ys[1]
if not paste_image("logmel_delta_comparison_sample.png", (680, y + 112, 1005, y + 312)):
    matrix((680, y + 112, 1005, y + 312), ["#2e1065", "#7e22ce", "#db2777", "#fbbf24"], 18, 24)
text(842, y + 330, "log-Mel / delta / delta2", F_TINY, MUTED, "mm")
cnn2d_icon((1065, y + 120, 1400, y + 300))
matrix((1460, y + 120, 1730, y + 300), ["#2e1065", "#7e22ce", "#db2777", "#fbbf24"], 12, 18)
code_box((1790, y + 118, 2545, y + 305), ["Conv2d(3,48)", "ResidualSE 48->64->96->128", "GAP -> z_spectral[192]"], CYAN)
for x0, x1 in [(1005, 1065), (1400, 1460), (1730, 1790)]:
    arrow(x0, y + 210, x1, y + 210, CYAN, 3)

# Branch C
y = ys[2]
transformer_icon((680, y + 118, 1100, y + 305))
code_box((1160, y + 118, 1710, y + 305), ["WavLM-base-plus", "frozen backbone", "mean hidden states"], PURPLE)
code_box((1770, y + 118, 2545, y + 305), ["Adapter MLP", "Linear(D,384) -> GELU", "Linear(384,160) -> z_wavlm"], PURPLE)
for x0, x1 in [(1100, 1160), (1710, 1770)]:
    arrow(x0, y + 210, x1, y + 210, PURPLE, 3)

# Branch D
y = ys[3]
bar_vector((680, y + 118, 950, y + 305))
code_box((1010, y + 118, 1510, y + 305), ["stats_full(x)", "mean std min max", "IQR skew kurtosis"], GREEN)
svm_icon((1570, y + 118, 1905, y + 305))
code_box((1965, y + 118, 2545, y + 305), ["StatsMLP -> z_stats[128]", "RBF-SVM -> p_svm[6]", "PCA + scaler"], GREEN)
for x0, x1 in [(950, 1010), (1510, 1570), (1905, 1965)]:
    arrow(x0, y + 210, x1, y + 210, GREEN, 3)

# Right panels
fx, fw = 2750, 900
card((fx, 390, fx + fw, 1515), fill=WHITE, outline="#94a3b8", width=4, r=28)
text(fx + 45, 450, "Fusion + Heads", F_H, INK)
text(fx + 47, 492, "combine 4 embeddings into final logits", F_BODY, MUTED)

embs = [("z_temporal", "192-D", BLUE), ("z_spectral", "192-D", CYAN), ("z_wavlm", "160-D", PURPLE), ("z_stats", "128-D", GREEN)]
for i, (name, dim, col) in enumerate(embs):
    x = fx + 55 + (i % 2) * 390
    yy = 565 + (i // 2) * 95
    d.rounded_rectangle((x, yy, x + 330, yy + 66), radius=16, fill=WHITE, outline=col, width=4)
    text(x + 22, yy + 16, name, F_SMALL, col)
    text(x + 245, yy + 40, dim, F_TINY, MUTED)

code_box((fx + 55, 805, fx + fw - 55, 955), ["FusionMLP: concat(zA,zB,zC,zD)", "LayerNorm -> Linear(fusion_dim,256)", "GELU -> Dropout -> fused representation"], DARK)
code_box((fx + 55, 1010, fx + 420, 1145), ["Emotion head", "Linear(256,6)", "softmax -> p_deep"], GREEN)
code_box((fx + 475, 1010, fx + fw - 55, 1145), ["Adversarial head", "GRL(f) regularizer", "reduce speaker/domain leakage"], RED)
code_box((fx + 55, 1215, fx + fw - 55, 1395), ["Final Stacking block", "p_final = w1*p_deep + w2*p_svm", "last decision layer before Output"], ORANGE)

arrow(fx + 265, 955, fx + 265, 1010, GREEN, 4)
arrow(fx + 640, 955, fx + 640, 1010, RED, 4)
poly_arrow([(fx + 235, 1145), (fx + 235, 1180), (fx + 450, 1180), (fx + 450, 1215)], GREEN, 4)

# Embedding arrows
for ymid, col, yy in [(570, BLUE, 598), (975, CYAN, 598), (1380, PURPLE, 693), (1785, GREEN, 693)]:
    poly_arrow([(bx + bw, ymid), (2685, ymid), (2685, yy), (fx + 55, yy)], col, 4)

# Output separate block
card((fx, 1600, fx + fw, 1945), fill="#f0fdf4", outline=GREEN, width=4, r=28)
text(fx + 45, 1660, "Output", F_H, GREEN)
text(fx + 47, 1702, "system result returned by the final block", F_BODY, INK)
pill((fx + 60, 1770, fx + 300, 1820), "6 probabilities", GREEN)
pill((fx + 330, 1770, fx + 570, 1820), "emotion label", GREEN)
pill((fx + 600, 1770, fx + 840, 1820), "confidence", GREEN)
for i, lab in enumerate(["neutral", "happy", "sad", "angry", "fear", "disgust"]):
    x = fx + 60 + (i % 3) * 270
    yy = 1850 + (i // 3) * 42
    d.rounded_rectangle((x, yy, x + 220, yy + 32), radius=16, fill=WHITE, outline="#bbf7d0", width=2)
    text(x + 110, yy + 16, lab, F_TINY, GREEN, "mm")
arrow(fx + 450, 1395, fx + 450, 1600, GREEN, 5)

text(120, 2110, "References: DANN/GRL, WavLM, SENet/SE, SpecAugment, mixup, Transformer attention, SER ensemble/statistical feature engineering.", F_SMALL, MUTED)

img.save(OUT, quality=96)
print(OUT)

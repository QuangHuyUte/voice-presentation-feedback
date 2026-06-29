from pathlib import Path
import math

from PIL import Image, ImageDraw, ImageFont, ImageFilter


HERE = Path(__file__).resolve().parent
SRC = HERE / "06_advanced_architecture_visual_clean.png"
OUT = HERE / "06_advanced_architecture_visual_clean.png"

WHITE = "#ffffff"
INK = "#132238"
MUTED = "#64748b"
LINE = "#d7e3f2"
BLUE = "#2563eb"
CYAN = "#0891b2"
PURPLE = "#7c3aed"
GREEN = "#059669"
ORANGE = "#d97706"
RED = "#dc2626"
DARK = "#0f172a"


def font(size, bold=False):
    names = ["seguisb.ttf", "segoeuib.ttf", "arialbd.ttf"] if bold else ["segoeui.ttf", "arial.ttf"]
    for name in names:
        p = Path("C:/Windows/Fonts") / name
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


F_H = font(33, True)
F_BODY = font(20)
F_SMALL = font(17)
F_TINY = font(14)
F_CODE = font(19)

img = Image.open(SRC).convert("RGB")
d = ImageDraw.Draw(img)
W, H = img.size


def shadow(box, radius=24, alpha=26):
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(layer)
    sd.rounded_rectangle((box[0] + 8, box[1] + 10, box[2] + 8, box[3] + 10), radius=radius, fill=(15, 23, 42, alpha))
    layer = layer.filter(ImageFilter.GaussianBlur(13))
    img.paste(layer, (0, 0), layer)


def card(box, fill=WHITE, outline=LINE, width=3, r=24, sh=True):
    if sh:
        shadow(box, r)
    d.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=width)


def text(x, y, s, f=F_BODY, fill=INK, anchor=None):
    d.text((x, y), s, font=f, fill=fill, anchor=anchor)


def arrow(x1, y1, x2, y2, color, width=5):
    d.line((x1, y1, x2, y2), fill=color, width=width)
    ang = math.atan2(y2 - y1, x2 - x1)
    size = 16
    p1 = (x2 - size * math.cos(ang - 0.45), y2 - size * math.sin(ang - 0.45))
    p2 = (x2 - size * math.cos(ang + 0.45), y2 - size * math.sin(ang + 0.45))
    d.polygon([(x2, y2), p1, p2], fill=color)


def poly_arrow(points, color, width=4):
    d.line(points, fill=color, width=width, joint="curve")
    x1, y1 = points[-2]
    x2, y2 = points[-1]
    ang = math.atan2(y2 - y1, x2 - x1)
    size = 15
    p1 = (x2 - size * math.cos(ang - 0.45), y2 - size * math.sin(ang - 0.45))
    p2 = (x2 - size * math.cos(ang + 0.45), y2 - size * math.sin(ang + 0.45))
    d.polygon([(x2, y2), p1, p2], fill=color)


def code_box(box, lines, color, fill="#fbfdff"):
    card(box, fill=fill, outline=color, width=3, r=16, sh=False)
    x0, y0, x1, y1 = box
    yy = y0 + 16
    for line in lines:
        text(x0 + 16, yy, line, F_CODE, INK)
        yy += 29


# Clear the old right-side fusion/output area while preserving the main model container.
d.rounded_rectangle((2690, 700, 3665, 1930), radius=28, fill=WHITE, outline=WHITE, width=1)

# Redraw incoming branch arrows to the new fusion panel.
for ymid, col, yy in [(865, BLUE, 925), (1165, CYAN, 925), (1465, PURPLE, 1011), (1765, GREEN, 1011)]:
    poly_arrow([(2600, ymid), (2665, ymid), (2665, yy), (2790, yy)], col, 4)

# Fusion panel, now without the output inside it.
card((2730, 735, 3630, 1585), fill=WHITE, outline="#94a3b8", width=4, r=26)
text(2780, 790, "Fusion + Heads", F_H, INK)
text(2782, 830, "one prediction from four embeddings", F_BODY, MUTED)

embs = [("z_temporal", "192-D", BLUE), ("z_spectral", "192-D", CYAN), ("z_wavlm", "160-D", PURPLE), ("z_stats", "128-D", GREEN)]
for i, (name, dim, col) in enumerate(embs):
    x = 2790 + (i % 2) * 335
    yy = 895 + (i // 2) * 86
    d.rounded_rectangle((x, yy, x + 295, yy + 60), radius=15, fill=WHITE, outline=col, width=4)
    text(x + 18, yy + 13, name, F_SMALL, col)
    text(x + 220, yy + 35, dim, F_TINY, MUTED)

code_box((2790, 1105, 3570, 1245), ["FusionMLP: concat(zA,zB,zC,zD)", "LayerNorm -> Linear(fusion_dim,256)", "GELU -> Dropout -> emotion logits"], DARK)
code_box((2790, 1280, 3180, 1415), ["Emotion head", "Linear(256,6)", "softmax -> p_deep"], GREEN)
code_box((3220, 1280, 3570, 1415), ["Stacking", "p_final = w1*p_deep", "        + w2*p_svm"], ORANGE)
code_box((2790, 1450, 3570, 1555), ["Domain adversarial head: GRL(f)", "forward identity | backward -lambda*dL_domain"], RED)

# Separate output block outside Fusion + Heads.
card((2730, 1660, 3630, 1875), fill="#f0fdf4", outline=GREEN, width=4, r=26)
text(2780, 1708, "Output", F_H, GREEN)
text(2782, 1748, "6 probabilities + predicted emotion + confidence", F_BODY, INK)
labels = ["neutral", "happy", "sad", "angry", "fear", "disgust"]
for i, lab in enumerate(labels):
    x = 2780 + (i % 3) * 250
    yy = 1790 + (i // 3) * 42
    d.rounded_rectangle((x, yy, x + 190, yy + 34), radius=17, fill=WHITE, outline="#bbf7d0", width=2)
    text(x + 95, yy + 17, lab, F_TINY, GREEN, "mm")

# Arrow from final stacking block to separate output block.
arrow(3395, 1415, 3395, 1660, GREEN, 5)

img.save(OUT, quality=96)
print(OUT)

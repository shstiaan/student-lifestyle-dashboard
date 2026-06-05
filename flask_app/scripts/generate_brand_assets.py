from __future__ import annotations

import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


BASE_DIR = Path(__file__).resolve().parents[1]
ASSET_DIR = BASE_DIR / "static" / "assets"
ASSET_DIR.mkdir(parents=True, exist_ok=True)

BLACK = "#111111"
CREAM = "#FFF4D6"
WHITE = "#FFFFFF"
BLUE = "#2F6BFF"
YELLOW = "#FFD43B"
CORAL = "#FF5A5F"
MINT = "#54D17A"
PURPLE = "#A78BFA"
PINK = "#FF8BD1"
INK = "#111111"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/ariblk.ttf" if bold else "C:/Windows/Fonts/bahnschrift.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, image_font: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=image_font)
    return box[2] - box[0], box[3] - box[1]


def polygon_shadow(draw: ImageDraw.ImageDraw, points, fill, outline=INK, width=8, shadow_offset=(12, 12)):
    shadow_points = [(x + shadow_offset[0], y + shadow_offset[1]) for x, y in points]
    draw.polygon(shadow_points, fill=INK)
    draw.polygon(points, fill=fill, outline=outline)
    draw.line(points + [points[0]], fill=outline, width=width, joint="curve")


def rectangle_shadow(draw: ImageDraw.ImageDraw, xy, fill, outline=INK, width=8, radius=0, shadow_offset=(12, 12)):
    x1, y1, x2, y2 = xy
    shadow = (x1 + shadow_offset[0], y1 + shadow_offset[1], x2 + shadow_offset[0], y2 + shadow_offset[1])
    if radius:
        draw.rounded_rectangle(shadow, radius=radius, fill=INK)
        draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)
    else:
        draw.rectangle(shadow, fill=INK)
        draw.rectangle(xy, fill=fill, outline=outline, width=width)


def draw_star(draw: ImageDraw.ImageDraw, center, radius, fill, outline=INK, width=5):
    cx, cy = center
    points = []
    for i in range(10):
        angle = -math.pi / 2 + i * math.pi / 5
        r = radius if i % 2 == 0 else radius * 0.42
        points.append((cx + math.cos(angle) * r, cy + math.sin(angle) * r))
    draw.polygon(points, fill=fill, outline=outline)
    draw.line(points + [points[0]], fill=outline, width=width)


def draw_lightning(draw: ImageDraw.ImageDraw, x: int, y: int, scale: float, fill=YELLOW):
    pts = [
        (x + 28 * scale, y),
        (x, y + 72 * scale),
        (x + 42 * scale, y + 64 * scale),
        (x + 18 * scale, y + 132 * scale),
        (x + 96 * scale, y + 42 * scale),
        (x + 52 * scale, y + 50 * scale),
    ]
    pts = [(int(px), int(py)) for px, py in pts]
    draw.polygon([(px + 8, py + 8) for px, py in pts], fill=INK)
    draw.polygon(pts, fill=fill, outline=INK)
    draw.line(pts + [pts[0]], fill=INK, width=max(4, int(7 * scale)))


def draw_face(draw: ImageDraw.ImageDraw, center, radius, mood: str, fill=YELLOW):
    cx, cy = center
    draw.ellipse((cx - radius + 8, cy - radius + 8, cx + radius + 8, cy + radius + 8), fill=INK)
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=fill, outline=INK, width=8)
    eye_y = cy - radius * 0.18
    draw.ellipse((cx - radius * 0.42, eye_y - 8, cx - radius * 0.24, eye_y + 10), fill=INK)
    draw.ellipse((cx + radius * 0.24, eye_y - 8, cx + radius * 0.42, eye_y + 10), fill=INK)
    if mood == "low":
        draw.arc((cx - radius * 0.42, cy - radius * 0.08, cx + radius * 0.42, cy + radius * 0.58), 10, 170, fill=INK, width=7)
    elif mood == "medium":
        draw.line((cx - radius * 0.42, cy + radius * 0.34, cx + radius * 0.42, cy + radius * 0.34), fill=INK, width=7)
    else:
        draw.arc((cx - radius * 0.45, cy + radius * 0.18, cx + radius * 0.45, cy + radius * 0.82), 190, 350, fill=INK, width=7)
        draw.line((cx - radius * 0.55, cy - radius * 0.38, cx - radius * 0.2, cy - radius * 0.26), fill=INK, width=7)
        draw.line((cx + radius * 0.2, cy - radius * 0.26, cx + radius * 0.55, cy - radius * 0.38), fill=INK, width=7)


def generate_logo() -> None:
    image = Image.new("RGBA", (1200, 330), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    # Compact transparent wordmark; the page topbar already provides the outer card.
    rectangle_shadow(draw, (58, 54, 252, 248), "#D9E1FF", width=8, radius=28, shadow_offset=(12, 12))
    draw.ellipse((93, 88, 217, 212), fill="#FFF4D6", outline=INK, width=8)
    draw.arc((118, 120, 192, 196), 190, 350, fill=BLUE, width=12)
    draw.polygon([(151, 153), (199, 116), (174, 168)], fill=CORAL, outline=INK)
    draw.line([(116, 168), (136, 144), (157, 166), (181, 132)], fill=INK, width=7)
    draw_lightning(draw, 214, 44, 0.52, fill=YELLOW)

    title_font = font(104, bold=True)
    subtitle_font = font(28, bold=True)
    draw.text((306 + 6, 78 + 6), "MindMeter", font=title_font, fill=CORAL)
    draw.text((306, 78), "MindMeter", font=title_font, fill=INK)

    pill_xy = (314, 205, 740, 254)
    draw.rounded_rectangle((pill_xy[0] + 6, pill_xy[1] + 6, pill_xy[2] + 6, pill_xy[3] + 6), radius=8, fill=INK)
    draw.rounded_rectangle(pill_xy, radius=8, fill=YELLOW, outline=INK, width=5)
    draw.text((336, 216), "STUDENT STRESS AI", font=subtitle_font, fill=INK)

    draw.line((806, 104, 1070, 104), fill=INK, width=8)
    draw.line((806, 146, 1006, 146), fill=INK, width=8)
    draw.line((806, 188, 1118, 188), fill=INK, width=8)
    draw.ellipse((1084, 66, 1148, 130), fill=MINT, outline=INK, width=6)
    draw.rectangle((1036, 150, 1102, 216), fill=PURPLE, outline=INK, width=6)
    image.save(ASSET_DIR / "mindmeter-logo.png")


def generate_pattern() -> None:
    random.seed(42)
    image = Image.new("RGB", (1400, 900), CREAM)
    draw = ImageDraw.Draw(image)
    for x in range(-40, 1450, 80):
        draw.line((x, 0, x - 420, 900), fill="#E8DDBD", width=4)
    colors = [BLUE, YELLOW, CORAL, MINT, PURPLE, PINK, WHITE]
    for _ in range(72):
        x = random.randint(0, 1320)
        y = random.randint(0, 820)
        size = random.randint(28, 86)
        color = random.choice(colors)
        shape = random.choice(["rect", "circle", "star", "bolt", "zig"])
        if shape == "rect":
            rectangle_shadow(draw, (x, y, x + size, y + size), color, width=5, radius=8, shadow_offset=(6, 6))
        elif shape == "circle":
            draw.ellipse((x + 6, y + 6, x + size + 6, y + size + 6), fill=INK)
            draw.ellipse((x, y, x + size, y + size), fill=color, outline=INK, width=5)
        elif shape == "star":
            draw_star(draw, (x + size // 2, y + size // 2), size // 2, color, width=4)
        elif shape == "bolt":
            draw_lightning(draw, x, y, max(0.35, size / 120), fill=color)
        else:
            pts = [(x, y + size), (x + size // 3, y), (x + size * 2 // 3, y + size), (x + size, y)]
            draw.line([(px + 5, py + 5) for px, py in pts], fill=INK, width=8)
            draw.line(pts, fill=color, width=8)
    image.save(ASSET_DIR / "neo-pattern.png")


def generate_hero() -> None:
    image = Image.new("RGB", (1500, 980), CREAM)
    draw = ImageDraw.Draw(image)
    rectangle_shadow(draw, (80, 80, 1420, 900), WHITE, width=10, radius=26, shadow_offset=(22, 22))
    draw.rectangle((80, 80, 1420, 178), fill=YELLOW, outline=INK, width=10)
    draw.text((120, 112), "MINDMETER ML BOARD", font=font(42, bold=True), fill=INK)
    draw_lightning(draw, 1320, 98, 0.42, fill=CORAL)

    # Student portrait.
    draw.ellipse((196, 246, 596, 646), fill=BLUE, outline=INK, width=10)
    draw.pieslice((256, 170, 540, 470), 195, 345, fill=INK, outline=INK)
    draw.ellipse((294, 278, 508, 520), fill="#F5B885", outline=INK, width=9)
    draw.ellipse((330, 380, 356, 406), fill=INK)
    draw.ellipse((446, 380, 472, 406), fill=INK)
    draw.arc((356, 410, 450, 486), 15, 165, fill=INK, width=7)
    draw.rectangle((252, 562, 548, 760), fill=CORAL, outline=INK, width=10)
    draw.rectangle((322, 532, 478, 600), fill="#F5B885", outline=INK, width=8)

    # Dashboard cards.
    rectangle_shadow(draw, (690, 238, 1268, 438), MINT, width=9, radius=16, shadow_offset=(14, 14))
    draw.text((724, 268), "Stress Classifier", font=font(48, bold=True), fill=INK)
    for idx, color in enumerate([BLUE, YELLOW, CORAL]):
        y = 340 + idx * 45
        draw.rectangle((730, y, 1210, y + 24), fill=WHITE, outline=INK, width=5)
        draw.rectangle((730, y, 730 + 260 + idx * 75, y + 24), fill=color, outline=INK, width=0)

    rectangle_shadow(draw, (700, 520, 960, 778), YELLOW, width=9, radius=16, shadow_offset=(14, 14))
    draw.text((730, 552), "F1", font=font(58, bold=True), fill=INK)
    draw.text((730, 626), "0.8768", font=font(64, bold=True), fill=INK)
    draw.text((730, 712), "Macro score", font=font(28, bold=True), fill=INK)

    rectangle_shadow(draw, (1036, 520, 1290, 778), PURPLE, width=9, radius=16, shadow_offset=(14, 14))
    draw.text((1070, 552), "20", font=font(70, bold=True), fill=INK)
    draw.text((1070, 644), "stress", font=font(31, bold=True), fill=INK)
    draw.text((1070, 688), "signals", font=font(31, bold=True), fill=INK)

    draw_star(draw, (640, 200), 56, PINK)
    draw_lightning(draw, 126, 710, 0.8, fill=YELLOW)
    image.save(ASSET_DIR / "mindmeter-hero.png")


def generate_result(filename: str, title: str, mood: str, bg: str, accent: str) -> None:
    image = Image.new("RGB", (900, 680), bg)
    draw = ImageDraw.Draw(image)
    rectangle_shadow(draw, (66, 66, 834, 614), WHITE, width=10, radius=24, shadow_offset=(18, 18))
    draw.rectangle((66, 66, 834, 170), fill=accent, outline=INK, width=10)
    draw.text((104, 98), title, font=font(50, bold=True), fill=INK)
    draw_face(draw, (450, 370), 150, mood=mood, fill=YELLOW if mood != "high" else CORAL)
    for x, y, c in [(150, 240, BLUE), (720, 268, MINT), (690, 506, PURPLE), (170, 530, CORAL)]:
        draw_star(draw, (x, y), 42, c, width=5)
    draw.rectangle((300, 558, 600, 588), fill=INK)
    image.save(ASSET_DIR / filename)


def main() -> None:
    generate_logo()
    generate_pattern()
    generate_hero()
    generate_result("result-low.png", "LOW STRESS", "low", "#DFF8E8", MINT)
    generate_result("result-medium.png", "MEDIUM STRESS", "medium", "#FFF4D6", YELLOW)
    generate_result("result-high.png", "HIGH STRESS", "high", "#FFE0E0", CORAL)
    print(f"Generated assets in {ASSET_DIR}")


if __name__ == "__main__":
    main()

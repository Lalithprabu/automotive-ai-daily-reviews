"""Builds a short, looping GIF that visually SIMULATES the diffusion
denoising idea behind Marigold: a photo-like scene, then noise resolving
step-by-step into a depth map, ending on a side-by-side reveal.

IMPORTANT: this is an illustrative simulation for social-media
communication, not a real model inference run -- the scene and depth map
are hand-drawn synthetic shapes, and the "denoising" is a blur+blend
animation, not an actual diffusion sampler. Labelled as such in the caption.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# Draw scenes at a larger internal resolution (so shapes have room), then
# downscale to the smaller DISPLAY size before adding caption bars.
W, H = 640, 640            # internal drawing canvas
DW, DH = 420, 420          # final display / GIF size
FONT_BOLD = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
FONT_SMALL = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)

BLUE, ORANGE, AQUA, YELLOW = (42, 120, 214), (235, 104, 52), (27, 175, 122), (237, 161, 0)
INK = (11, 11, 11)


def draw_scene(depth_mode: bool) -> Image.Image:
    """Draws a simple road scene. depth_mode=False -> colour photo.
    depth_mode=True -> grayscale 'near=bright, far=dark' depth map of the
    same geometry."""
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)

    if not depth_mode:
        # Sky gradient
        for y in range(0, 380):
            t = y / 380
            c = (int(120 + t * 80), int(170 + t * 60), int(235 - t * 20))
            d.line([(0, y), (W, y)], fill=c)
        d.ellipse([460, 40, 540, 120], fill=(255, 221, 89))       # sun
        d.rectangle([0, 380, W, H], fill=(96, 168, 96))            # ground
        d.polygon([(220, 380), (420, 380), (360, H), (280, H)], fill=(110, 110, 118))  # road
        d.line([(320, 380), (320, H)], fill=(235, 235, 235), width=4)  # centre line (approx)
        # tree
        d.rectangle([90, 300, 110, 380], fill=(120, 82, 45))
        d.ellipse([55, 230, 145, 320], fill=(60, 140, 70))
        # car
        d.rectangle([280, 440, 380, 490], fill=(200, 60, 60))
        d.rectangle([300, 415, 360, 445], fill=(200, 60, 60))
        d.ellipse([290, 480, 315, 505], fill=(30, 30, 30))
        d.ellipse([345, 480, 370, 505], fill=(30, 30, 30))
    else:
        # far (sky/background) = dark, near (road/car, bottom) = bright
        for y in range(0, 380):
            t = y / 380
            g = int(40 + t * 40)
            d.line([(0, y), (W, y)], fill=(g, g, g))
        d.ellipse([460, 40, 540, 120], fill=(70, 70, 70))
        for y in range(380, H):
            t = (y - 380) / (H - 380)
            g = int(120 + t * 110)
            d.line([(0, y), (W, y)], fill=(g, g, g))
        d.polygon([(220, 380), (420, 380), (360, H), (280, H)], fill=(200, 200, 200))
        d.rectangle([90, 300, 110, 380], fill=(90, 90, 90))
        d.ellipse([55, 230, 145, 320], fill=(85, 85, 85))
        d.rectangle([280, 440, 380, 490], fill=(245, 245, 245))
        d.rectangle([300, 415, 360, 445], fill=(245, 245, 245))
        d.ellipse([290, 480, 315, 505], fill=(230, 230, 230))
        d.ellipse([345, 480, 370, 505], fill=(230, 230, 230))
    return img


def make_noise() -> Image.Image:
    rng = np.random.default_rng(7)
    arr = (rng.random((H, W)) * 255).astype(np.uint8)
    return Image.fromarray(np.stack([arr] * 3, axis=-1))


def to_display(img: Image.Image) -> Image.Image:
    """Downscale from the internal drawing canvas to the final GIF size."""
    return img.resize((DW, DH), Image.LANCZOS)


def caption_bar(img: Image.Image, text: str, color) -> Image.Image:
    img = img.copy()
    d = ImageDraw.Draw(img)
    bar_h = 42
    d.rectangle([0, DH - bar_h, DW, DH], fill=color)
    bbox = d.textbbox((0, 0), text, font=FONT_BOLD)
    tw = bbox[2] - bbox[0]
    d.text(((DW - tw) / 2, DH - bar_h + 10), text, font=FONT_BOLD, fill="white")
    return img


def header_bar(img: Image.Image, text: str) -> Image.Image:
    img = img.copy()
    d = ImageDraw.Draw(img)
    bar_h = 30
    d.rectangle([0, 0, DW, bar_h], fill=INK)
    bbox = d.textbbox((0, 0), text, font=FONT_SMALL)
    tw = bbox[2] - bbox[0]
    d.text(((DW - tw) / 2, 6), text, font=FONT_SMALL, fill="white")
    return img


photo = draw_scene(depth_mode=False)
depth = draw_scene(depth_mode=True)
noise = make_noise()

photo_arr = np.asarray(photo).astype(np.float32)
depth_arr = np.asarray(depth).astype(np.float32)
noise_arr = np.asarray(noise).astype(np.float32)

frames = []
durations = []

# 1) Show the plain photo
frames.append(caption_bar(header_bar(to_display(photo), "STEP 0 · ONE ORDINARY PHOTO"), "A single camera frame", BLUE))
durations.append(1400)

# 2) Pure noise, "AI starts here"
noise_img = Image.fromarray(noise_arr.astype(np.uint8))
frames.append(caption_bar(header_bar(to_display(noise_img), "STEP 1 · START FROM RANDOM NOISE"), "Just like AI image generators", ORANGE))
durations.append(900)

# 3) Progressive denoise: blend noise -> blurred depth -> sharp depth over 4 steps
blend_steps = [
    (0.30, 14, "STEP 2 · REFINING (1 of 4)"),
    (0.58, 8, "STEP 2 · REFINING (2 of 4)"),
    (0.82, 3, "STEP 2 · REFINING (3 of 4)"),
    (1.00, 0, "STEP 2 · REFINING (4 of 4)"),
]
for ratio, blur_sigma, label in blend_steps:
    target_img = Image.fromarray(depth_arr.astype(np.uint8))
    if blur_sigma > 0:
        target_img = target_img.filter(ImageFilter.GaussianBlur(blur_sigma))
    target_arr = np.asarray(target_img).astype(np.float32)
    blended = (1 - ratio) * noise_arr + ratio * target_arr
    blended_img = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))
    frames.append(caption_bar(header_bar(to_display(blended_img), label), "Noise clears into a distance map", AQUA))
    durations.append(650)

# 4) Final side-by-side reveal, held longer
side = Image.new("RGB", (DW, DH), "white")
photo_small = photo.resize((DW // 2, DH), Image.LANCZOS)
depth_small = depth.resize((DW // 2, DH), Image.LANCZOS)
side.paste(photo_small, (0, 0))
side.paste(depth_small, (DW // 2, 0))
d = ImageDraw.Draw(side)
d.line([(DW // 2, 0), (DW // 2, DH)], fill="white", width=3)
side = header_bar(side, "PHOTO  →  DISTANCE MAP")
side = caption_bar(side, "Brighter = closer · Darker = farther", YELLOW)
frames.append(side)
durations.append(2600)

quantized = [f.convert("P", palette=Image.ADAPTIVE, colors=96) for f in frames]

quantized[0].save(
    "marigold_simulation.gif",
    save_all=True,
    append_images=quantized[1:],
    duration=durations,
    loop=0,
    optimize=True,
    disposal=2,
)
print("saved marigold_simulation.gif")

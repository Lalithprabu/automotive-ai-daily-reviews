"""Render tools/banner.html -> assets/banner.png (needs: pip install playwright && playwright install chromium)."""
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1280, "height": 420}, device_scale_factor=2)
    pg.goto((ROOT / "tools" / "banner.html").as_uri())
    pg.wait_for_timeout(300)
    pg.screenshot(path=str(ROOT / "assets" / "banner.png"))
    b.close()
print("assets/banner.png written")

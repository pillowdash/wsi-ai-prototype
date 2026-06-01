import argparse
import base64
import time
from pathlib import Path

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError


TILE_MARKERS = (
    "/tiles/",
    "/heatmap-tiles/",
)


class TileIdleWatcher:
    """
    Tracks OpenSeadragon tile requests and waits until tile loading becomes quiet.

    This is more reliable than only using Playwright's networkidle state because
    OpenSeadragon can continue requesting tiles after the initial page load.
    """

    def __init__(self, page):
        self.page = page
        self.active = 0
        self.last_activity = time.time()

        page.on("request", self._on_request)
        page.on("requestfinished", self._on_request_done)
        page.on("requestfailed", self._on_request_done)

    def _is_tile_request(self, request):
        return any(marker in request.url for marker in TILE_MARKERS)

    def _on_request(self, request):
        if self._is_tile_request(request):
            self.active += 1
            self.last_activity = time.time()

    def _on_request_done(self, request):
        if self._is_tile_request(request):
            self.active = max(0, self.active - 1)
            self.last_activity = time.time()

    def wait_for_idle(self, quiet_ms=1500, timeout_ms=45000):
        deadline = time.time() + timeout_ms / 1000

        while time.time() < deadline:
            quiet_for_ms = (time.time() - self.last_activity) * 1000

            if self.active == 0 and quiet_for_ms >= quiet_ms:
                return

            self.page.wait_for_timeout(150)

        print(
            f"Warning: tile requests did not fully settle. "
            f"active={self.active}"
        )


def set_range_value(page, selector: str, value: str):
    page.eval_on_selector(
        selector,
        """(element, value) => {
            element.value = value;
            element.dispatchEvent(new Event("input", { bubbles: true }));
            element.dispatchEvent(new Event("change", { bubbles: true }));
        }""",
        value,
    )


def wait_for_viewer_ready(page, tile_watcher: TileIdleWatcher):
    page.wait_for_selector("#viewer", timeout=30000)
    page.wait_for_selector("#viewer canvas", timeout=30000)

    page.wait_for_function(
        """() => {
            const status = document.querySelector("#status");
            return status && status.textContent.includes("DZ levels");
        }""",
        timeout=30000,
    )

    tile_watcher.wait_for_idle()


def create_pdf_report(browser, screenshot_path: Path, pdf_path: Path, title: str):
    image_bytes = screenshot_path.read_bytes()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="UTF-8" />
      <title>{title}</title>
      <style>
        body {{
          font-family: Arial, sans-serif;
          margin: 32px;
          color: #222;
        }}
        h1 {{
          margin-bottom: 4px;
        }}
        .subtitle {{
          color: #666;
          margin-bottom: 24px;
        }}
        img {{
          width: 100%;
          border: 1px solid #ccc;
        }}
        .note {{
          margin-top: 18px;
          font-size: 12px;
          color: #666;
        }}
      </style>
    </head>
    <body>
      <h1>{title}</h1>
      <div class="subtitle">Generated from the ZFP WSI Viewer prototype</div>

      <img src="data:image/png;base64,{image_b64}" />

      <div class="note">
        Research prototype only. Not clinically validated. Not for diagnosis.
      </div>
    </body>
    </html>
    """

    report_page = browser.new_page()
    report_page.set_content(html, wait_until="load")
    report_page.pdf(
        path=str(pdf_path),
        format="A4",
        print_background=True,
        margin={
            "top": "1cm",
            "right": "1cm",
            "bottom": "1cm",
            "left": "1cm",
        },
    )
    report_page.close()


def capture_viewer(args):
    screenshot_path = Path(args.screenshot)
    screenshot_path.parent.mkdir(parents=True, exist_ok=True)

    pdf_path = Path(args.pdf) if args.pdf else None
    if pdf_path:
        pdf_path.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not args.headed)

        page = browser.new_page(
            viewport={
                "width": args.width,
                "height": args.height,
            },
            device_scale_factor=args.device_scale_factor,
        )

        tile_watcher = TileIdleWatcher(page)

        print(f"Opening viewer: {args.url}")
        page.goto(args.url, wait_until="domcontentloaded")

        page.wait_for_selector("#slideSelect", timeout=30000)

        page.wait_for_function(
            """() => {
                const select = document.querySelector("#slideSelect");
                return select && select.options && select.options.length > 0;
            }""",
            timeout=30000,
        )

        available_slides = page.eval_on_selector(
            "#slideSelect",
            """select => Array.from(select.options).map(option => option.value)"""
        )

        print(f"Available slides: {available_slides}")

        if args.slide:
            if args.slide not in available_slides:
                raise RuntimeError(
                    f"Slide '{args.slide}' not found. Available slides: {available_slides}"
                )

            print(f"Selecting slide: {args.slide}")
            page.select_option("#slideSelect", args.slide)
            page.click("#loadButton")


        wait_for_viewer_ready(page, tile_watcher)

        if args.run_inference:
            print("Running AI inference from viewer...")
            page.click("#runInferenceButton")

            page.wait_for_function(
                """() => {
                    const status = document.querySelector("#status");
                    if (!status) return false;
                    return (
                      status.textContent.includes("AI inference complete") ||
                      status.textContent.includes("AI inference failed")
                    );
                }""",
                timeout=args.inference_timeout_ms,
            )

            status_text = page.locator("#status").inner_text()

            if "AI inference failed" in status_text:
                inspector_text = page.locator("#inspectorContent").inner_text()
                raise RuntimeError(
                    f"AI inference failed. Inspector message:\n{inspector_text}"
                )

            tile_watcher.wait_for_idle(timeout_ms=60000)

        if args.heatmap:
            heatmap_toggle = page.locator("#heatmapToggle")

            if heatmap_toggle.is_disabled():
                print("Heatmap toggle is disabled. No heatmap available for this slide.")
            else:
                print("Turning on heatmap overlay...")
                if not heatmap_toggle.is_checked():
                    heatmap_toggle.check()

                set_range_value(page, "#opacitySlider", str(args.opacity))
                tile_watcher.wait_for_idle(timeout_ms=60000)

        if args.inspect_center:
            print("Clicking center of viewer for prediction inspection...")
            viewer_box = page.locator("#viewer").bounding_box()

            if viewer_box:
                page.mouse.click(
                    viewer_box["x"] + viewer_box["width"] / 2,
                    viewer_box["y"] + viewer_box["height"] / 2,
                )

                try:
                    page.wait_for_selector(
                        "#inspectorPanel:not(.panel-hidden)",
                        timeout=10000,
                    )
                except PlaywrightTimeoutError:
                    print("Warning: inspector panel did not open.")

        print(f"Saving screenshot: {screenshot_path}")
        page.locator("#viewer").screenshot(path=str(screenshot_path))

        if pdf_path:
            print(f"Saving PDF report: {pdf_path}")
            title = args.report_title or f"WSI Viewer Snapshot - {args.slide or 'slide'}"
            create_pdf_report(browser, screenshot_path, pdf_path, title)

        if args.headed:
            print("Browser is visible. Waiting 35 seconds before closing...")
            page.wait_for_timeout(35000)


        browser.close()

    print("Done.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Capture ZFP WSI viewer screenshots and optional PDF report."
    )

    parser.add_argument(
        "--url",
        default="http://localhost:8000/viewer",
        help="Viewer URL.",
    )

    parser.add_argument(
        "--slide",
        default=None,
        help="Slide filename from the viewer dropdown, e.g. tumor_005.tif.",
    )

    parser.add_argument(
        "--screenshot",
        default="outputs/viewer_snapshots/wsi_viewer_snapshot.png",
        help="Output screenshot path.",
    )

    parser.add_argument(
        "--pdf",
        default=None,
        help="Optional PDF report output path.",
    )

    parser.add_argument(
        "--report-title",
        default=None,
        help="Optional PDF report title.",
    )

    parser.add_argument(
        "--heatmap",
        action="store_true",
        help="Turn on AI heatmap overlay before capture.",
    )

    parser.add_argument(
        "--opacity",
        type=float,
        default=0.45,
        help="Heatmap opacity from 0 to 1.",
    )

    parser.add_argument(
        "--run-inference",
        action="store_true",
        help="Click Run AI Inference before capture.",
    )

    parser.add_argument(
        "--inference-timeout-ms",
        type=int,
        default=20 * 60 * 1000,
        help="Max time to wait for inference.",
    )

    parser.add_argument(
        "--inspect-center",
        action="store_true",
        help="Click the center of the viewer before capture.",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=1920,
        help="Browser viewport width.",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=1080,
        help="Browser viewport height.",
    )

    parser.add_argument(
        "--device-scale-factor",
        type=float,
        default=1,
        help="Browser device scale factor.",
    )

    parser.add_argument(
        "--headed",
        action="store_true",
        help="Run browser visibly instead of headless.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    capture_viewer(parse_args())

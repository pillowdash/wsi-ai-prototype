import time

from playwright.sync_api import Page, expect


TILE_MARKERS = (
    "/tiles/",
    "/heatmap-tiles/",
)


class TileIdleWatcher:
    """
    Watches OpenSeadragon tile requests and waits until tile loading becomes quiet.
    This is more reliable than waiting only for page load because WSI tiles continue
    loading after the initial HTML page is ready.
    """

    def __init__(self, page: Page):
        self.page = page
        self.active = 0
        self.last_activity = time.time()

        page.on("request", self._on_request)
        page.on("requestfinished", self._on_done)
        page.on("requestfailed", self._on_done)

    def _is_tile_request(self, request):
        return any(marker in request.url for marker in TILE_MARKERS)

    def _on_request(self, request):
        if self._is_tile_request(request):
            self.active += 1
            self.last_activity = time.time()

    def _on_done(self, request):
        if self._is_tile_request(request):
            self.active = max(0, self.active - 1)
            self.last_activity = time.time()

    def wait_for_idle(self, quiet_ms=1200, timeout_ms=30000):
        deadline = time.time() + timeout_ms / 1000

        while time.time() < deadline:
            quiet_for_ms = (time.time() - self.last_activity) * 1000

            if self.active == 0 and quiet_for_ms >= quiet_ms:
                return

            self.page.wait_for_timeout(150)

        raise TimeoutError(
            f"WSI tile requests did not settle in time. active={self.active}"
        )


def wait_for_slide_options(page: Page):
    """
    Do not wait for '#slideSelect option' to be visible.
    HTML option elements are often not considered visible by Playwright unless
    the dropdown is open. Instead, wait until the select has options.
    """

    page.wait_for_selector("#slideSelect", timeout=30000)

    page.wait_for_function(
        """() => {
            const select = document.querySelector("#slideSelect");
            return select && select.options && select.options.length > 0;
        }""",
        timeout=30000,
    )


def get_available_slides(page: Page):
    return page.locator("#slideSelect").evaluate(
        """select => Array.from(select.options).map(option => option.value)"""
    )


def wait_for_viewer_open(page: Page):
    page.wait_for_selector("#viewer canvas", timeout=30000)

    page.wait_for_function(
        """() => {
            const status = document.querySelector("#status");
            if (!status) return false;

            const text = status.textContent || "";

            return (
                text.includes("DZ levels") ||
                text.includes("loaded") ||
                text.includes(".tif") ||
                text.includes(".svs")
            );
        }""",
        timeout=30000,
    )


def open_slide(page: Page, viewer_url: str, slide_name: str):
    watcher = TileIdleWatcher(page)

    page.goto(viewer_url, wait_until="domcontentloaded")

    expect(page.locator("#viewer")).to_be_visible(timeout=30000)
    expect(page.locator("#slideSelect")).to_be_visible(timeout=30000)
    expect(page.locator("#loadButton")).to_be_visible(timeout=30000)

    wait_for_slide_options(page)

    available_slides = get_available_slides(page)

    assert slide_name in available_slides, (
        f"Slide '{slide_name}' not found in dropdown. "
        f"Available slides: {available_slides}"
    )

    page.select_option("#slideSelect", slide_name)
    page.click("#loadButton")

    wait_for_viewer_open(page)

    watcher.wait_for_idle()

    return watcher


def set_range_value(page: Page, selector: str, value: str):
    page.eval_on_selector(
        selector,
        """(element, value) => {
            element.value = value;
            element.dispatchEvent(new Event("input", { bubbles: true }));
            element.dispatchEvent(new Event("change", { bubbles: true }));
        }""",
        value,
    )


def test_viewer_loads_slide(page: Page, viewer_url: str, slide_name: str):
    open_slide(page, viewer_url, slide_name)

    expect(page.locator("#viewer canvas").nth(0)).to_be_visible()
    expect(page.locator("#slideSelect")).to_have_value(slide_name)


def test_heatmap_toggle_requests_heatmap_tiles(
    page: Page,
    viewer_url: str,
    slide_name: str,
):
    open_slide(page, viewer_url, slide_name)

    heatmap_toggle = page.locator("#heatmapToggle")
    expect(heatmap_toggle).to_be_visible()

    assert not heatmap_toggle.is_disabled(), (
        "Heatmap toggle is disabled. "
        "Run inference first or make sure the prediction CSV exists for this slide."
    )

    with page.expect_response(
        lambda response: "/heatmap-tiles/" in response.url,
        timeout=30000,
    ) as response_info:
        heatmap_toggle.check()

    response = response_info.value

    assert response.status == 200, (
        f"Expected heatmap tile response 200, got {response.status}: "
        f"{response.url}"
    )

    expect(heatmap_toggle).to_be_checked()


def test_heatmap_opacity_slider_updates_value(
    page: Page,
    viewer_url: str,
    slide_name: str,
):
    open_slide(page, viewer_url, slide_name)

    heatmap_toggle = page.locator("#heatmapToggle")
    opacity_slider = page.locator("#opacitySlider")

    expect(heatmap_toggle).to_be_visible()
    expect(opacity_slider).to_be_visible()

    assert not heatmap_toggle.is_disabled(), (
        "Heatmap toggle is disabled. "
        "Run inference first or make sure the prediction CSV exists for this slide."
    )

    heatmap_toggle.check()
    set_range_value(page, "#opacitySlider", "0.25")

    expect(opacity_slider).to_have_value("0.25")


def test_prediction_inspector_opens_on_click(
    page: Page,
    viewer_url: str,
    slide_name: str,
):
    open_slide(page, viewer_url, slide_name)

    viewer_box = page.locator("#viewer").bounding_box()
    assert viewer_box is not None

    page.mouse.click(
        viewer_box["x"] + viewer_box["width"] / 2,
        viewer_box["y"] + viewer_box["height"] / 2,
    )

    expect(page.locator("#inspectorPanel")).to_be_visible(timeout=10000)




import os
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--wsi-viewer-url",
        action="store",
        default=None,
        help="URL for the ZFP WSI viewer",
    )

    parser.addoption(
        "--wsi-slide",
        action="store",
        default=None,
        help="Slide filename available in the viewer dropdown",
    )


@pytest.fixture
def viewer_url(request):
    return (
        request.config.getoption("--wsi-viewer-url")
        or os.getenv("WSI_VIEWER_URL")
        or "http://localhost:8000/viewer"
    )


@pytest.fixture
def slide_name(request):
    return (
        request.config.getoption("--wsi-slide")
        or os.getenv("WSI_SLIDE")
        or "tumor_005.tif"
    )

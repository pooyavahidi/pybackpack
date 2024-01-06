import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--vscode-launch",
        action="store_true",
        default=False,
        help="Indicates if tests are being run from VSCode's debugger",
    )


@pytest.fixture
def in_vscode_launch(request):
    """Return True if running in VSCode's debugger."""
    return request.config.getoption("--vscode-launch", default=False)

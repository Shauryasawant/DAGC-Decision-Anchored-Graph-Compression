import re
from pathlib import Path

import dagc
import dagc_eval


def test_release_version_matches_package_metadata():
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    metadata_version = re.search(
        r'^version = "([^"]+)"$', pyproject.read_text(), re.MULTILINE
    ).group(1)
    assert dagc.__version__ == "0.1.13"
    assert dagc_eval.__version__ == dagc.__version__
    assert dagc.__version__ == metadata_version

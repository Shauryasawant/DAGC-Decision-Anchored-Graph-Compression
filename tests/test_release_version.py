from importlib.metadata import version

import dagc
import dagc_eval


def test_release_version_matches_package_metadata():
    assert dagc.__version__ == "0.1.12"
    assert dagc_eval.__version__ == dagc.__version__
    assert dagc.__version__ == version("dagc")

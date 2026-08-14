from importlib.metadata import version

import dagc
import dagc_eval


def test_release_version_matches_package_metadata():
    # Keep this test anchored to the current release version; bump when
    # releasing a new version.
    assert dagc.__version__ == "0.1.8"
    assert dagc_eval.__version__ == dagc.__version__
    assert dagc.__version__ == version("dagc")

"""RTL -> GDSII and PPA, through OpenROAD (ORFS) only.

There is no second backend. If OpenROAD is unavailable or the flow fails,
this module raises; it never estimates PPA from anything else.
"""

from .orfs import DEFAULT_IMAGE, ORFSConfig, ORFSResult, check_image, run_orfs

__all__ = ["DEFAULT_IMAGE", "ORFSConfig", "ORFSResult", "check_image", "run_orfs"]

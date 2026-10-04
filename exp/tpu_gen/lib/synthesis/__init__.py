"""Design staging and pre-flight elaboration.

This module never reports PPA. Area, timing and power come from OpenROAD
only; here we just make sure what we hand OpenROAD actually elaborates, and
lay the sources out the way the ORFS flow expects.
"""

from .stage import ElaborationResult, elaborate, stage_sources

__all__ = ["ElaborationResult", "elaborate", "stage_sources"]

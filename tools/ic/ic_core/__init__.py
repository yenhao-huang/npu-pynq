"""ic_core -- the tools themselves.

Nothing in this package knows that an agent or a daemon exists. Each category
is directly importable, directly testable, and directly usable by a person at
a shell.
"""

from .dispatch import dispatch  # noqa: F401
from .errors import IcError  # noqa: F401

__version__ = "0.1.0"

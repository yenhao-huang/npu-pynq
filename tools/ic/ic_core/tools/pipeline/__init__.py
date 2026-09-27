"""Paper-specific compositions of existing registry operations."""
from ...registry import Category, Op, register_category

CATEGORY = register_category(Category(name='pipeline', summary='Paper-informed tool compositions.',
                                     ops=[], default_backend='rtlrewriter'))
from . import rtlrewriter  # noqa: E402,F401

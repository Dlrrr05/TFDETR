"""Dedicated matcher registration for the axial + low-rank experiment bundle."""

from ...core import register
from ..rtdetr.matcher import HungarianMatcher


@register()
class HungarianMatcherAxialLowRank(HungarianMatcher):
    """A named registration alias so the experiment can live in its own config namespace."""

    pass

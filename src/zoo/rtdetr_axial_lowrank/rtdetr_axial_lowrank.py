"""Dedicated RT-DETR registration for the axial + low-rank experiment bundle."""

from ...core import register
from ..rtdetr.rtdetr import RTDETR


@register()
class RTDETRAxialLowRank(RTDETR):
    """Alias of the current RT-DETR model with a dedicated experiment-facing name."""

    pass

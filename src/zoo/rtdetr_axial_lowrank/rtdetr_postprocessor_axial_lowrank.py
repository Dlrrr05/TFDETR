"""Dedicated postprocessor registration for the axial + low-rank experiment bundle."""

from ...core import register
from ..rtdetr.rtdetr_postprocessor import RTDETRPostProcessor


@register()
class RTDETRPostProcessorAxialLowRank(RTDETRPostProcessor):
    """Alias of the current RT-DETR postprocessor used by the dedicated bundle."""

    pass

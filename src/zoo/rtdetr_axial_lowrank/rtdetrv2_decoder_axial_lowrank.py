"""Dedicated decoder registration for the axial + low-rank experiment bundle."""

from ...core import register
from ..rtdetr.rtdetrv2_decoder import RTDETRTransformerv2


@register()
class RTDETRTransformerv2AxialLowRank(RTDETRTransformerv2):
    """Alias of the current RT-DETR v2 decoder for the axial + low-rank pipeline."""

    pass

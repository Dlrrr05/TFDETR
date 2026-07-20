"""Dedicated criterion registration for the axial + low-rank experiment bundle."""

from ...core import register
from ..rtdetr.rtdetrv2_criterion import RTDETRCriterionv2


@register()
class RTDETRCriterionv2AxialLowRank(RTDETRCriterionv2):
    """Alias of the current structured low-rank criterion used by the dedicated bundle."""

    pass

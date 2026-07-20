"""Dedicated encoder registration for the axial + low-rank experiment bundle."""

from ...core import register
from ..rtdetr.hybrid_encoder import HybridEncoder


@register()
class HybridEncoderAxialLowRank(HybridEncoder):
    """Alias of the current HybridEncoder used by the axial + low-rank pipeline."""

    pass

# RT-DETR Axial Low-Rank Bundle

This directory groups the experiment-facing registrations for the
`AxialCNNTransformerBackbone + structured low-rank regularization` setup.

Why this folder exists:

- keep the axial + low-rank experiment in its own config namespace
- avoid reusing the generic `RTDETR / HybridEncoder / RTDETRCriterionv2` names
- make it easier to copy only the dedicated experiment entrypoints to another machine

The actual implementation logic still lives in the current mainline modules:

- `src/nn/backbone/axial_cnn_transformer_backbone.py`
- `src/zoo/rtdetr/rtdetr.py`
- `src/zoo/rtdetr/hybrid_encoder.py`
- `src/zoo/rtdetr/rtdetrv2_decoder.py`
- `src/zoo/rtdetr/rtdetrv2_criterion.py`
- `src/zoo/rtdetr/rtdetr_postprocessor.py`

Suggested config:

- `rtdetrv2_axial_lowrank_bundle.yml`

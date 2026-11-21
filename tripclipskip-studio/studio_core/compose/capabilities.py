"""Conservative, read-only capability discovery for A1111 wrappers.

Detecting a standard SDXL wrapper does NOT prove that the model's ADM
vector layout is known. No pooled or geometry slicing is authorized here.
"""
from __future__ import annotations
from dataclasses import dataclass
from .errors import UnsupportedModelFamily


@dataclass(frozen=True)
class ModelCapabilities:
    family: str                          # sd15 | sdxl
    width_l: int = 768
    width_g: int | None = None
    pooled_width: int | None = None
    chunk_length: int = 77
    named_pooled_available: bool = False  # must be observed from actual G result
    geometry_layout_verified: bool = False
    conditioner_layout_id: str | None = None
    model_fingerprint: str | None = None

    def __post_init__(self):
        if self.family not in ('sd15', 'sdxl'):
            raise UnsupportedModelFamily('Only SD1.x and SDXL are in scope')
        if self.width_l != 768 or self.chunk_length != 77:
            raise UnsupportedModelFamily('Unverified CLIP-L width/chunk capacity')
        if self.family == 'sdxl':
            if self.width_g != 1280 or self.pooled_width != 1280:
                raise UnsupportedModelFamily('Unverified SDXL CLIP-G widths')
        else:
            if self.width_g is not None or self.pooled_width is not None or self.named_pooled_available:
                raise UnsupportedModelFamily('SD1.x cannot expose an SDXL G/pooled channel')
        if self.geometry_layout_verified and not self.conditioner_layout_id:
            raise UnsupportedModelFamily('geometry layout requires a verified layout identifier')


def _is_l_encoder(emb):
    return callable(getattr(emb, 'encode_with_transformers', None)) and \
        getattr(getattr(emb, 'wrapped', None), 'transformer', None) is not None


def _is_g_encoder(emb):
    return callable(getattr(emb, 'encode_with_transformers', None)) and \
        getattr(getattr(getattr(emb, 'wrapped', None), 'model', None), 'transformer', None) is not None


def probe_a1111_model(model) -> ModelCapabilities:
    """Do not call forward or mutate an installed checkpoint.

    This detects wrapper families only; it never claims named pooled, token
    alignment, concrete layout of vector, or geometry ownership.
    """
    if model is None or getattr(model, 'is_sd2', False) or getattr(model, 'is_sd3', False):
        raise UnsupportedModelFamily('SD2/SD3/unknown models are unsupported')
    if getattr(model, 'is_sdxl', False):
        cond = getattr(model, 'conditioner', None)
        embedders = getattr(cond, 'embedders', None)
        if not isinstance(embedders, (list, tuple)):
            raise UnsupportedModelFamily('SDXL conditioner embedders unavailable')
        l = [e for e in embedders if _is_l_encoder(e)]
        g = [e for e in embedders if _is_g_encoder(e)]
        if len(l) != 1 or len(g) != 1 or l[0] is g[0]:
            raise UnsupportedModelFamily('Expected exactly one A1111 CLIP-L and one OpenCLIP-G')
        return ModelCapabilities('sdxl', width_g=1280, pooled_width=1280)
    emb = getattr(model, 'cond_stage_model', None)
    if not _is_l_encoder(emb):
        raise UnsupportedModelFamily('Unknown SD1.x CLIP wrapper')
    return ModelCapabilities('sd15')

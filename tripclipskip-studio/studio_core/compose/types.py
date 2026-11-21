"""Runtime-only dataclasses for SD1.x and SDXL conditioning. No tensor persistence."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SourceRef:
    id: str
    branch: str                         # positive | negative
    text: str                           # runtime ONLY; do not serialize into PNG
    model_fingerprint: str
    tokenizer_fingerprint: str
    lora_ti_fingerprint: str

    def __post_init__(self):
        if not self.id or not isinstance(self.id, str) or len(self.id) > 64:
            raise ValueError('SourceRef.id must be a nonempty <=64-character string')
        if self.branch not in ('positive', 'negative'):
            raise ValueError('SourceRef.branch must be positive or negative')
        if not isinstance(self.text, str):
            raise ValueError('SourceRef.text must be text')
        for name in ('model_fingerprint', 'tokenizer_fingerprint', 'lora_ti_fingerprint'):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f'{name} must be provided to avoid cross-model reuse')


@dataclass(frozen=True)
class TokenMap:
    source_id: str
    encoder: str
    chunks: tuple[Any, ...] = ()
    exact: bool = False

    def __post_init__(self):
        if self.encoder not in ('l', 'g'):
            raise ValueError('unknown token encoder')
        if not isinstance(self.exact, bool):
            raise ValueError('TokenMap.exact must be boolean')


@dataclass(frozen=True)
class ConditioningChannels:
    """Borrowed named tensors. Object freezing does NOT freeze nested tensors.

    `source_ref` is legacy single-source provenance. Per-channel records enable
    split prompts without losing source identity; they never authorize pooled
    replacement or micro-conditioning edits on their own.
    """
    l: Any = None
    g: Any = None
    pooled_g: Any = None
    microcond: Any = None                 # opaque; no inferred vector slicing
    native: Any = None                    # original native reference / provenance
    branch: str = 'positive'
    source_id: str = ''
    source_ref: SourceRef | None = None
    l_source: SourceRef | None = None
    g_source: SourceRef | None = None
    pooled_source: SourceRef | None = None

    def source_for(self, channel: str) -> SourceRef | None:
        if channel not in ('l', 'g', 'pooled_g'):
            raise ValueError('Unknown channel')
        return {'l': self.l_source, 'g': self.g_source,
                'pooled_g': self.pooled_source}[channel] or self.source_ref


@dataclass(frozen=True)
class CompositionOp:
    op: str
    input_ids: tuple[str, ...]
    params: dict[str, Any] = field(default_factory=dict)
    channel: str = 'l'


@dataclass(frozen=True)
class Recipe:
    format: str = 'tcs-recipe-v1'
    model_family: str = 'sdxl'
    sources: tuple[SourceRef, ...] = ()
    operations: tuple[CompositionOp, ...] = ()
    phase_policy: Any = None

"""TCS WP-01: inert, strictly typed conditioning contracts.

Nothing in this package patches A1111, registers hooks, or runs CLIP by import.
It is a CPU-tested foundation; not a generation feature or pooled-G injection.
"""

from .errors import ComposeError, UnsupportedModelFamily, PooledLayoutUnknown, ComposerOwnerConflict
from .types import SourceRef, TokenMap, ConditioningChannels, CompositionOp, Recipe
from .capabilities import ModelCapabilities, probe_a1111_model
from .validation import validate_channels, validate_compatible, assemble_cross
from .layers import LayerSample, mix_two_layers, preview_bundle_layer_mix, snapshot_layer_sample, validate_observed_layer_index
from .signal_diagnostics import LayerSignalDiagnostics, diagnose_layer_signal

__all__ = [
    'ComposeError', 'UnsupportedModelFamily', 'PooledLayoutUnknown',
    'ComposerOwnerConflict', 'SourceRef', 'TokenMap', 'ConditioningChannels',
    'CompositionOp', 'Recipe', 'ModelCapabilities', 'probe_a1111_model',
    'validate_channels', 'validate_compatible', 'assemble_cross',
    'LayerSample', 'mix_two_layers', 'preview_bundle_layer_mix',
    'snapshot_layer_sample', 'validate_observed_layer_index',
    'LayerSignalDiagnostics', 'diagnose_layer_signal',
]

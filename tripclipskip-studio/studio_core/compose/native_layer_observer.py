"""WP-03B0: bounded read-only *same-call* A1111 CLIP layer observation.

This module never invokes a transformer on its own, retains no tensors after
its callbacks, and never patches the CLIP wrapper. The *existing* Native Proof
wrapper is its sole invocation owner. Hook outputs describe executed modules,
NOT verified A1111 native conditioning or known positive/negative branches.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

MAX_OBSERVED_LAYERS = 64


@dataclass(frozen=True)
class LayerHookTarget:
    index: int
    coordinate: str
    stage: str
    module: torch.nn.Module


def discover_layer_targets(wrapper, encoder: str):
    """Find supported *actual* torch modules without assuming model depth.

    A1111 SD1.x and SDXL CLIP-L use HF text_model.encoder.layers;
    OpenCLIP-G uses model.transformer.resblocks. Unknown adapters reject
    observation (but the native wrapper continues unchanged).
    """
    if encoder == 'l':
        transformer = getattr(getattr(wrapper, 'wrapped', None), 'transformer', None)
        text_model = getattr(transformer, 'text_model', None)
        blocks = getattr(getattr(text_model, 'encoder', None), 'layers', None)
        final_norm = getattr(text_model, 'final_layer_norm', None)
        coordinate = 'hf_text_model.encoder.layers.zero_based_output'
        norm_coordinate = 'hf_text_model.final_layer_norm.executed'
    elif encoder == 'g':
        model = getattr(getattr(wrapper, 'wrapped', None), 'model', None)
        blocks = getattr(getattr(model, 'transformer', None), 'resblocks', None)
        final_norm = getattr(model, 'ln_final', None)
        coordinate = 'openclip_transformer.resblocks.zero_based_output'
        norm_coordinate = 'openclip_model.ln_final.executed'
    else:
        return (), 'UNKNOWN_ENCODER'
    if blocks is None:
        return (), 'UNSUPPORTED_WRAPPER_LAYOUT'
    try:
        size = len(blocks)
    except (TypeError, AttributeError):
        return (), 'UNSUPPORTED_BLOCK_COLLECTION'
    if not (0 < size <= MAX_OBSERVED_LAYERS):
        return (), 'UNSUPPORTED_LAYER_COUNT'
    targets = []
    try:
        for index in range(size):
            block = blocks[index]
            if not isinstance(block, torch.nn.Module):
                return (), 'NON_MODULE_LAYER'
            targets.append(LayerHookTarget(index, coordinate, 'block_output', block))
    except (KeyError, AttributeError, IndexError, TypeError):
        return (), 'UNSUPPORTED_BLOCK_COLLECTION'
    if isinstance(final_norm, torch.nn.Module):
        targets.append(LayerHookTarget(-1, norm_coordinate, 'final_norm_if_executed', final_norm))
    return tuple(targets), 'SUPPORTED_LAYOUT_NOT_NATIVE_VERIFIED'


def first_tensor(value):
    """Only named PyTorch tensor outputs; never traverse arbitrary objects."""
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)) and value and isinstance(value[0], torch.Tensor):
        return value[0]
    return None


class LayerCaptureScope:
    """Per-invocation hook scope with eager, metadata-only callbacks.

    `record` is called while the native forward executes, using a detached
    tensor view but never retaining a reference to it in this class.
    """
    def __init__(self, targets, record: Callable):
        self.targets = tuple(targets)
        self.record = record
        self.rows = []
        self.errors = []
        self._handles = []

    def __enter__(self):
        try:
            for target in self.targets:
                def hook(_module, _inputs, output, target=target):
                    try:
                        tensor = first_tensor(output)
                        if tensor is None:
                            self.errors.append('NON_TENSOR_OUTPUT:' + target.stage + ':' + str(target.index))
                            return None
                        metadata = self.record(tensor.detach())
                        if not isinstance(metadata, dict):
                            raise TypeError('recorder must return metadata dict')
                        self.rows.append({
                            'layer_index': target.index,
                            'stage': target.stage,
                            'coordinate': target.coordinate,
                            'tensor': metadata,
                        })
                    except Exception as exc:
                        self.errors.append('RECORD_ERROR:' + type(exc).__name__)
                    # Never substitute a module result.
                    return None
                self._handles.append(target.module.register_forward_hook(hook))
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, _type, _exc, _tb):
        for handle in reversed(self._handles):
            try:
                handle.remove()
            except Exception as exc:
                self.errors.append('HOOK_REMOVE_ERROR:' + type(exc).__name__)
        self._handles.clear()
        return False

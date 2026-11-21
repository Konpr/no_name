"""Explicit opt-in, read-only A1111 native-conditioning GPU/CPU proof capture.

No code path here modifies conditioning, calls an encoder again, or writes tensor
values. SHA-256 of complete tensor bytes allows comparing two *separate* runs
for exact equality. A matching digest is not evidence of an identical image,
identical U-Net internals, or a genuinely equivalent pipeline on its own.

Enabled ONLY when TCS_PROOF_MODE=shapes or sha256 AND TCS_PROOF_DIR is set.
Capturing CUDA tensors to CPU causes synchronization and performance overhead.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from threading import RLock
from datetime import datetime, timezone
from contextvars import ContextVar
from hashlib import sha256
from math import sqrt, isfinite
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

from .native_layer_observer import discover_layer_targets, LayerCaptureScope
from .native_layer_diagnostics import summarize_layer_metadata, assess_observed_layer_rows
from .branch_provenance import (
    ConstructionScope, classify_construction_request, classify_construction_pass, scope_metadata,
)
from .native_encoder_fingerprint import (
    describe_encoder_capabilities, observe_encoder_output_contract,
)
from .native_token_provenance import (
    chunk_input_summary, ti_fixes_summary, observe_prompt_markers,
)
from secrets import token_bytes

import torch
from .runtime import JobHookLease

PROOF_SCHEMA = 'tcs-native-proof-v1'
MODES = frozenset(('shapes', 'sha256'))
MAX_ELEMENTS = 4_000_000
# Default limits preserve G0 behavior; hard caps bound opt-in tracing.
DEFAULT_MAX_EVENTS = 64
MAX_EVENTS = 128
MAX_LEAVES = 24
MAX_DEPTH = 8
DEFAULT_FORWARD_CALLS = 2
MAX_FORWARD_CALLS = 32
DEFAULT_LAYER_INVOCATIONS = 8  # opt-in: 4 base + 4 Hires SDXL TE invocations
MAX_DENOISER_INSTANCES = 4
MAX_LAYER_INVOCATIONS = 16


@dataclass(frozen=True)
class ProofConfig:
    mode: str
    directory: Path
    max_elements: int = MAX_ELEMENTS
    max_events: int = DEFAULT_MAX_EVENTS
    max_forward_calls: int = DEFAULT_FORWARD_CALLS
    capture_layers: bool = False
    max_layer_invocations: int = DEFAULT_LAYER_INVOCATIONS

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError('TCS proof mode must be shapes or sha256')
        limits = (
            (self.max_elements, MAX_ELEMENTS),
            (self.max_events, MAX_EVENTS),
            (self.max_forward_calls, MAX_FORWARD_CALLS),
            (self.max_layer_invocations, MAX_LAYER_INVOCATIONS),
        )
        if any(type(value) is not int or not (0 < value <= cap) for value, cap in limits):
            raise ValueError('Invalid proof limits')
        if type(self.capture_layers) is not bool:
            raise ValueError('capture_layers must be boolean')


def config_from_env(env=None):
    env = os.environ if env is None else env
    mode = str(env.get('TCS_PROOF_MODE', '')).strip().lower()
    if not mode or mode in ('off', '0'):
        return None
    if mode not in MODES:
        raise ValueError('TCS_PROOF_MODE must be off, shapes or sha256')
    layer_flag = str(env.get('TCS_PROOF_LAYERS', '0')).strip()
    if layer_flag not in ('0', '1'):
        raise ValueError('TCS_PROOF_LAYERS must be 0 or 1')
    directory = str(env.get('TCS_PROOF_DIR', '')).strip()
    if not directory:
        raise ValueError('TCS_PROOF_DIR is required for opt-in proof capture')
    def optional_limit(name, default):
        raw = env.get(name)
        if raw is None:
            return default
        try:
            # Reject floats and expressions, including non-ASCII numerals.
            value = str(raw).strip()
            if not value or not value.isascii() or not value.isdecimal():
                raise ValueError
            return int(value)
        except (ValueError, TypeError):
            raise ValueError(f'{name} must be a positive integer') from None

    return ProofConfig(
        mode, Path(directory).expanduser(),
        max_events=optional_limit('TCS_PROOF_MAX_EVENTS', DEFAULT_MAX_EVENTS),
        max_forward_calls=optional_limit('TCS_PROOF_MAX_FORWARD_CALLS', DEFAULT_FORWARD_CALLS),
        capture_layers=(layer_flag == '1'),
        max_layer_invocations=optional_limit('TCS_PROOF_MAX_LAYER_INVOCATIONS', DEFAULT_LAYER_INVOCATIONS),
    )


def _safe_step(x):
    try:
        return int(x) if x is not None else None
    except (ValueError, TypeError, OverflowError):
        return None


def _tensor_record(tensor, mode, max_elements):
    """Describe a detached tensor without persisting raw values or GPU refs."""
    row = {
        'shape': list(tensor.shape),
        'dtype': str(tensor.dtype),
        'device_type': tensor.device.type,
        'element_count': int(tensor.numel()),
        'digest_status': 'NOT_REQUESTED',
    }
    if mode != 'sha256':
        return row
    if tensor.device.type == 'meta' or not tensor.numel():
        row['digest_status'] = 'UNMATERIALIZED_OR_EMPTY'
        return row
    if tensor.numel() > max_elements:
        row['digest_status'] = 'SKIPPED_TOO_LARGE'
        return row
    try:
        with torch.no_grad():
            # View as uint8 handles bf16 and avoids precision-losing casts.
            native = tensor.detach().contiguous().cpu()
            if native.is_floating_point() and not bool(torch.isfinite(native).all()):
                row['digest_status'] = 'NONFINITE'
                return row
            data = native.view(torch.uint8).reshape(-1)
            buffer = data.numpy()
            digest = sha256()
            # Avoid a second huge temporary allocation for the hash.
            view = memoryview(buffer)
            for start in range(0, len(view), 1024 * 1024):
                digest.update(view[start:start + 1024 * 1024])
            row['sha256'] = digest.hexdigest()
            row['digest_status'] = 'COMPLETE'
    except (RuntimeError, TypeError, ValueError) as exc:
        row['digest_status'] = 'ERROR'
        row['error_type'] = type(exc).__name__
    return row



def _layer_record(tensor, mode, max_elements):
    """Optional expensive full-tensor statistics, exclusively in sha256 mode."""
    row = _tensor_record(tensor, mode, max_elements)
    if mode != 'sha256':
        row['stats_status'] = 'NOT_REQUESTED'
        return row
    if row['digest_status'] == 'NONFINITE':
        row['finite'] = False
        row['stats_status'] = 'NONFINITE'
        return row
    if row['digest_status'] != 'COMPLETE':
        row['stats_status'] = row['digest_status']
        return row
    if not tensor.is_floating_point():
        row['stats_status'] = 'NOT_FLOATING'
        return row
    try:
        with torch.no_grad():
            # Opt-in only. Double precision avoids overflow in ordinary FP16/
            # FP32 accumulation; no data or reference is kept after callback.
            native = tensor.detach().cpu().to(dtype=torch.float64)
            finite = bool(torch.isfinite(native).all())
            row['finite'] = finite
            if finite and native.numel():
                rms = sqrt(float(native.square().mean()))
                if isfinite(rms):
                    row['rms'] = rms
                    row['stats_status'] = 'COMPLETE_FULL_TENSOR_FLOAT64'
                else:
                    row['stats_status'] = 'NONFINITE_RMS'
            else:
                row['stats_status'] = 'NONFINITE_OR_EMPTY'
    except (RuntimeError, TypeError, ValueError) as exc:
        row['stats_status'] = 'ERROR:' + type(exc).__name__
    return row


def tensor_records(obj, mode, max_elements, *, max_leaves=MAX_LEAVES):
    """Collect only recognized A1111 data containers; never stringify prompts."""
    rows = []
    seen = set()

    def walk(value, path, depth):
        if depth > MAX_DEPTH or len(rows) >= max_leaves:
            return
        if isinstance(value, torch.Tensor):
            rows.append({'path': path, **_tensor_record(value, mode, max_elements)})
            pooled = getattr(value, 'pooled', None)
            if isinstance(pooled, torch.Tensor):
                walk(pooled, path + '.pooled', depth + 1)
            return
        if value is None or isinstance(value, (str, int, float, bool)):
            return
        ident = id(value)
        if ident in seen:
            return
        seen.add(ident)
        if isinstance(value, dict):
            # Unknown keys intentionally anonymized, and order retained.
            for idx, (key, item) in enumerate(islice(value.items(), 16)):
                safe = key if isinstance(key, str) and key in (
                    'crossattn', 'vector', 'pooled', 'cond', 'conds',
                    'c_crossattn', 'c_adm', 'c_concat') else 'other'
                walk(item, f'{path}.{safe}[{idx}]', depth + 1)
        elif isinstance(value, (tuple, list)):
            for idx, item in enumerate(value[:16]):
                walk(item, f'{path}[{idx}]', depth + 1)
        else:
            # Known conditioning wrappers only; no generic __dict__ introspection.
            for name in ('crossattn', 'vector', 'pooled', 'conds', 'cond', 'schedules', 'batch'):
                try:
                    item = getattr(value, name, None)
                except (AttributeError, RuntimeError, TypeError):
                    continue
                if item is not None:
                    walk(item, f'{path}.{name}', depth + 1)
    walk(obj, 'root', 0)
    return rows


def _method_equivalent(a, b):
    return getattr(a, '__self__', None) is getattr(b, '__self__', None) and (
        getattr(a, '__func__', a) is getattr(b, '__func__', b))


class LiveProofCapture:
    """Scoped observer; records a bounded trace and writes JSON at teardown.

    Hash capture is a diagnostic operation with CUDA synchronization, NOT a
    transparent performance benchmark. Encoder events may carry a scoped A1111
    *construction* branch, but final encoder/pooled/token provenance remains
    unknown; prepared and forward events keep their own branch labels.
    """
    def __init__(self, config: ProofConfig, *, enabled: bool, model_family: str,
                 settings: dict | None = None):
        self.config = config
        self._lease = JobHookLease()
        self._entered = False
        self._closed = False
        self._methods = []
        self._denoisers = []  # strong per-job references, prevent id reuse
        self._forward_calls = 0
        self._layer_invocations = 0
        self._layer_capture_lock = RLock()  # only used by opt-in B0; prevents concurrent hook mixing
        # Per-instance context variables: no global cross-job or cross-thread branch.
        self._in_setup = ContextVar('tcs_proof_setup', default=False)
        self._in_hires = ContextVar('tcs_proof_calculate_hr_conds', default=False)
        self._construction_scope = ContextVar('tcs_proof_construction', default=None)
        self._token_forward = ContextVar('tcs_proof_text_wrapper_forward', default=None)
        self._token_chunk = ContextVar('tcs_proof_native_token_chunk', default=None)
        self._token_fingerprint_key = token_bytes(32)  # ephemeral: NEVER serialized
        self._construction_scopes = []  # bounded metadata only; no prompts/tensors
        self._denoiser_bindings = []  # bounded; no addresses, object IDs or raw values
        self._forward_instance_counts = []  # [object, calls]; job-scoped, identity-safe
        self._forward_pass_counts = []  # [object, base|hires, calls]; bounded, no raw IDs
        self._hires_prepared_observations = 0
        self.errors = []
        self.numeric_alerts = []
        self.events = []
        self.filename = None
        self.report = {
            'schema': PROOF_SCHEMA,
            'id': uuid4().hex,
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'mode': config.mode,
            'tcs': 'ON' if enabled else 'OFF',
            'model_family': model_family,
            'settings': settings or {},
            'capture_limits': {
                'max_events': config.max_events,
                'max_elements_per_tensor': config.max_elements,
                'max_forward_calls': config.max_forward_calls,
                'forward_sample_policy': 'FIRST_N_CALLS_PER_DENOISER_AND_PASS',
                'max_leaves_per_event': MAX_LEAVES,
                'capture_layers': config.capture_layers,
                'max_layer_invocations': config.max_layer_invocations,
            },
            'notes': [
                'No raw embeddings or prompt strings in this report.',
                'SHA-256 complete digests compare identical bytes; no max_abs_diff across runs.',
                'Capture synchronizes CUDA and may change timing/memory usage.',
                'Prepared conditioning does not prove actual use in denoiser.',
                'Forward input is observed before CFGDenoiser internal reconstruction/refiner logic.',
                'Tensor channel/ADM layout is not inferred from vector offsets.',
                'B0 layer hooks see executed module outputs from one wrapper call, not full native provenance.',
                'Encoder final branch remains UNKNOWN; B1b construction_branch records only A1111 setup_conds/get_conds call scope.',
                'Construction branch does NOT prove token/chunk/pooled/REGION composition or denoiser branch.',
                'Construction pass uses A1111 first-cache-slot identity when available; is_hr_pass alone cannot identify precomputed hires conditioning.',
                'If a scoped caching call returns without encoder forward, this is NOT proof of a cache hit.',
                'B1a capability metadata digest is NOT a checkpoint/weights/tokenizer/LoRA fingerprint.',
                'B1a same-invocation linkage does NOT prove positive/negative or token/chunk provenance.',
                'B0.2: nonfinite executed blocks may be outside the observed wrapper root; do not infer unused layers or semantic safety.',
                'B1b-r2: an observed finite layer is NOT a trusted injection source; numerical gate and provenance gate are separate.',
                'B1b-r2: Hires construction is scoped through calculate_hr_conds only when the first cache slot proves hires, even outside setup_conds.',
                'B1b-r2: distinct denoiser instances may appear within one job; each separately observed instance is temporarily patched and restored.',
                'B1b-r2: max_forward_calls samples the first N calls PER observed denoiser instance AND pass; forward_calls_seen is still the total.',
                'Exact SHA matches compare detached serialized tensor bytes, not full A1111 native identity.',
                'B1c: token fingerprints are job-local keyed HMACs of A1111 process_tokens input, BEFORE possible padding rewrite; the secret key is never persisted.',
                'B1c: chunk and TI metadata record observed A1111 wrapper calls, not verified tokens at transformer input or after custom composition.',
                'B1c: matching first-chunk pooled object identity proves only that a particular wrapper forward returned that object; assembled final pooled origin remains UNKNOWN.',
                'B1c: custom parser markers are lexical hints only, NOT parser execution evidence or proof of token/pooled lineage.',
            ],
        }

    def _observe(self, stage, branch, value, *, pass_name='base', invocation_id=None, construction=None):
        if len(self.events) >= self.config.max_events:
            if 'EVENT_CAP' not in self.errors:
                self.errors.append('EVENT_CAP')
            return
        try:
            records = tensor_records(value, self.config.mode, self.config.max_elements)
            if not records:
                self.errors.append(f'NO_TENSORS:{stage}:{branch}')
            event = {'stage': stage, 'branch': branch,
                     'pass': pass_name, 'ordinal': 1 + sum(
                                    event['stage'] == stage and event['branch'] == branch and
                                    event['pass'] == pass_name for event in self.events),
                     'tensors': records}
            if invocation_id is not None:
                event['invocation_id'] = invocation_id
                event['branch_evidence'] = 'UNVERIFIED'
                event.update(scope_metadata(construction))
            self.events.append(event)
        except Exception as exc:
            self.errors.append(f'CAPTURE_ERROR:{stage}:{type(exc).__name__}')

    def _patch(self, obj, name, function):
        original = getattr(obj, name)
        replacement = function(original)
        self._lease.method(obj, name, replacement)
        self._methods.append((obj, name, replacement, original))

    def install(self, p, clip_l=None, clip_g=None, *, on_close=None):
        if self._entered:
            raise RuntimeError('Proof capture already installed')
        self._lease.__enter__()
        self._entered = True
        try:
            for label, wrapper in (('l', clip_l), ('g', clip_g)):
                if wrapper is None:
                    continue
                self._patch(wrapper, 'encode_with_transformers', self._encoder(label, wrapper))
                if self.config.capture_layers:
                    try:
                        supported = (callable(getattr(wrapper, 'process_tokens', None))
                                     and callable(getattr(wrapper, 'forward', None)))
                    except Exception as exc:
                        supported = False
                        self.errors.append('TOKEN_OBSERVER_CAPABILITY_ERROR:' + type(exc).__name__)
                    if supported:
                        # Only an A1111-style wrapper with both methods supports
                        # chunk and pooled lineage. Mocks/unknown adapters stay
                        # on the existing observational path. Optional patches
                        # never prevent the already supported native TE path.
                        try:
                            self._patch(wrapper, 'process_tokens', self._process_tokens(label, wrapper))
                            self._patch(wrapper, 'forward', self._text_forward(label, wrapper))
                        except Exception as exc:
                            self.errors.append('TOKEN_OBSERVER_PATCH_ERROR:' + type(exc).__name__)
            self._patch(p, 'setup_conds', self._setup(p))
            # Strictly opt-in (TCS_PROOF_LAYERS=1); supports user A1111
            # without modifying processing.py or the prompt parser.
            if self.config.capture_layers and callable(getattr(p, 'get_conds_with_caching', None)):
                self._patch(p, 'get_conds_with_caching', self._get_conds_scope(p))
                # A1111 calls calculate_hr_conds() outside setup_conds() for
                # the second pass. Bind ONLY its exact hires cache-slot calls.
                if callable(getattr(p, 'calculate_hr_conds', None)):
                    self._patch(p, 'calculate_hr_conds', self._hires_construction(p))
            if on_close is not None and callable(getattr(p, 'close', None)):
                self._patch(p, 'close', self._close_wrapper(on_close))
        except BaseException:
            self.close()  # fail closed without leaving a partial probe
            raise

    @staticmethod
    def _close_wrapper(on_close):
        def make(original):
            def wrapped(*args, **kwargs):
                try:
                    return original(*args, **kwargs)
                finally:
                    on_close()
            return wrapped
        return make

    def _record_token_event(self, entry):
        if len(self.events) >= self.config.max_events:
            if 'EVENT_CAP:token' not in self.errors:
                self.errors.append('EVENT_CAP:token')
        else:
            self.events.append(entry)

    def _process_tokens(self, label, wrapper):
        """Observe real A1111 per-chunk inputs without tokenizer/TE reruns."""
        def make(original):
            def observed(*args, **kwargs):
                parent = self._token_forward.get()
                # Nested encoder calls must never inherit another encoder's
                # forward/chunk identity (e.g. custom prompt operators).
                if parent is not None and parent['wrapper'] is not wrapper:
                    parent = None
                if parent is not None:
                    parent['chunk_count'] += 1
                index = parent['chunk_count'] if parent is not None else None
                chunk_ref = uuid4().hex
                identity = {'token_chunk_ref': chunk_ref,
                            'token_forward_ref': parent['forward_ref'] if parent else None,
                            'chunk_index_one_based': index}
                # Keep the owning wrapper in ContextVar only; never persist
                # the model object or a Python id in JSON.
                token = self._token_chunk.set((wrapper, identity))
                try:
                    try:
                        ids = args[0] if args else kwargs.get('remade_batch_tokens')
                        weights = args[1] if len(args) > 1 else kwargs.get('batch_multipliers')
                        summary = chunk_input_summary(
                            ids, weights,
                            id_start=getattr(wrapper, 'id_start', None),
                            id_end=getattr(wrapper, 'id_end', None),
                            id_pad=getattr(wrapper, 'id_pad', None),
                            key=self._token_fingerprint_key)
                        ti = ti_fixes_summary(getattr(getattr(wrapper, 'hijack', None), 'fixes', None))
                    except Exception as exc:
                        self.errors.append('TOKEN_INPUT_DIAGNOSTIC_ERROR:' + type(exc).__name__)
                        summary, ti = {'status': 'UNAVAILABLE'}, {'status': 'UNAVAILABLE'}
                    entry = {
                        'stage': 'a1111_token_chunk', 'encoder': label,
                        **identity, **scope_metadata(self._construction_scope.get()),
                        'input': summary, 'textual_inversion': ti,
                        'pooled_from_chunk': {'status': 'NOT_OBSERVED'},
                        'status': 'ACTIVE', 'trusted_final_provenance': False,
                    }
                    try:
                        result = original(*args, **kwargs)
                    except BaseException:
                        entry['status'] = 'NATIVE_EXCEPTION_PROPAGATED'
                        self._record_token_event(entry)
                        raise
                    entry['status'] = 'RETURNED'
                    try:
                        pooled = getattr(result, 'pooled', None) if isinstance(result, torch.Tensor) else None
                        if isinstance(pooled, torch.Tensor):
                            entry['pooled_from_chunk'] = _tensor_record(
                                pooled, self.config.mode, self.config.max_elements)
                            if parent is not None and index == 1:
                                parent['first_chunk_pooled'] = pooled
                    except Exception as exc:
                        self.errors.append('TOKEN_POOLED_DIAGNOSTIC_ERROR:' + type(exc).__name__)
                    self._record_token_event(entry)
                    return result
                finally:
                    self._token_chunk.reset(token)
            return observed
        return make

    def _text_forward(self, label, wrapper):
        """Inspect returned A1111 wrapper pooled; compare object identity only."""
        def make(original):
            def observed(*args, **kwargs):
                parent = {
                    'forward_ref': uuid4().hex, 'chunk_count': 0,
                    'first_chunk_pooled': None, 'wrapper': wrapper,
                }
                token = self._token_forward.set(parent)
                try:
                    result = original(*args, **kwargs)
                except BaseException:
                    self._record_token_event({
                        'stage': 'a1111_text_forward', 'encoder': label,
                        'token_forward_ref': parent['forward_ref'],
                        'chunks_observed': parent['chunk_count'],
                        **scope_metadata(self._construction_scope.get()),
                        'status': 'NATIVE_EXCEPTION_PROPAGATED',
                        'trusted_final_provenance': False,
                    })
                    raise
                else:
                    try:
                        pooled = None
                        if type(result) in (tuple, list) and len(result) == 2:
                            pooled = result[1]
                        elif isinstance(result, torch.Tensor):
                            pooled = getattr(result, 'pooled', None)
                        if not isinstance(pooled, torch.Tensor):
                            pooled = None
                        first = parent['first_chunk_pooled']
                        entry = {
                            'stage': 'a1111_text_forward', 'encoder': label,
                            'token_forward_ref': parent['forward_ref'],
                            'chunks_observed': parent['chunk_count'],
                            **scope_metadata(self._construction_scope.get()),
                            'status': 'RETURNED',
                            'pooled_returned': (_tensor_record(pooled, self.config.mode, self.config.max_elements)
                                                if pooled is not None else {'status': 'NOT_OBSERVED'}),
                            'pooled_object_identity_first_chunk': (
                                pooled is first if pooled is not None and first is not None else None),
                            'pooled_composition_provenance': 'UNKNOWN_AFTER_CUSTOM_PROMPT_OPERATORS',
                            'trusted_final_provenance': False,
                        }
                        self._record_token_event(entry)
                    except Exception as exc:
                        self.errors.append('TEXT_FORWARD_DIAGNOSTIC_ERROR:' + type(exc).__name__)
                    return result
                finally:
                    self._token_forward.reset(token)
                    # Do not retain any pooled tensor after this native forward.
                    parent['first_chunk_pooled'] = None
            return observed
        return make

    def _encoder(self, label, wrapper=None):
        def make(original):
            def observed(*args, **kwargs):
                if self.config.capture_layers:
                    # Only one instrumented wrapper forward at once per job:
                    # hooks from one invocation cannot record another thread.
                    with self._layer_capture_lock:
                        return observed_once(*args, **kwargs)
                return observed_once(*args, **kwargs)

            def observed_once(*args, **kwargs):
                construction = self._construction_scope.get() if self.config.capture_layers else None
                active_chunk = self._token_chunk.get() if self.config.capture_layers else None
                token_chunk_ref = (active_chunk[1] if active_chunk is not None
                                   and active_chunk[0] is wrapper else None)
                if construction is not None:
                    for entry in reversed(self._construction_scopes):
                        if entry['scope_id'] == construction.scope_id:
                            entry['encoder_invocations_observed'] += 1
                            break
                # A fresh invocation ID is created *inside* this actual wrapper
                # call; it is not a trusted positive/negative identity ticket.
                scope = None
                invocation = uuid4().hex if self.config.capture_layers and not self._closed else None
                capability = None
                status = None
                targets = ()
                if self.config.capture_layers and not self._closed:
                    self._layer_invocations += 1
                    if self._layer_invocations > self.config.max_layer_invocations:
                        status = 'INVOCATION_LIMIT'
                    else:
                        try:
                            targets, status = discover_layer_targets(wrapper, label)
                        except Exception as exc:
                            status = 'DISCOVERY_FAILED:' + type(exc).__name__
                            self.errors.append(status)
                        try:
                            capability = describe_encoder_capabilities(wrapper, label, targets, status)
                        except Exception as exc:
                            self.errors.append('CAPABILITY_DIAGNOSTIC_ERROR:' + type(exc).__name__)
                if targets:
                    # Hook installation is optional instrumentation. If it
                    # fails *before* the native call, disable this observation
                    # and still execute original() exactly once.
                    candidate = LayerCaptureScope(targets, lambda tensor: _layer_record(
                        tensor, self.config.mode, self.config.max_elements))
                    try:
                        candidate.__enter__()
                    except Exception as exc:
                        status = 'HOOK_INSTALL_FAILED:' + type(exc).__name__
                        self.errors.append(status)
                        result = original(*args, **kwargs)
                    else:
                        scope = candidate
                        try:
                            result = original(*args, **kwargs)
                        finally:
                            scope.__exit__(None, None, None)
                else:
                    result = original(*args, **kwargs)
                # Any failure in *reporting* must never alter the already
                # completed native encoder output or cause another forward.
                try:
                    self._record_layer_event(label, invocation, status, scope, targets, result, args, capability, construction, token_chunk_ref)
                except Exception as exc:
                    self.errors.append('LAYER_EVENT_ERROR:' + type(exc).__name__)
                self._observe('encoder_' + label, 'unknown', result, invocation_id=invocation, construction=construction if self.config.capture_layers else None)
                if token_chunk_ref is not None and self.events:
                    event = self.events[-1]
                    if event.get('stage') == 'encoder_' + label and event.get('invocation_id') == invocation:
                        event.update(token_chunk_ref)
                return result
            return observed
        return make

    def _record_layer_event(self, label, invocation, status, scope, targets, result, args, capability, construction, token_chunk_ref=None):
        output_contract = observe_encoder_output_contract(result, args[0] if args else None)
        if scope is not None:
            if scope.errors:
                self.errors.extend('LAYER_' + error for error in scope.errors)
            if len(self.events) < self.config.max_events:
                native_outputs = tensor_records(result, self.config.mode, self.config.max_elements)
                summary, alerts = summarize_layer_metadata(label, scope.rows, native_outputs)
                self.numeric_alerts.extend(alerts)
                self.events.append({
                    'stage': 'encoder_layers', 'branch': 'unknown',
                    **scope_metadata(construction),
                    'encoder': label, 'invocation_id': invocation,
                    **(token_chunk_ref or {}),
                    'status': status, 'observed_count': len(scope.rows),
                    'capability_descriptor': capability,
                    'observed_output_contract': output_contract,
                    'trusted_provenance': False,
                    'available_blocks': sum(t.stage == 'block_output' for t in targets),
                    'layers': assess_observed_layer_rows(scope.rows),
                    'native_wrapper_output': native_outputs,
                    'numeric_diagnostics': summary,
                    'token_input': [
                        {'shape': list(arg.shape), 'dtype': str(arg.dtype)}
                        for arg in args[:1] if isinstance(arg, torch.Tensor)
                    ],
                })
            else:
                self.errors.append('EVENT_CAP:encoder_layers')
        elif status is not None and len(self.events) < self.config.max_events:
            if status.startswith(('UNSUPPORTED_', 'NON_MODULE_LAYER', 'UNKNOWN_ENCODER')):
                self.errors.append('LAYER_' + status + ':' + label)
            self.events.append({
                'stage': 'encoder_layers', 'branch': 'unknown',
                **scope_metadata(construction),
                'encoder': label, 'invocation_id': invocation,
                **(token_chunk_ref or {}),
                'status': status, 'observed_count': 0, 'layers': [],
                'capability_descriptor': capability,
                'observed_output_contract': output_contract,
                'trusted_provenance': False,
            })

    def _get_conds_scope(self, p):
        """Bound a verified primary A1111 conditioning-building call.

        This wrapper is read-only; it invokes the original method exactly
        once. Custom REGION work before get_conds is intentionally unknown.
        """
        def make(original):
            def observed(*args, **kwargs):
                in_setup = self._in_setup.get()
                in_hires = self._in_hires.get()
                if self._closed or not (in_setup or in_hires):
                    return original(*args, **kwargs)
                function = args[0] if args else kwargs.get('function')
                prompts = args[1] if len(args) > 1 else kwargs.get('required_prompts')
                try:
                    branch = classify_construction_request(function, prompts)
                    pass_name, pass_evidence = classify_construction_pass(p, args, kwargs)
                    # Outside setup, an arbitrary custom cache call must not
                    # be promoted to positive/negative merely because its
                    # callable happens to match. We require exact HR slot.
                    verified_hr = pass_evidence == 'A1111_CACHE_SLOT_IDENTITY_HIRES'
                    scope = (ConstructionScope(uuid4().hex, branch, pass_name, pass_evidence=pass_evidence)
                             if branch is not None and (in_setup or (in_hires and verified_hr)) else None)
                except Exception:
                    # Diagnostic introspection must not block a native call.
                    scope = None
                if scope is None:
                    return original(*args, **kwargs)
                try:
                    marker_hints = observe_prompt_markers(prompts)
                except Exception as exc:
                    self.errors.append('OPERATOR_MARKER_DIAGNOSTIC_ERROR:' + type(exc).__name__)
                    marker_hints = {'status': 'UNAVAILABLE', 'markers': [], 'evidence': 'NOT_PARSED'}
                entry = {
                    'scope_id': scope.scope_id, 'construction_branch': branch,
                    'pass': scope.pass_name, 'evidence': scope.evidence,
                    'pass_evidence': scope.pass_evidence,
                    'encoder_invocations_observed': 0,
                    'operator_marker_hints': marker_hints,
                    'final_token_pooled_provenance': 'NOT_VERIFIED_AFTER_CONSTRUCTION',
                    'status': 'ACTIVE',
                }
                # Bounded even if the engine calls get_conds repeatedly.
                if len(self._construction_scopes) < self.config.max_events:
                    self._construction_scopes.append(entry)
                elif 'CONSTRUCTION_SCOPE_CAP' not in self.errors:
                    self.errors.append('CONSTRUCTION_SCOPE_CAP')
                token = self._construction_scope.set(scope)
                try:
                    result = original(*args, **kwargs)
                except BaseException:
                    entry['status'] = 'NATIVE_EXCEPTION_PROPAGATED'
                    raise
                else:
                    entry['status'] = 'RETURNED'
                    try:
                        # This is the output of the *A1111 construction call*,
                        # which may contain post-operator composition. It is
                        # NOT proof of which source token/pool produced it or
                        # of what the denoiser will finally consume.
                        entry['construction_result_tensors'] = tensor_records(
                            result, self.config.mode, self.config.max_elements)
                        entry['construction_result_evidence'] = (
                            'A1111_SCOPED_CONSTRUCTION_RETURN_NOT_FINAL_CONSUMPTION')
                    except Exception as exc:
                        self.errors.append('CONSTRUCTION_RESULT_DIAGNOSTIC_ERROR:' + type(exc).__name__)
                    return result
                finally:
                    self._construction_scope.reset(token)
            return observed
        return make

    def _hires_construction(self, p):
        """Observe the A1111 Hires builder, including calls outside setup_conds.

        This scope marks an execution boundary, not ownership of output tokens.
        Its original is called exactly once, with the same exception semantics.
        """
        def make(original):
            def observed(*args, **kwargs):
                token = self._in_hires.set(True)
                try:
                    result = original(*args, **kwargs)
                finally:
                    self._in_hires.reset(token)
                # These are prepared values, NOT a claim of use by U-Net.
                # Do not treat an absent hr_c/uc as a failed generation.
                try:
                    positive = getattr(p, 'hr_c', None)
                    negative = getattr(p, 'hr_uc', None)
                    if positive is not None:
                        self._observe('prepared', 'positive', positive, pass_name='hires')
                        self._hires_prepared_observations += 1
                    if negative is not None:
                        self._observe('prepared', 'negative', negative, pass_name='hires')
                        self._hires_prepared_observations += 1
                except Exception as exc:
                    self.errors.append('HIRES_PREPARED_DIAGNOSTIC_ERROR:' + type(exc).__name__)
                return result
            return observed
        return make

    def _setup(self, p):
        def make(original):
            def observed(*args, **kwargs):
                hires_prepared_before = self._hires_prepared_observations
                token = self._in_setup.set(True)
                try:
                    result = original(*args, **kwargs)
                finally:
                    self._in_setup.reset(token)
                if getattr(p, 'is_hr_pass', False):
                    # A1111's Hires setup_conds() can return after populating
                    # hr_c/hr_uc, leaving c/uc from the BASE pass. Never
                    # mislabel those stale base tensors as prepared hires.
                    if self._hires_prepared_observations == hires_prepared_before:
                        for branch, name in (('positive', 'hr_c'), ('negative', 'hr_uc')):
                            prepared = getattr(p, name, None)
                            if prepared is not None:
                                self._observe('prepared', branch, prepared, pass_name='hires')
                else:
                    self._observe('prepared', 'positive', getattr(p, 'c', None), pass_name='base')
                    self._observe('prepared', 'negative', getattr(p, 'uc', None), pass_name='base')
                return result
            return observed
        return make

    def attach_forward(self, denoiser):
        """Observe each distinct denoiser instance, bounded per job.

        A1111 can replace the denoiser on Hires.fix; the previous implementation
        skipped the new instance and emitted DENOISER_INSTANCE_CHANGED.
        Never stack two proof wrappers onto the same instance. The job lease
        restores all of them, including on interruption.
        """
        if not self._entered or self._closed or denoiser is None:
            return
        if any(existing is denoiser for existing in self._denoisers):
            return
        if len(self._denoisers) >= MAX_DENOISER_INSTANCES:
            if 'DENOISER_INSTANCE_LIMIT' not in self.errors:
                self.errors.append('DENOISER_INSTANCE_LIMIT')
            return
        self._patch(denoiser, 'forward', self._forward(denoiser))
        self._denoisers.append(denoiser)
        try:
            p = getattr(denoiser, 'p', None)
            hr_flag = getattr(p, 'is_hr_pass', None)
        except Exception as exc:
            # Exotic foreign denoisers must not block native inference.
            hr_flag = None
            self.errors.append('DENOISER_BINDING_METADATA_ERROR:' + type(exc).__name__)
        self._denoiser_bindings.append({
            'ordinal': len(self._denoisers),
            'is_hr_pass_flag': hr_flag if type(hr_flag) is bool else None,
            'evidence': 'DENOISER_INSTANCE_OBSERVED_WITH_PROCESSING_HR_FLAG_ONLY',
        })

    def record_forward(self, denoiser, args, kwargs):
        # Controller-owned forward probes also call here; no additional patch.
        self._forward_calls += 1
        counter = next((x for x in self._forward_instance_counts if x[0] is denoiser), None)
        if counter is None:
            if len(self._forward_instance_counts) >= MAX_DENOISER_INSTANCES:
                if 'DENOISER_FORWARD_INSTANCE_LIMIT' not in self.errors:
                    self.errors.append('DENOISER_FORWARD_INSTANCE_LIMIT')
                return
            counter = [denoiser, 0]
            self._forward_instance_counts.append(counter)
        counter[1] += 1
        pass_name = 'hires' if bool(getattr(getattr(denoiser, 'p', None), 'is_hr_pass', False)) else 'base'
        pass_counter = next((x for x in self._forward_pass_counts
                             if x[0] is denoiser and x[1] == pass_name), None)
        if pass_counter is None:
            if len(self._forward_pass_counts) >= 2 * MAX_DENOISER_INSTANCES:
                if 'DENOISER_PASS_SAMPLE_LIMIT' not in self.errors:
                    self.errors.append('DENOISER_PASS_SAMPLE_LIMIT')
                return
            pass_counter = [denoiser, pass_name, 0]
            self._forward_pass_counts.append(pass_counter)
        pass_counter[2] += 1
        # Hires can use a new denoiser OR reuse an earlier instance; sampling
        # only the first N calls of a whole job or instance can silently omit
        # Hires. Keep one bounded sample per (instance, pass) pair.
        if pass_counter[2] <= self.config.max_forward_calls:
            cond = kwargs.get('cond', args[3] if len(args) > 3 else None)
            uncond = kwargs.get('uncond', args[2] if len(args) > 2 else None)
            self._observe('denoiser_input', 'positive', cond, pass_name=pass_name)
            self._observe('denoiser_input', 'negative', uncond, pass_name=pass_name)

    def _forward(self, denoiser):
        def make(original):
            def observed(*args, **kwargs):
                try:
                    self.record_forward(denoiser, args, kwargs)
                except Exception as exc:
                    self.errors.append('DENOISER_RECORD_METADATA_ERROR:' + type(exc).__name__)
                # Never allow read-only diagnosis to skip/duplicate native forward.
                return original(*args, **kwargs)
            return observed
        return make

    def close(self):
        if self._closed:
            return self.filename
        self._closed = True
        if self._entered:
            for obj, name, replacement, _original in self._methods:
                try:
                    if getattr(obj, name) is not replacement:
                        self.errors.append('METHOD_REPLACED_BY_ANOTHER_OWNER:' + name)
                except Exception:
                    self.errors.append('METHOD_INSPECTION_FAILED:' + name)
            try:
                self._lease.close()
            except Exception as exc:
                self.errors.append('LEASE_CLOSE_ERROR:' + type(exc).__name__)
            for obj, name, _replacement, original in self._methods:
                try:
                    if not _method_equivalent(getattr(obj, name), original):
                        self.errors.append('METHOD_NOT_RESTORED:' + name)
                except Exception:
                    self.errors.append('METHOD_RESTORE_INSPECTION_FAILED:' + name)
        self.report['events'] = self.events
        if self.config.capture_layers:
            self.report['construction_scopes'] = list(self._construction_scopes)
        self.report['warnings'] = list(self.errors) + list(self.numeric_alerts)
        self.report['numeric_alerts'] = list(self.numeric_alerts)
        self.report['forward_calls_seen'] = self._forward_calls
        self.report['denoiser_bindings'] = list(self._denoiser_bindings)
        self.report['denoiser_instances_observed'] = len(self._denoiser_bindings)
        self.report['forward_calls_by_instance'] = [count for _obj, count in self._forward_instance_counts]
        self.report['forward_samples_by_pass'] = [
            {'pass': pass_name, 'calls': count} for _obj, pass_name, count in self._forward_pass_counts
        ]
        self.report['layer_invocations_seen'] = self._layer_invocations
        self.report['status'] = ('INCOMPLETE' if self.errors else
                                 'CAPTURED_WITH_NUMERIC_WARNINGS_NOT_GPU_ACCEPTED'
                                 if self.numeric_alerts else 'CAPTURED_NOT_GPU_ACCEPTED')
        try:
            directory = self.config.directory.resolve()
            directory.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile('w', encoding='utf-8', dir=directory,
                                    prefix='.tcs-proof-', suffix='.json', delete=False) as output:
                temp = Path(output.name)
                json.dump(self.report, output, indent=2, sort_keys=True, ensure_ascii=False)
            name = f'tcs-proof-{self.report["tcs"].lower()}-{self.report["id"][:12]}.json'
            dest = directory / name
            os.replace(temp, dest)
            self.filename = dest
        except Exception as exc:
            self.errors.append('WRITE_ERROR:' + type(exc).__name__)
        self._methods.clear()
        self._denoisers.clear()
        self._forward_instance_counts.clear()
        self._forward_pass_counts.clear()
        return self.filename

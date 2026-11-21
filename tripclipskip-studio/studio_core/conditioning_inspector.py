"""Opt-in, bounded observations of *prepared* A1111 conditioning.

Never runs a text encoder, never mutates conditioning, never saves embeddings.
The expensive sampled mode can synchronize a CUDA device; it is OFF by default.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re

import torch

MODES = ('Выключено', 'Только формы', 'Нормы и сравнение фаз')
MAX_LEAVES = 8
MAX_DEPTH = 10
MAX_SAMPLE = 1024
MAX_FULL_VECTOR = 4096  # tiny 1-D vectors can be compared in full in sampled mode
MAX_PROFILE_CHUNKS = 8
PROFILE_BANDS = 4
PROFILE_ROWS_PER_BAND = 4
PROFILE_CHANNELS = 32
MAX_TENSOR_ELEMENTS = 4_000_000


def _safe_key(key):
    # Do not leak arbitrary dictionary keys, which might encode prompt text.
    if isinstance(key, str) and key in ('crossattn', 'vector', 'pooled', 'cond', 'conds'):
        return key
    return 'element'


def tensor_leaves(obj, *, max_leaves=MAX_LEAVES):
    """Bounded walk over native tensors, including .pooled and scheduled wrappers.

    No tensor copies or CPU transfers occur here.  Opaque objects without
    supported conditioning fields are deliberately not introspected.
    """
    out = []
    visited = set()

    def visit(x, path, depth):
        if len(out) >= max_leaves or depth > MAX_DEPTH:
            return
        if torch.is_tensor(x):
            out.append((path, x))
            pooled = getattr(x, 'pooled', None)
            if pooled is not None:
                visit(pooled, path + '.pooled', depth + 1)
            return
        if x is None or isinstance(x, (int, str, float, bool)):
            return
        if id(x) in visited:
            return
        visited.add(id(x))
        if isinstance(x, dict):
            for i, (key, value) in enumerate(x.items()):
                if i >= 16:
                    break
                visit(value, f'{path}.{_safe_key(key)}{i}', depth + 1)
        elif isinstance(x, (list, tuple)):
            for i, value in enumerate(x[:16]):
                visit(value, f'{path}[{i}]', depth + 1)
        else:
            for field in ('crossattn', 'vector', 'pooled', 'conds', 'cond', 'batch', 'schedules'):
                try:
                    value = getattr(x, field, None)
                except (AttributeError, TypeError, RuntimeError):
                    continue
                if value is not None:
                    visit(value, path + '.' + field, depth + 1)
    visit(obj, 'cond', 0)
    return out


def _spaced_indices(length, size, device):
    """Small, deterministic endpoint-inclusive index set; never touch tensor data."""
    size = min(int(size), int(length))
    if size <= 0:
        return torch.empty(0, device=device, dtype=torch.long)
    if size == 1:
        return torch.zeros(1, device=device, dtype=torch.long)
    return torch.tensor([i * (length - 1) // (size - 1) for i in range(size)],
                        device=device, dtype=torch.long)


def _sample(tensor, count=MAX_SAMPLE):
    """Bounded, reproducible stratified sample of positions and channels.

    For [..., positions, channels] select across BOTH axes, not a prefix
    dominated by BOS. Up to two batches are represented in a 3-D tensor.
    Never convert an entire non-contiguous CLIP-G slice to a flat copy.
    This is descriptive measurement, not a native equivalence proof.
    """
    if (not tensor.numel() or not tensor.is_floating_point()
            or tensor.numel() > MAX_TENSOR_ELEMENTS or count <= 0):
        return None
    x = tensor.detach()
    if x.ndim == 1:
        return x.index_select(0, _spaced_indices(x.shape[0], count, x.device)).float().reshape(-1)
    if x.ndim == 2 or x.ndim == 3:
        batch_count = min(2, x.shape[0]) if x.ndim == 3 else 1
        token_count = min(32, x.shape[-2], max(1, count // batch_count))
        channel_count = min(x.shape[-1], max(1, count // (batch_count * token_count)))
        if x.ndim == 3:
            x = x.index_select(0, _spaced_indices(x.shape[0], batch_count, x.device))
        x = x.index_select(-2, _spaced_indices(x.shape[-2], token_count, x.device))
        x = x.index_select(-1, _spaced_indices(x.shape[-1], channel_count, x.device))
        return x.float().reshape(-1)
    # Unknown conditioning layout: avoid guessing which axes encode tokens.
    return None


def position_profile(reference, observed):
    """Per-chunk, four-band sampled RMS(delta)/RMS(reference).

    This is a positional diagnostic, NOT alignment of tokens to source words.
    For multi-batch 3-D tensors the first batch is visualized only. Device
    reductions are transferred to CPU once for the full bounded profile.
    """
    if (reference.shape != observed.shape or reference.dtype != observed.dtype
            or reference.device != observed.device or not reference.is_floating_point()
            or reference.numel() > MAX_TENSOR_ELEMENTS
            or reference.ndim not in (2, 3)):
        return None
    a, b = reference.detach(), observed.detach()
    if a.ndim == 3:
        a, b = a[0], b[0]
    length, channels = a.shape
    if not length or not channels:
        return None
    chunks = min(MAX_PROFILE_CHUNKS, (length + 76) // 77)
    channel_ix = _spaced_indices(channels, min(PROFILE_CHANNELS, channels), a.device)
    pos_ix = []
    for chunk in range(chunks):
        for band in range(PROFILE_BANDS):
            start = chunk * 77 + (band * 77) // PROFILE_BANDS
            stop = min(length, chunk * 77 + ((band + 1) * 77) // PROFILE_BANDS)
            if start >= stop:
                return None
            # Compute integer indices on CPU; gather from GPU only once.
            span = stop - start
            pos_ix.extend(start + i * (span - 1) // (PROFILE_ROWS_PER_BAND - 1)
                          for i in range(PROFILE_ROWS_PER_BAND))
    index = torch.tensor(pos_ix, device=a.device, dtype=torch.long)
    xa = a.index_select(0, index).index_select(1, channel_ix).float()
    xb = b.index_select(0, index).index_select(1, channel_ix).float()
    xa = xa.reshape(-1, PROFILE_ROWS_PER_BAND * channel_ix.numel())
    xb = xb.reshape_as(xa)
    finite = torch.isfinite(xa) & torch.isfinite(xb)
    counts = finite.sum(-1).clamp_min(1)
    xa = torch.where(finite, xa, torch.zeros_like(xa))
    xb = torch.where(finite, xb, torch.zeros_like(xb))
    base = (xa.square().sum(-1) / counts).sqrt()
    error = ((xa - xb).square().sum(-1) / counts).sqrt()
    ratios = error / base.clamp_min(1e-6)
    ratios = torch.where(finite.any(-1), ratios, torch.full_like(ratios, float('nan')))
    numbers = ratios.detach().cpu().tolist()
    return [numbers[i:i+PROFILE_BANDS] for i in range(0, len(numbers), PROFILE_BANDS)]


def _profile_bar(numbers):
    """Four relative-difference bands: glyphs are diagnostic, not attention."""
    symbols = '▁▂▃▄▅▆▇█'
    cutoffs = (0.001, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0)
    return ''.join('?' if not math.isfinite(n) else ('·' if n == 0 else
                   symbols[sum(n > c for c in cutoffs)]) for n in numbers)

def sample_statistics(tensor):
    sample = _sample(tensor)
    if sample is None:
        return None
    finite = torch.isfinite(sample)
    if not bool(finite.any().item()):
        return {'count': int(sample.numel()), 'valid': 0, 'rms': None, 'mean': None}
    x = sample[finite]
    return {'count': int(sample.numel()), 'valid': int(x.numel()),
            'rms': float(torch.sqrt(torch.mean(x * x)).item()),
            'mean': float(x.mean().item())}


def sampled_difference(a, b):
    """Difference on matched, position/channel-balanced samples.

    For small 1-D vectors (e.g. 2816-element SDXL ADM) compare ALL elements.
    Even a complete per-phase vector comparison is not a native A1111 baseline.
    """
    if tuple(a.shape) != tuple(b.shape) or a.dtype != b.dtype or a.device != b.device:
        return None
    full_vector = a.ndim == 1 and 0 < a.numel() <= MAX_FULL_VECTOR
    if full_vector:
        x, y = a.detach().float().reshape(-1), b.detach().float().reshape(-1)
    else:
        x, y = _sample(a), _sample(b)
    if x is None or y is None or x.shape != y.shape:
        return None
    good = torch.isfinite(x) & torch.isfinite(y)
    if not bool(good.any().item()):
        return None
    x, y = x[good], y[good]
    dx = x - y
    maxdiff = float(dx.abs().max().item())
    denom = x.norm() * y.norm()
    cos = float((x * y).sum().div(denom).item()) if float(denom.item()) > 1e-12 else None
    return {'count': int(x.numel()), 'max_abs_diff': maxdiff, 'cosine': cos,
            'full_vector': bool(full_vector and x.numel() == a.numel())}


def _fmt(num):
    return 'н/д' if num is None or not math.isfinite(num) else f'{num:.5g}'


@dataclass
class TensorRow:
    phase: str
    branch: str
    path: str
    shape: tuple
    dtype: str
    device_type: str
    sample_rms: float | None = None
    sample_mean: float | None = None
    sample_count: int = 0
    extra: str = ''


class ConditioningRecorder:
    def __init__(self, mode):
        if mode not in MODES:
            raise ValueError('Неизвестный режим инспектора conditioning')
        self.mode = mode
        self.rows = []
        self.comparisons = []
        self.position_maps = []
        self.warnings = []
        self.phases = []

    @property
    def sampled(self):
        return self.mode == 'Нормы и сравнение фаз'

    def observe(self, phase, positive, negative, *, reference=None, sdxl=False):
        """Observe already prepared c/uc; best effort, no mutation or caching."""
        if self.mode == 'Выключено':
            return
        if phase in self.phases:
            return  # A1111 may repeat setup_conds for the same base/Hires pass
        self.phases.append(phase)
        for branch, obj, ref_obj in (
                ('Положительное', positive, reference[0] if reference is not None else None),
                ('Негативное', negative, reference[1] if reference is not None else None)):
            leaves = tensor_leaves(obj)
            if not leaves:
                self.warnings.append(f'{phase}: {branch.lower()} conditioning — доступных тензорных листьев не найдено; структура A1111 может быть отложенной')
                continue
            ref_leaves = dict(tensor_leaves(ref_obj)) if ref_obj is not None else {}
            for path, tensor in leaves:
                stat = sample_statistics(tensor) if self.sampled else None
                row = TensorRow(phase, branch, path, tuple(int(n) for n in tensor.shape),
                                str(tensor.dtype).replace('torch.', ''), tensor.device.type,
                                sample_rms=stat['rms'] if stat else None,
                                sample_mean=stat['mean'] if stat else None,
                                sample_count=stat['count'] if stat else 0)
                if len(tensor.shape) == 1 and tensor.shape[-1] == 2816 and sdxl:
                    row.extra = ('SDXL ADM 2816: в типичной схеме pooled G 1280 + '
                                 'размеры/crop 1536; порядок зависит от обёртки')
                self.rows.append(row)
                if self.sampled and path in ref_leaves:
                    diff = sampled_difference(ref_leaves[path], tensor)
                    if diff is not None:
                        self.comparisons.append((phase, branch, path, diff))
                if sdxl and len(tensor.shape) >= 2 and tensor.shape[-1] == 2048:
                    row.extra = 'SDXL токеновые каналы: CLIP-L 768 + OpenCLIP-G 1280 (предполагаемый порядок)'
                    if self.sampled:
                        for name, piece in [('L', tensor[..., :768]), ('G', tensor[..., 768:])]:
                            piece_stats = sample_statistics(piece)
                            if piece_stats:
                                self.rows.append(TensorRow(phase, branch, path + ':' + name,
                                            tuple(int(n) for n in piece.shape), row.dtype, row.device_type,
                                            sample_rms=piece_stats['rms'], sample_mean=piece_stats['mean'],
                                            sample_count=piece_stats['count'], extra='Срез каналов, не отдельный запуск энкодера'))
                            if path in ref_leaves and getattr(ref_leaves[path], 'shape', None) == tensor.shape:
                                original_piece = (ref_leaves[path][..., :768] if name == 'L'
                                                  else ref_leaves[path][..., 768:])
                                delta = sampled_difference(original_piece, piece)
                                if delta is not None:
                                    self.comparisons.append((phase, branch, path + ':' + name, delta))
                                profile = position_profile(original_piece, piece)
                                if profile:
                                    for chunk_idx, bands in enumerate(profile, 1):
                                        self.position_maps.append((phase, branch, name, chunk_idx, bands))

    def markdown(self):
        if not self.phases:
            return 'Измерений пока нет. Включи инспектор и выполни генерацию.'
        header = ('### Conditioning Inspector — фактически подготовленные тензоры\n'
                  f'Режим: **{self.mode}** · Подготовленные состояния: **{", ".join(self.phases)}**\n\n'
                  'Измерения сделаны **внутри текущего запуска A1111** после `setup_conds()`; '
                  'не выполняют повторный CLIP-forward. Текст промпта и значения эмбеддингов не сохраняются.\n\n'
                  '| Фаза | Ветка | Путь | Форма | dtype / устройство | RMS (проба) |\n'
                  '|---|---|---|---|---|---|')
        lines = [header]
        for row in self.rows[:72]:
            lines.append(f'| {row.phase} | {row.branch} | `{row.path}` | '
                         f'`{str(row.shape)}` | `{row.dtype} / {row.device_type}` | '
                         f'{_fmt(row.sample_rms)} |')
        if self.sampled:
            lines.append(f'\n**Измерения v2:** не более {MAX_SAMPLE} элементов, '
                         'распределённых по токеновым позициям и каналам '
                         '(для 3D — максимум два batch-элемента). '
                         f'Для 1D-векторов до {MAX_FULL_VECTOR} чисел разница '
                         'между фазами измеряется по всему вектору и отмечается «полный». '
                         'RMS и cosine не измеряют качество изображения, '
                         'не доказывают native equivalence и могут синхронизировать GPU.')
        if self.comparisons:
            lines += ['\n**Отличия относительно P1 (выборка одинаковых тензоров):**',
                      '| Фаза | Ветка | Путь | max abs diff | cosine |',
                      '|---|---|---|---:|---:|']
            for phase, branch, path, diff in self.comparisons[:24]:
                lines.append(f'| {phase} | {branch} | `{path}` | '
                             f'{_fmt(diff["max_abs_diff"])} | {_fmt(diff["cosine"])}'
                             f'{" (полный)" if diff.get("full_vector") else ""} |')
        if self.position_maps:
            lines.extend(['\n**Карта изменений по участкам чанков (P2–P4 против P1)**',
                          'Четыре символа = последовательные четверти **77 позиций** '
                          'в одном чанке. Это относительный RMS изменения на ограниченной '
                          'сетке позиций/каналов; символ `·` — измеренное отсутствие '
                          'разницы, `?` — нечисловая область. Это не attention '
                          'и не влияние на изображение.',
                          '| Фаза | Ветка | CLIP | Чанк | Четверти 1–4 | Макс. Δ RMS / RMS P1 |',
                          '|---|---|---|---:|---|---:|'])
            for phase, branch, name, chunk, bands in self.position_maps[:48]:
                maximum = max((v for v in bands if math.isfinite(v)), default=float('nan'))
                lines.append(f'| {phase} | {branch} | {name} | {chunk} | '
                             f'`{_profile_bar(bands)}` | {_fmt(maximum)} |')
            if len(self.position_maps) > 48:
                lines.append(f'Показаны первые 48 из {len(self.position_maps)} строк карты.')
            lines.append('Карта для 3D batch-тензоров отображает только первый элемент; '
                         f'покрыто не более {MAX_PROFILE_CHUNKS} чанков. '
                         'В карту входят также BOS/EOS и паддинг; точного '
                         'соответствия исходным тегам здесь нет. При prompt editing '
                         'разные расписания могут использовать разные токены '
                         'на одной позиции.')
        if any(row.extra.startswith('SDXL ADM') for row in self.rows):
            lines.append('\n**SDXL ADM (vector1):** 2816 чисел могут содержать '
                         'pooled G и дополнительные признаки размера/crop. '
                         '«Полный» означает сравнение всех элементов этого вектора '
                         'между подготовленными фазами, **не** сравнение с нативным A1111.')
        if self.warnings:
            lines.append('\n**Ограничения структуры:** ' + '; '.join(self.warnings[:4]))
        return '\n'.join(lines)

    def summary(self):
        return (f'v2; {self.mode}; состояния {len(self.phases)}; '
                f'тензоров/срезов {len(self.rows)}; сравнений {len(self.comparisons)}')

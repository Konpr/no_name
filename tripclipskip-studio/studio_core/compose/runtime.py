"""Opt-in *single-owner* job-local patch lease. Not wired to A1111 generation.

The existing TCS PatchStack has no inter-lease ownership arbitration, so this
module rejects overlapping TCS leases *before* applying an instance patch.
Foreign patches cannot always be identified: if an external owner overwrites
our method, PatchStack deliberately preserves that newer foreign replacement.
"""
from __future__ import annotations
from threading import RLock
from studio_core.patches import PatchStack
from .errors import ComposerOwnerConflict

_LOCK = RLock()
# Strong object references exist ONLY for an active lease; avoid id-reuse bugs.
_OWNED: dict[tuple[int, str], tuple[object, object]] = {}


class JobHookLease:
    """One lease per job, used via `with JobHookLease() as lease`.

    This registry coordinates instances of *this class* only. It does not
    override or modify the existing Script/ClipController's patch ownership.
    """

    def __init__(self):
        self._stack = PatchStack()
        self._active = False
        self._installed: set[tuple[int, str]] = set()

    def __enter__(self):
        with _LOCK:
            if self._active:
                raise ComposerOwnerConflict('JobHookLease cannot be entered twice')
            self._active = True
        return self

    def method(self, obj, name, replacement):
        with _LOCK:
            if not self._active:
                raise ComposerOwnerConflict('Patch lease is not active')
            if not isinstance(name, str) or not name:
                raise ComposerOwnerConflict('Patched method name must be nonempty')
            key = (id(obj), name)
            existing = _OWNED.get(key)
            if existing is not None:
                raise ComposerOwnerConflict(f'Another Composer lease already owns {name}')
            # Register first under the lock, then apply patch atomically.
            # setattr failures must never leave stale ownership metadata.
            _OWNED[key] = (obj, self)
            try:
                patch = self._stack.method(obj, name, replacement)
            except BaseException:
                _OWNED.pop(key, None)
                raise
            self._installed.add(key)
            return patch

    def add(self, removable):
        with _LOCK:
            if not self._active:
                raise ComposerOwnerConflict('Patch lease is not active')
            return self._stack.add(removable)

    def close(self):
        with _LOCK:
            if not self._active:
                return
            try:
                self._stack.close()
            finally:
                for key in self._installed:
                    owned = _OWNED.get(key)
                    if owned is not None and owned[1] is self:
                        _OWNED.pop(key, None)
                self._installed.clear()
                self._active = False

    def __exit__(self, _type, _value, _tb):
        self.close()
        return False


def forward_hook_counts(module):
    """Read-only summary; counts alone cannot establish complete rollback."""
    return tuple((name, len(getattr(child, '_forward_hooks', {})))
                 for name, child in module.named_modules())

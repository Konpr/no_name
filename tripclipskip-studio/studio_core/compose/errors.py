"""Explicit fail-closed error types for Composer experiments."""

class ComposeError(ValueError):
    """Conditioning would not be safe to compose. No silent fallback."""


class UnsupportedModelFamily(ComposeError):
    """The running A1111 model/conditioner was not recognized."""


class PooledLayoutUnknown(ComposeError):
    """A real named pooled-G output cannot be established safely."""


class ComposerOwnerConflict(ComposeError):
    """An incompatible job-local patch lease is already active."""

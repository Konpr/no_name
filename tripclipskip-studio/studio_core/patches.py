"""Safe-ish, reversible instance-level patching for A1111 wrappers.

We never overwrite another extension's newer patch during cleanup.
"""

class MethodPatch:
    def __init__(self, obj, name, replacement):
        self.obj = obj
        self.name = name
        self.original = getattr(obj, name)
        self.replacement = replacement
        setattr(obj, name, replacement)

    def remove(self):
        if self.obj is not None:
            if getattr(self.obj, self.name) is self.replacement:
                setattr(self.obj, self.name, self.original)
            self.obj = None
            self.original = None
            self.replacement = None


class PatchStack:
    def __init__(self):
        self.patches = []

    def method(self, obj, name, replacement):
        p = MethodPatch(obj, name, replacement)
        self.patches.append(p)
        return p

    def add(self, removable):
        self.patches.append(removable)
        return removable

    def close(self):
        errors = []
        for patch in reversed(self.patches):
            try:
                patch.remove()
            except Exception as exc:
                errors.append(str(exc))
        self.patches.clear()
        if errors:
            raise RuntimeError('Error releasing TripClipSkip patches: ' + '; '.join(errors))

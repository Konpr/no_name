"""Conditioning cache invalidation. A1111 caches are [key, value] pairs."""

CACHE_NAMES = ('cached_c', 'cached_uc', 'cached_hr_c', 'cached_hr_uc')


def invalidate(p):
    for name in CACHE_NAMES:
        cache = getattr(p, name, None)
        if isinstance(cache, list) and cache:
            cache[0] = None

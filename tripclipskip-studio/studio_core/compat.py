"""Conservative compatibility policy for A1111 forks/custom modules.

This module uses runtime object capabilities rather than a hard-coded fork name.
REGION modifies attention hooks and conditioning in setup_conds, so one extra
setup pass is not safe until the backend can capture/restore its complete state.
"""


def processing_features(p):
    """Read-only inference from the installed processing implementation."""
    names = set()
    for cls in (p if isinstance(p, type) else type(p)).__mro__:
        func = cls.__dict__.get('setup_conds')
        code = getattr(func, '__code__', None)
        if code is not None:
            names.update(code.co_names)
    return {
        'region_aware': bool({'get_prompt_regions', 'prepare_region_data', 'region_data'} & names),
        'toneg_aware': 'get_prompt_with_toneg' in names,
    }


def active_regions(p):
    """True if the *first native pass* actually produced spatial REGION data.

    Rely on prepared region_data instead of trying to parse user's prompts:
    custom prompt preprocessors may expand macros/wildcards before REGION.
    """
    region_data = getattr(p, 'region_data', None)
    return bool(region_data and any(bool(row) for row in region_data))


def reset_negative_prompts(p, values):
    """Do not let custom setup_conds append TONEG repeatedly."""
    if values is not None:
        p.negative_prompts = list(values)


def preserve_conditioning(p):
    """Snapshot the first-pass outputs, including precomputed Hires conditions."""
    return {key: getattr(p, key, None) for key in ('c', 'uc', 'hr_c', 'hr_uc')}


def restore_conditioning(p, saved):
    for key, value in saved.items():
        setattr(p, key, value)

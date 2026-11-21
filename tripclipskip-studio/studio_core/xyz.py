import os
from typing import Union

from modules import scripts
from modules.processing import (
    StableDiffusionProcessing,
    StableDiffusionProcessingTxt2Img,
    StableDiffusionProcessingImg2Img,
)

# ─────────────────────────────────────────────────────────────
#  BUG FIX: the old code hard-coded 12.0 here, which caused an
#  AssertionError whenever CLIP-G skip values > 12 were passed
#  via the XYZ grid.  max_skip is now forwarded from the script
#  that calls init_xyz so the limit always matches _SKIP_MAX.
# ─────────────────────────────────────────────────────────────

def __set_value(
    p: StableDiffusionProcessing,
    script: type,
    value: Union[float, None] = None,
    typ: Union[str, None] = None,
    max_skip: float = 32.0,
    late_value: Union[float, None] = None,
    switch_at: Union[float, None] = None,
    clipg_value: Union[float, None] = None,
):
    if value is None and typ is None and late_value is None and switch_at is None and clipg_value is None:
        return

    if value is not None:
        assert 1.0 <= value <= max_skip, (
            f"invalid clip skip value: {value} (must be in [1, {max_skip}])"
        )
    if clipg_value is not None:
        assert 1.0 <= clipg_value <= 31.0, "CLIP-G skip must be in 1..31"
    if late_value is not None:
        assert 1.0 <= late_value <= max_skip, (
            f"invalid late clip skip value: {late_value} (must be in [1, {max_skip}])"
        )
    if switch_at is not None:
        assert 0.0 <= switch_at <= 1.0, (
            f"invalid switch fraction: {switch_at} (must be in [0, 1])"
        )
    if typ is not None:
        assert typ in ["lerp", "slerp"], f"invalid interpolation value: {typ}"

    args = list(p.script_args)

    if isinstance(p, StableDiffusionProcessingTxt2Img):
        all_scripts = scripts.scripts_txt2img.scripts
    else:
        all_scripts = scripts.scripts_img2img.scripts

    matching_scripts = [x for x in all_scripts if isinstance(x, script)]
    for entry in matching_scripts:
        idx = entry.args_from
        assert idx is not None
        # Only touch parameters owned by this TCS panel. Older builds with
        # fewer script_args may have another extension immediately after it.
        declared_end = getattr(entry, 'args_to', None)
        limit = min(declared_end, len(args)) if isinstance(declared_end, int) else len(args)
        args[idx + 0] = True  # enable
        if value is not None:
            args[idx + 1] = value
        if typ is not None:
            args[idx + 2] = typ
        if clipg_value is not None:
            args[idx + 3] = True
            args[idx + 4] = clipg_value
        # clipg_override and clipg_skip are at idx+3, idx+4
        # dual_enabled is at idx+5 — automatically enabled when late_value/switch_at set
        if late_value is not None:
            args[idx + 5] = True   # enable dual-phase
            args[idx + 6] = late_value
        if switch_at is not None:
            args[idx + 5] = True   # enable dual-phase
            args[idx + 7] = switch_at
        if late_value is not None or switch_at is not None:
            # Multi-Phase has priority over legacy Dual-Phase.  An XYZ axis
            # explicitly editing the legacy timeline MUST select that mode,
            # otherwise the cell silently ignores its chosen value.
            if idx + 13 < limit:
                args[idx + 13] = 0
        # All legacy XYZ depth/phase axes must win over an opt-in absolute
        # timeline, otherwise plotted values are silently ignored.
        if any(v is not None for v in (value, clipg_value, late_value, switch_at, typ)):
            if isinstance(declared_end, int) and idx + 25 < limit:
                args[idx + 25] = False
            if isinstance(declared_end, int) and idx + 27 < limit:
                args[idx + 27] = 'Выключено'  # Avoid per-cell GPU sync during XYZ sweeps.

    p.script_args = type(p.script_args)(args)


__init = False


def init_xyz(script: type, max_skip: float = 32.0):
    """Register Trip Clip Skip axes in the XYZ grid script.

    Parameters
    ----------
    script:    The Script class from tripclipskip.py.
    max_skip:  Maximum allowed clip skip value (should equal _SKIP_MAX).
               Defaults to 32 to match the SDXL CLIP-G range.
    """
    global __init

    if __init:
        return

    found = False
    for data in scripts.scripts_data:
        name = os.path.basename(data.path)
        if name in ("xy_grid.py", "xyz_grid.py"):
            found = True
            AxisOption = data.module.AxisOption

            v1 = AxisOption(
                "Trip Clip Skip value",
                float,
                lambda p, x, xs, _ms=max_skip: __set_value(p, script, value=x, max_skip=_ms),
            )
            v2 = AxisOption(
                "Trip Clip Skip type",
                str,
                lambda p, x, xs, _ms=max_skip: __set_value(p, script, typ=x, max_skip=_ms),
                choices=lambda: ["lerp", "slerp"],
            )
            v3 = AxisOption(
                "Trip Clip Skip late value",
                float,
                lambda p, x, xs, _ms=max_skip: __set_value(p, script, late_value=x, max_skip=_ms),
            )
            v4 = AxisOption(
                "Trip Clip Skip switch fraction",
                float,
                lambda p, x, xs: __set_value(p, script, switch_at=x),
            )
            data.module.axis_options.append(v1)
            data.module.axis_options.append(v2)
            data.module.axis_options.append(v3)
            data.module.axis_options.append(v4)
            v5 = AxisOption(
                "Trip Clip Skip Studio CLIP-G",
                float,
                lambda p, x, xs: __set_value(p, script, clipg_value=x),
            )
            data.module.axis_options.append(v5)

    # If XYZ was not loaded yet, retry when TripClipSkip Studio creates its UI.
    __init = found

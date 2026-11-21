"""Pure helpers for fixed-seed CLIP-L/G Layer Explorer grids.

No A1111 import here: parsing/grid generation are independently testable.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
from itertools import product
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

MAX_CELLS = 16
MAX_AXIS = 8


def parse_depths(text: str, *, maximum: float, maximum_items: int = MAX_AXIS) -> list[float]:
    """Parse comma-separated depths, optionally start:stop:step (inclusive)."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Enter at least one value, e.g. 1,1.5,2')
    parsed: list[Decimal] = []
    for fragment in text.split(','):
        part = fragment.strip()
        if not part:
            raise ValueError('Empty value in depth list')
        try:
            if ':' not in part:
                parsed.append(Decimal(part))
            else:
                values = [Decimal(x.strip()) for x in part.split(':')]
                if len(values) != 3:
                    raise ValueError('Use start:stop:step, e.g. 1:2:0.5')
                start, stop, step = values
                if not all(x.is_finite() for x in values) or step < Decimal('0.01') or stop < start:
                    raise ValueError('Range must increase using a finite step of at least 0.01')
                n = int((stop - start) // step) + 1
                if n > maximum_items:
                    raise ValueError(f'Too many values in one range (maximum {maximum_items})')
                parsed.extend(start + i * step for i in range(n))
        except (InvalidOperation, OverflowError) as exc:
            raise ValueError(f'Invalid depth: {part}') from exc
        if len(parsed) > maximum_items:
            raise ValueError(f'Maximum {maximum_items} values per axis')
    unique = []
    for value in parsed:
        if not value.is_finite() or not 1 <= value <= Decimal(str(maximum)):
            raise ValueError(f'Depth must be between 1 and {maximum}: {value}')
        if value not in unique:
            unique.append(value)
    if not unique:
        raise ValueError('No depths supplied')
    return [float(v) for v in unique]


def grid_cells(l_values: Sequence[float], g_values: Sequence[float], *, max_cells: int = MAX_CELLS):
    if not l_values or not g_values:
        raise ValueError('Both axes require values')
    if len(l_values) * len(g_values) > max_cells:
        raise ValueError(f'Grid too large: {len(l_values)}×{len(g_values)}; maximum {max_cells} cells')
    # X = L (columns); Y = G (rows)
    return [(float(l), float(g)) for g, l in product(g_values, l_values)]


def make_labeled_grid(images: Sequence[Image.Image], cells: Sequence[tuple[float, float]], columns: int) -> Image.Image:
    if len(images) != len(cells) or not images or columns <= 0:
        raise ValueError('Exactly one image per grid cell is required')
    width = max(im.width for im in images)
    height = max(im.height for im in images)
    header = max(24, min(40, height // 12))
    rows = (len(images) + columns - 1) // columns
    if width * columns * (height + header) * rows > 250_000_000:
        raise ValueError('Resulting grid is too large; reduce width/height or grid size')
    output = Image.new('RGB', (width * columns, (height + header) * rows), '#20242b')
    draw = ImageDraw.Draw(output)
    font = ImageFont.load_default()
    for i, (image, (l, g)) in enumerate(zip(images, cells)):
        x = (i % columns) * width
        y = (i // columns) * (height + header)
        draw.rectangle((x, y, x + width - 1, y + header - 1), fill='#17212e')
        draw.text((x + 10, y + 7), f'CLIP-L {l:g}  |  CLIP-G {g:g}', fill='white', font=font)
        image_rgb = image.convert('RGB')
        output.paste(image_rgb, (x + (width-image.width)//2, y + header + (height-image.height)//2))
    return output


def safe_grid_infotext(source: str, summary: str) -> str:
    """Add an Explorer tag only in an existing A1111 metadata line.

    A grid PNG's positive prompt must remain identical to the real cell's
    positive prompt. Never prepend the Explorer label to `parameters`.
    If we cannot identify A1111's `Steps:` key line, keep the source intact.
    """
    if not isinstance(source, str) or not isinstance(summary, str):
        raise TypeError('A1111 infotext and Explorer summary must be strings')
    lines = source.splitlines(keepends=True)
    for idx in range(len(lines) - 1, -1, -1):
        line = lines[idx].rstrip('\r\n')
        # Steps starts the standard A1111 metadata row, not a prompt line.
        if line.startswith('Steps: ') and ', Sampler:' in line:
            suffix = lines[idx][len(line):]
            label = summary.replace(',', ';').replace('\n', ' ').replace('\r', ' ')
            lines[idx] = line + ', TCS Explorer: ' + label + suffix
            return ''.join(lines)
    return source

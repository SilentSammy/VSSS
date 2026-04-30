"""
path_utils.py — helpers for building and manipulating paths,
                and saving/loading them as SVG files.

Paths are plain Nx2 numpy arrays of (x, y) coordinates in meters.

SVG notes:
    Y is flipped on write so the SVG looks correct in Inkscape/browsers
    (SVG Y increases downward). It is un-flipped on read, so round-trips
    are transparent.
"""

import numpy as np
from xml.etree import ElementTree as ET


def from_fn(fn, t_start=0.0, t_end=2 * np.pi, n=200, loop=True):
    """Sample fn(t) -> (x, y) at n evenly-spaced t values.

    Args:
        fn:      Callable taking a scalar t, returning (x, y).
        t_start: Start of parameter range.
        t_end:   End of parameter range.
        n:       Number of sample points.
        loop:    If True, exclude the endpoint (t_end) to avoid duplicating
                 the start point when the curve is closed.

    Returns:
        Nx2 numpy array.
    """
    ts = np.linspace(t_start, t_end, n, endpoint=not loop)
    return np.array([fn(t) for t in ts], dtype=float)


def circle(r=0.15, n=200, cx=0.0, cy=0.0):
    """Return a circular path.

    Args:
        r:  Radius in meters.
        n:  Number of points.
        cx: Centre x.
        cy: Centre y.

    Returns:
        Nx2 numpy array.
    """
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a)])


def resample(points, spacing=0.005):
    """Redistribute points at uniform arc-length intervals.

    Args:
        points:  Nx2 array-like.
        spacing: Desired distance between consecutive points in meters.

    Returns:
        Mx2 numpy array.
    """
    pts = np.asarray(points, dtype=float)
    diffs = np.diff(pts, axis=0)
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    cumlen = np.concatenate([[0], np.cumsum(seg_lens)])
    total = cumlen[-1]
    n = max(2, int(total / spacing))
    new_s = np.linspace(0, total, n, endpoint=False)
    new_x = np.interp(new_s, cumlen, pts[:, 0])
    new_y = np.interp(new_s, cumlen, pts[:, 1])
    return np.column_stack([new_x, new_y])


def roll_to_closest(points, x, y):
    """Re-order points so they start at the one nearest to (x, y).

    Args:
        points: Nx2 array-like.
        x, y:   Reference position in meters.

    Returns:
        Nx2 numpy array.
    """
    pts = np.asarray(points, dtype=float)
    dists = np.hypot(pts[:, 0] - x, pts[:, 1] - y)
    idx = int(np.argmin(dists))
    return np.roll(pts, -idx, axis=0)


def reverse(points):
    """Reverse traversal direction, keeping the same start point.

    Args:
        points: Nx2 array-like.

    Returns:
        Nx2 numpy array.
    """
    pts = np.asarray(points, dtype=float)
    return np.concatenate([pts[:1], pts[1:][::-1]])


def save_svg(points, filepath, loop=False, stroke='red', stroke_width=0.005):
    """Save a path to an SVG file.

    Args:
        points:       Nx2 array-like of (x, y) in meters.
        filepath:     Output .svg path.
        loop:         If True, close the path back to the first point.
        stroke:       Stroke colour.
        stroke_width: Stroke width in meters.
    """
    points = np.asarray(points, dtype=float)

    xs, ys = points[:, 0], points[:, 1]
    vx_min, vx_max = xs.min(), xs.max()
    vy_min, vy_max = ys.min(), ys.max()
    vw = vx_max - vx_min or 1.0
    vh = vy_max - vy_min or 1.0

    svg = ET.Element('svg', {
        'xmlns': 'http://www.w3.org/2000/svg',
        'viewBox': f'{vx_min:.6f} {-vy_max:.6f} {vw:.6f} {vh:.6f}',
    })

    pts_str = ' '.join(f'{x:.6f},{-y:.6f}' for x, y in points)
    if loop:
        x0, y0 = points[0]
        pts_str += f' {x0:.6f},{-y0:.6f}'

    ET.SubElement(svg, 'polyline', {
        'points': pts_str,
        'fill': 'none',
        'stroke': stroke,
        'stroke-width': str(stroke_width),
    })

    tree = ET.ElementTree(svg)
    ET.indent(tree)
    tree.write(filepath, xml_declaration=True, encoding='unicode')
    print(f"[path_utils] Saved {len(points)} points to '{filepath}'")


def load_svg(filepath):
    """Load a path from an SVG file created by save_svg.

    Returns:
        Nx2 numpy array of (x, y) in meters. Closing loop point is excluded.
    """
    tree = ET.parse(filepath)
    root = tree.getroot()

    ns_prefix = root.tag.split('}')[0] + '}' if '}' in root.tag else ''
    polyline = root.find(f'{ns_prefix}polyline')

    if polyline is None:
        raise ValueError(f"No <polyline> element found in '{filepath}'")

    pts_str = polyline.attrib['points'].strip()
    points = []
    for pair in pts_str.split():
        x_str, y_str = pair.split(',')
        points.append((float(x_str), -float(y_str)))   # un-flip Y

    arr = np.array(points, dtype=float)

    # Drop trailing duplicate point if the path was saved as a loop
    if len(arr) > 1 and np.allclose(arr[0], arr[-1]):
        arr = arr[:-1]

    print(f"[path_utils] Loaded {len(arr)} points from '{filepath}'")
    return arr

if __name__ == '__main__':
    # Parametric heart curve (standard algebraic heart, scaled to ~0.5 m wide by default)
    def heart(t, scale=0.016):
        x = scale * 16 * np.sin(t) ** 3
        y = scale * (13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
        return (x, y)

    heart_path = resample(
        roll_to_closest(
            from_fn(lambda t: heart(t), n=2000),
            0.3, 0
        ),
        spacing=0.003
    )

    inf_path = resample(
        reverse(
            from_fn(lambda t: (
                0.284 * np.cos(t) / (1 + np.sin(t) ** 2),
                0.284 * np.sin(t) * np.cos(t) / (1 + np.sin(t) ** 2)
            ), n=1000)
        ),
        spacing=0.003
    )

    combined_path = np.concatenate([heart_path, inf_path])

    small_heart_path = resample(
        from_fn(lambda t: heart(t, scale=0.3 / 32), n=2000),
        spacing=0.003
    )

    circle_path = resample(circle(r=0.25), spacing=0.005)

    paths = {
        'heart.svg':        (heart_path,       True),
        'lemniscate.svg':   (inf_path,         True),
        'combined.svg':     (combined_path,    False),
        'circle.svg':       (circle_path,      True),
        'small_heart.svg':  (small_heart_path, True),
    }

    for filename, (pts, loop) in paths.items():
        save_svg(pts, filename, loop=loop)
        print(f"  → {filename}: {len(pts)} points")
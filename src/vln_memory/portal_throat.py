"""Verify a short physical narrowing inside a visible passage using synchronized depth and scan."""
import numpy as np


def passage_throat(cue, depth, K, transform, grid, clearance):
    x0, y0, x1, y1 = (
        np.asarray(cue["bbox"], float) / 1000 * [K.width, K.height, K.width, K.height]
    )
    if x1 - x0 < K.width * 0.16 or y1 - y0 < K.height * 0.25:
        return None
    ray = transform[:3, :3] @ np.array([((x0 + x1) / 2 - K.cx) / K.fx, 0, -1])
    forward = ray[[0, 2]]
    forward /= np.linalg.norm(forward)
    across = np.array([-forward[1], forward[0]])
    eye = transform[:3, 3]
    start = eye[[0, 2]]
    step = grid.resolution
    offsets = np.arange(-1.65, 1.65 + step / 2, step)
    stations = np.arange(0.35, 3.1, step)
    pts = start + stations[:, None, None] * forward + offsets[None, :, None] * across
    ij = np.floor((pts - [grid.origin_x, grid.origin_y]) / step).astype(int)
    cols, rows = ij[:, :, 0], ij[:, :, 1]
    valid = (
        (cols >= 0)
        & (cols < grid.log_odds.shape[1])
        & (rows >= 0)
        & (rows < grid.log_odds.shape[0])
    )
    odds = np.zeros(cols.shape)
    odds[valid] = grid.log_odds[rows[valid], cols[valid]]

    # Anchor each side on an occupied return, with continuous free interior.
    def spans(k):
        local = odds[max(0, k - 1) : min(k + 2, len(stations))].max(axis=0)
        left = np.flatnonzero((offsets < 0) & (local >= 0.8))
        right = np.flatnonzero((offsets > 0) & (local >= 0.8))
        if not len(left) or not len(right):
            return None
        a, b = left[-1], right[0]
        if (
            offsets[b] - offsets[a] < 2 * clearance + 0.1
            or offsets[b] - offsets[a] > 2.4
        ):
            return None
        if not np.all(odds[k, a + 2 : b - 1] <= -0.4):
            return None
        return a, b

    spans_by_row = [spans(k) for k in range(len(stations))]
    gap = max(1, round(0.45 / step))
    widen = max(1, round(0.18 / step))

    def depth_support(xz):
        # Project the jamb at camera height, then count supporting measured RGB-D rows.
        point = np.array([xz[0], eye[1], xz[1]])
        cam = transform[:3, :3].T @ (point - eye)
        if cam[2] >= -0.15:
            return False
        u = round(K.cx + K.fx * cam[0] / (-cam[2]))
        if u < 0 or u >= K.width:
            return False
        vs = np.linspace(
            max(0, y0 + 0.25 * (y1 - y0)), min(K.height - 1, y0 + 0.8 * (y1 - y0)), 9
        ).astype(int)
        patch = depth[vs, max(0, u - 6) : min(K.width, u + 7)]
        return (np.abs(patch + cam[2]) < 0.24).any(axis=1).sum() >= 3

    for k, span in enumerate(spans_by_row):
        if span is None or k < gap:
            continue
        a, b = span
        if a < widen or b + widen >= len(offsets):
            continue
        near_free = odds[k - gap] <= -0.4
        # A wall may continue on one side of an asymmetric doorway.
        middle = (a + b) // 2
        if not near_free[middle]:
            continue
        lo = hi = middle
        while lo > 0 and near_free[lo - 1]:
            lo -= 1
        while hi + 1 < len(offsets) and near_free[hi + 1]:
            hi += 1
        # Compare face-to-face widths: lo/hi are free cells, a/b are occupied cells.
        open_width = (hi - lo + 1) * step
        throat_width = (b - a - 1) * step
        if lo > a + 1 or hi < b - 1 or open_width - throat_width < 0.32:
            continue
        # Parallel walls alone do not supply a transition. The near side must widen
        # and the narrow section must persist beyond the first pair of returns.
        stable = [
            s
            for s in spans_by_row[k + 1 : k + 1 + round(0.35 / step)]
            if s is not None and max(abs(s[0] - a), abs(s[1] - b)) * step <= 0.16
        ]
        if len(stable) < 3:
            continue
        ends = pts[k, [a, b]]
        if all(depth_support(p) for p in ends):
            # Place the boundary inside the measured throat, not at its near lip.
            supported = [k]
            for j in range(k + 1, min(len(stations), k + 1 + round(0.9 / step))):
                other = spans_by_row[j]
                if other and max(abs(other[0] - a), abs(other[1] - b)) * step <= 0.16:
                    supported.append(j)
                elif (j - supported[-1]) * step > 0.18:
                    break
            mid = supported[len(supported) // 2]
            return pts[mid, list(spans_by_row[mid])]
    return None

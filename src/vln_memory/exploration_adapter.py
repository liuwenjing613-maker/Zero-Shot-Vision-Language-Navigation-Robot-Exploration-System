"""Baseline complete detector for replay; callers may supply their own snapshot."""
import numpy as np
from scipy import ndimage as ndi


def detect(grid, xy):
    a = grid.grid()
    free = a == 0
    safe = ndi.distance_transform_edt(free) * grid.resolution >= 0.22
    labels, _ = ndi.label(safe)
    base = labels[grid.cell(xy)]
    groups, _ = ndi.label(free & ndi.binary_dilation(a < 0), structure=np.ones((3, 3)))
    candidates = []
    if base:
        dist, indices = ndi.distance_transform_edt(labels != base, return_indices=True)
        for n, box in enumerate(ndi.find_objects(groups), 1):
            cells = np.argwhere(groups[box] == n) + [s.start for s in box]
            if len(cells) * grid.resolution < 0.45:
                continue
            center = cells.mean(axis=0)
            viable = cells[dist[tuple(cells.T)] * grid.resolution <= 0.8]
            if not len(viable):
                continue
            r, c = viable[np.argmin(((viable - center) ** 2).sum(axis=1))]
            vr, vc = indices[:, r, c]

            def world(rc):
                return [
                    grid.origin_x + (rc[1] + 0.5) * grid.resolution,
                    grid.origin_y + (rc[0] + 0.5) * grid.resolution,
                ]

            candidates.append(
                dict(
                    local_candidate_id=str(n),
                    centroid_xy=world(center),
                    safe_viewpoint_xy=world((vr, vc)),
                    cluster_cells=cells.tolist(),
                )
            )
    return dict(
        snapshot_id=str(grid.map_version),
        map_version=grid.map_version,
        complete=True,
        candidates=candidates,
    )


def bounded_cells(grid, cluster):
    # Local target strip on the unknown side of the initial boundary (0.40 m).
    pts = np.array(list(cluster))
    radius = max(1, round(0.4 / grid.resolution))
    lo = np.maximum(0, pts.min(0) - radius)
    hi = np.minimum(grid.log_odds.shape, pts.max(0) + radius + 1)
    mask = np.zeros(hi - lo, bool)
    mask[tuple((pts - lo).T)] = True
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    unknown = np.abs(grid.log_odds[sl]) < 0.4
    selected = mask | ((ndi.distance_transform_edt(~mask) <= radius) & unknown)
    return {tuple(p) for p in (np.argwhere(selected) + lo).tolist()}

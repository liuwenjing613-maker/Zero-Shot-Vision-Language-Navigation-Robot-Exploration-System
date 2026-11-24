import numpy as np
from .geometry_memory import project_pixels


def localize(observations, depth, camera, transform):
    depth = np.asarray(depth)
    pose = np.asarray(transform, dtype=float)
    if depth.shape != (camera.height, camera.width) or depth.dtype.kind not in "fiu":
        raise ValueError("Depth must match calibrated image dimensions")
    if (
        pose.shape != (4, 4)
        or not np.isfinite(pose).all()
        or not np.allclose(pose[3], [0, 0, 0, 1])
        or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(pose[:3, :3]), 1)
    ):
        raise ValueError("Invalid camera pose")
    for obs in observations:
        x, y, right, bottom = (
            np.array(obs.bbox) * np.array([camera.width, camera.height] * 2) / 1000
        )
        cx, cy = (x + right) / 2, (y + bottom) / 2
        ray = pose[:3, :3] @ np.array(
            [
                (cx - 0.5 - camera.cx) / camera.fx,
                (camera.cy - cy + 0.5) / camera.fy,
                -1.0,
            ]
        )
        obs.camera_xyz = tuple(pose[:3, 3])
        obs.forward = tuple(-pose[:3, 2])
        obs.bearing_world = tuple(ray / np.linalg.norm(ray))
        obs.position_source = "BEARING_ONLY"
        dx, dy = (right - x) / 4, (bottom - y) / 4
        v, u = np.mgrid[
            int(cy - dy) : int(np.ceil(cy + dy)), int(cx - dx) : int(np.ceil(cx + dx))
        ]
        depths = depth[v, u]
        valid = np.isfinite(depths) & (depths > 0.05) & (depths < 8.0)
        obs.depth_valid_ratio = float(valid.mean())
        if valid.sum() < 6 or obs.depth_valid_ratio < 0.35:
            continue
        u, v, z = u[valid], v[valid], depths[valid]
        order = np.argsort(z)
        ordered = z[order]
        breaks = (
            np.flatnonzero(np.diff(ordered) > np.maximum(0.12, 0.05 * ordered[:-1])) + 1
        )
        cluster = max(np.split(order, breaks), key=len)
        purity = len(cluster) / len(z)
        if purity < 0.55 or len(cluster) < 6:
            continue
        points = project_pixels(u[cluster], v[cluster], z[cluster], camera, pose)[
            :, [0, 2]
        ]
        pos = np.median(points, axis=0)
        spread = np.median(np.linalg.norm(points - pos, axis=1))
        sigma = float(np.hypot(0.08, spread))
        quality = obs.depth_valid_ratio * purity / (1 + sigma)
        if sigma > 0.8 or quality < 0.3:
            continue
        obs.map_position_xz = tuple(pos)
        obs.position_sigma_m = sigma
        obs.geometry_quality = quality
        obs.position_source = "MAPPED"

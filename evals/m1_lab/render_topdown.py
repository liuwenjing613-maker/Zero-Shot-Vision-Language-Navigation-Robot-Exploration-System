"""Render a textured orthographic reference aligned to the MP3D scan grid."""
import argparse
import json
import math
from pathlib import Path

import habitat_sim as hs
import magnum as mn
import numpy as np
from PIL import Image

SCENE = "2azQ1b91cZZ"
PIXELS_PER_CELL = 8


def camera(name, rows, cols, kind):
    spec = hs.CameraSensorSpec()
    spec.uuid = name
    spec.sensor_type = hs.SensorType.COLOR
    spec.sensor_subtype = kind
    spec.resolution = mn.Vector2i(rows, cols)
    spec.position = mn.Vector3(0, 0, 0)
    spec.orientation = mn.Vector3(-math.pi / 2, 0, 0)
    spec.near, spec.far = 0.02, 2.50
    spec.clear_color = mn.Color4(0.80, 0.83, 0.83, 1.0)
    return spec


def render(dataset, snapshot, output, gpu):
    frame = json.loads(snapshot.read_text())
    rows, cols = frame["shape"]
    origin = np.array(frame["origin"])
    resolution = frame["resolution"]
    height, width = rows * PIXELS_PER_CELL, cols * PIXELS_PER_CELL
    root = Path(__file__).resolve().parents[2]
    recorded = json.loads((root / "results/verify-e02-new-view/e02-view-comparison.json").read_text())
    if recorded["scene"] != SCENE or not np.allclose(recorded["old_view"]["position_m"][::2], frame["scan"]["trajectory"][0]):
        raise ValueError("Unexpected scene or route")
    floor = recorded["old_view"]["position_m"][1]
    eye_height = floor + 2.05
    cfg = hs.SimulatorConfiguration()
    cfg.scene_dataset_config_file = str(dataset / "mp3d.scene_dataset_config.json")
    cfg.scene_id = str(dataset / SCENE / (SCENE + ".glb"))
    cfg.gpu_device_id = gpu
    cfg.enable_physics = False
    ortho = camera("overview", height, width, hs.SensorSubType.ORTHOGRAPHIC)
    ortho.ortho_scale = 1.0
    probe = camera("probe", 129, 129, hs.SensorSubType.PINHOLE)
    probe.hfov = mn.Deg(50)
    config = hs.agent.AgentConfiguration()
    config.sensor_specifications = [ortho, probe]
    with hs.Simulator(hs.Configuration(cfg, [config])) as sim:
        agent = sim.initialize_agent(0)
        def set_pose(xz):
            state = hs.AgentState()
            state.position = [xz[0], eye_height, xz[1]]
            agent.set_state(state)
        set_pose(origin + resolution * np.array([cols, rows]) / 2)
        sensor = sim._sensors["overview"]._sensor_object
        cam = sensor.render_camera
        target = 2.0 / (cols * resolution)
        sensor.zoom(target / np.asarray(cam.projection_matrix)[0, 0])
        rgb = sim.get_sensor_observations()["overview"][:, :, :3].copy()
        proj = np.asarray(cam.projection_matrix) @ np.asarray(cam.camera_matrix)
        # Include map corners and route start for rotation, sign, scale and offset checks.
        pixels = np.array([[0, 0], [width, 0], [0, height], [width, height],
                           (np.array(frame["xy"])-origin) * PIXELS_PER_CELL / resolution])
        xz = origin + pixels * resolution / PIXELS_PER_CELL
        world = np.column_stack((xz[:,0], np.full(len(xz), floor), xz[:,1], np.ones(len(xz))))
        clip = world @ proj.T
        projected = (clip[:,:2] / clip[:,3,None] * [1,-1] + 1) * [width/2, height/2]
        max_error = float(np.max(np.linalg.norm(projected - pixels, axis=1)))
        if max_error > 0.01:
            raise ValueError(f"Orthographic registration error: {max_error} px")

        # Independently re-render the same world points with a downward pinhole camera.
        diffs = []
        for row in np.linspace(0.16*height, 0.84*height, 7).astype(int):
            for col in np.linspace(0.16*width, 0.84*width, 7).astype(int):
                color = rgb[row,col].astype(float)
                if np.max(np.abs(color - [204,212,212])) < 3:
                    continue
                set_pose(origin + (np.array([col,row])+0.5)*resolution/PIXELS_PER_CELL)
                reference = sim.get_sensor_observations()["probe"][64,64,:3].astype(float)
                diffs.append(float(np.abs(reference-color).mean()))
        median = float(np.median(diffs)) if diffs else None
        if len(diffs)<12 or median>8:
            raise ValueError(f"Color registration check failed: {len(diffs)} samples, median {median}")

    metadata = dict(scene=SCENE, origin=origin.tolist(), shape=[rows,cols], resolution=resolution,
                    width=width, height=height, pixels_per_cell=PIXELS_PER_CELL,
                    reference_floor_world_y=floor, camera_world_y=eye_height, near_m=0.02, far_m=2.50,
                    axis_right="world +X", axis_down="world +Z", max_projection_error_px=max_error,
                    pinhole_probes=len(diffs), median_probe_rgb_mae=median)
    output.mkdir(parents=True,exist_ok=True)
    Image.fromarray(rgb).save(output/"topdown.jpg", quality=93, subsampling=0)
    (output/"topdown.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(json.dumps(metadata),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("dataset",type=Path)
    parser.add_argument("snapshot",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--gpu",type=int,default=3)
    args=parser.parse_args()
    render(args.dataset,args.snapshot,args.output,args.gpu)

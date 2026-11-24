"""Adapter for raw Habitat-Sim observations and matching agent.get_state()."""
import numpy as np


def update_from_habitat(memory, observations, agent_state, intrinsics, timestamp,
                        *, sensor_id="depth_sensor", **integration_options):
    import quaternion  # Provided by Habitat-Sim; the core package only needs numpy.

    sensor = agent_state.sensor_states[sensor_id]
    transform = np.eye(4)
    transform[:3, :3] = quaternion.as_rotation_matrix(sensor.rotation)
    transform[:3, 3] = sensor.position
    memory.update_depth(observations[sensor_id], intrinsics, transform,
                        agent_state.position, timestamp, **integration_options)

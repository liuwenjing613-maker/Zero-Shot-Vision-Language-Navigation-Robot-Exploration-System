# V4 module boundary

This directory starts the V4 modular architecture. It currently contains the
frozen shared schemas and the completed Decision module. The module has no GPU,
Habitat, NumPy, Qwen, or network dependency, so its state-machine policy can be
tested independently before systems integration.

The existing `lwj` branch remains the owner of the Qwen perception client. Its
output is adapted at the `PerceptionResult` boundary; RGB-D geometry and World
Memory remain separate modules by design.

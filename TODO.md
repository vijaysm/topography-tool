# TODO

## Critical correctness and robustness

- [ ] Fix division-by-zero in progress logging for small meshes (`<10` elements).
- [ ] Ensure MPI is always finalized on all return paths.
- [ ] Normalize `--remap-method` parsing to accept `da|nn` case-insensitively and fail fast on unknown values.
- [ ] Honor source coordinate overrides (`--lon-var`, `--lat-var`) instead of hardcoding source coordinates.
- [ ] Add strict read/write validation so requested fields must exist and be fully read/written; fail with explicit errors.
- [ ] Fix/retire incorrect unit-sphere check in unused KD helper path if retained.
- [ ] Improve distributed read behavior so each MPI rank reads its local slice rather than all data.
- [ ] Harden format detection for non-2D coordinate variable layouts.
- [ ] Avoid silent success when scalar variables are skipped (e.g., unsupported dimensions).
- [ ] Resolve manual-close/RAII double-close risk in NetCDF I/O.

## Dead code and API cleanup

- [ ] Remove unused `MOABCentroidCloud` + `radius_search_kdtree` helper path (or wire and test it).
- [ ] Remove unimplemented declarations in `ParallelPointCloudReader.hpp` (or implement all of them).
- [ ] Remove stale commented/deprecated code blocks in `mbda.cpp` and `MeshIO.cpp`.
- [ ] Remove unused members/helpers (e.g., `m_unique_points`, unused local utilities).

## Build and repository hygiene

- [ ] Expand `.gitignore` for generated artifacts (`*.nc`, `*.h5m`, logs, slurm outputs, binaries, temp files).
- [ ] Remove unused makefile variables/targets and align docs/help text with actual behavior.

## Regression coverage

- [ ] Add regression test for tiny meshes (`n<10`) to prevent modulo/division crashes.
- [ ] Add regression test for `--remap-method nn` parsing and unknown-method rejection.
- [ ] Add regression test for source coordinate override behavior.
- [ ] Add regression test for 1D/2D coordinate format detection edge cases.
- [ ] Add regression test for multi-rank read decomposition.

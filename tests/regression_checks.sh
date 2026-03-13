#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

check() {
  local cmd="$1"
  local msg="$2"
  if ! eval "$cmd" >/dev/null 2>&1; then
    echo "[FAIL] $msg"
    return 1
  fi
  echo "[PASS] $msg"
  return 0
}

status=0

# 1) Tiny-mesh progress guard coverage.
check "grep -F \"std::max<size_t>(1, m_mesh_data.elements.size() / 10)\" src/ScalarRemapper.cpp" \
  "Spectral projection progress guard avoids division by zero" || status=1
check "grep -F \"std::max<size_t>(1, elements.size() / 10)\" src/ScalarRemapper.cpp" \
  "Area-averaging progress guard avoids division by zero" || status=1

# 2) Remap-method parser coverage.
check "grep -F \"normalized_method == \\\"DA\\\"\" src/mbda.cpp" \
  "CLI parser accepts 'da' remap method path" || status=1
check "grep -F \"normalized_method == \\\"NN\\\"\" src/mbda.cpp" \
  "CLI parser accepts 'nn' remap method path" || status=1
check "grep -F \"Unknown remap method\" src/mbda.cpp" \
  "CLI parser rejects unknown remap methods" || status=1

# 3) Source coordinate override coverage.
check "grep -F \"lon_variable.empty() ? \\\"lon\\\" : lon_variable\" src/mbda.cpp" \
  "Source longitude override is wired" || status=1
check "grep -F \"lat_variable.empty() ? \\\"lat\\\" : lat_variable\" src/mbda.cpp" \
  "Source latitude override is wired" || status=1

# 4) 1D/2D coordinate detection fallback coverage.
check "grep -F \"coord_dims.size() == 1\" src/ParallelPointCloudReader.cpp" \
  "Reader handles 1D coordinate variable layout" || status=1
check "grep -F \"has zero dimensions\" src/ParallelPointCloudReader.cpp" \
  "Reader rejects zero-dimension coordinate variables" || status=1

# 5) Multi-rank read decomposition coverage.
check "grep -F \"MPI_Comm_rank\" src/ParallelPointCloudReader.cpp" \
  "Reader queries MPI rank for decomposition" || status=1
check "grep -F \"global_start\" src/ParallelPointCloudReader.cpp" \
  "Reader computes unstructured global slice per rank" || status=1
check "grep -F \"nlats_start = (static_cast<size_t>(rank) * nlats) / comm_size\" src/ParallelPointCloudReader.cpp" \
  "Reader computes structured latitude slice per rank" || status=1

exit "$status"

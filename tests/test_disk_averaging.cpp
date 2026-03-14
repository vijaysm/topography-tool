/**
 * @file test_disk_averaging.cpp
 * @brief Tests for disk-area averaging using RegularGridLocator
 *
 * Validates the core remapping algorithm: for each target point, find all
 * source grid points within a search radius and average their scalar values.
 * This is the heart of PCDiskAveragedProjectionRemapper.
 *
 * Test categories:
 *   1. Constant field preservation: averaging a uniform field returns that value
 *   2. Linear field: averaging a linear field over a symmetric disk returns
 *      the center value (mean of a symmetric distribution = center)
 *   3. Monotonicity / min-max: averaged values lie within [min,max] of inputs
 *   4. Smoothing: averaging reduces the range of the field
 *   5. Self-consistency: result doesn't depend on which pole convention is used
 *   6. Coverage: every target point gets a value (fallback to nearest neighbor)
 */

#define ELPP_DISABLE_LOGS

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "easylogging.hpp"
INITIALIZE_EASYLOGGINGPP

#include "RegularGridLocator.hpp"

using namespace moab;

// ---------------------------------------------------------------------------
// Tiny test harness
// ---------------------------------------------------------------------------

static int g_tests = 0, g_pass = 0, g_fail = 0;

#undef CHECK
#define CHECK(cond, msg)                                                       \
  do {                                                                         \
    ++g_tests;                                                                 \
    if (cond) {                                                                \
      ++g_pass;                                                                \
    } else {                                                                   \
      ++g_fail;                                                                \
      std::cerr << "[FAIL] " << __FILE__ << ":" << __LINE__ << " " << msg      \
                << "\n";                                                       \
    }                                                                          \
  } while (0)

#define NEAR(a, b, tol) (std::abs((a) - (b)) < (tol))

// ---------------------------------------------------------------------------
// Grid + scalar field builders
// ---------------------------------------------------------------------------

struct Grid {
  std::vector<double> lats, lons;
  size_t nlat, nlon;

  void build(double dlat, double dlon, double lon_start = 0.0) {
    lats.clear();
    lons.clear();
    for (double lat = -90.0; lat <= 90.0 + 1e-9; lat += dlat)
      lats.push_back(lat);
    for (double lon = lon_start; lon < lon_start + 360.0 - 1e-9; lon += dlon)
      lons.push_back(lon);
    nlat = lats.size();
    nlon = lons.size();
  }

  size_t size() const { return nlat * nlon; }

  double lon_at(size_t idx) const { return lons[idx % nlon]; }
  double lat_at(size_t idx) const { return lats[idx / nlon]; }
};

/// Disk-area averaging: for each grid point, find all neighbors within
/// `radius_deg` and return their simple mean.  This mirrors the production
/// PCDiskAveragedProjectionRemapper algorithm.
static std::vector<double>
disk_average(const Grid &grid, RegularGridLocator &loc,
             const std::vector<double> &field, double radius_deg) {
  std::vector<double> result(grid.size(), 0.0);
  std::vector<nanoflann::ResultItem<size_t, CoordinateType>> matches;

  for (size_t i = 0; i < grid.size(); ++i) {
    PointType3D qpt = {grid.lon_at(i), grid.lat_at(i), 0.0};
    matches.clear();
    loc.radiusSearch(qpt, radius_deg, matches, false);

    if (matches.empty()) {
      // Fallback: nearest neighbor (mimics production code)
      size_t nn_idx;
      CoordinateType nn_dist;
      loc.knnSearch(qpt, 1, &nn_idx, &nn_dist);
      result[i] = field[nn_idx];
    } else {
      double sum = 0.0;
      for (auto &m : matches)
        sum += field[m.first];
      result[i] = sum / static_cast<double>(matches.size());
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// 1.  Constant field preservation
// ---------------------------------------------------------------------------

static void test_constant_field() {
  std::cout << "  Constant field preservation ...\n";

  Grid g;
  g.build(5.0, 5.0);
  RegularGridLocator loc(g.lats, g.lons, HAVERSINE);

  // Constant field = 42.0
  std::vector<double> field(g.size(), 42.0);

  // Average with various radii
  for (double r : {1.0, 3.0, 5.0, 10.0, 20.0}) {
    auto result = disk_average(g, loc, field, r);
    double max_err = 0.0;
    for (size_t i = 0; i < result.size(); ++i)
      max_err = std::max(max_err, std::abs(result[i] - 42.0));
    CHECK(max_err < 1e-12,
          "constant field r=" + std::to_string(r) +
          " max_err=" + std::to_string(max_err));
  }
}

// ---------------------------------------------------------------------------
// 2.  Linear field: latitude-only
// ---------------------------------------------------------------------------

static void test_linear_field_lat() {
  std::cout << "  Linear latitude field ...\n";

  Grid g;
  g.build(2.0, 2.0);
  RegularGridLocator loc(g.lats, g.lons, HAVERSINE);

  // Field = latitude (linear in lat)
  std::vector<double> field(g.size());
  for (size_t i = 0; i < g.size(); ++i)
    field[i] = g.lat_at(i);

  // For a small radius, the averaged value at equatorial points should
  // be close to the original (mean of a locally linear field ≈ center value)
  auto result = disk_average(g, loc, field, 3.0);

  // Check mid-latitude points (avoid poles where the disk becomes
  // asymmetric in lat/lon space)
  int large_err = 0;
  for (size_t i = 0; i < g.size(); ++i) {
    double lat = g.lat_at(i);
    if (std::abs(lat) > 80.0)
      continue; // skip poles
    double err = std::abs(result[i] - field[i]);
    // For a 3° radius on a 2° grid, mean of a linear field deviates
    // only from asymmetric boundary effects
    if (err > 0.5)
      ++large_err;
  }
  CHECK(large_err == 0,
        "linear lat field: large errors at " + std::to_string(large_err) +
        " points");
}

// ---------------------------------------------------------------------------
// 3.  Min-max bounds (monotonicity of averaging)
// ---------------------------------------------------------------------------

static void test_min_max_bounds() {
  std::cout << "  Min-max bounds ...\n";

  Grid g;
  g.build(5.0, 5.0);
  RegularGridLocator loc(g.lats, g.lons, HAVERSINE);

  // Random field
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> val_d(-100.0, 100.0);
  std::vector<double> field(g.size());
  for (auto &v : field)
    v = val_d(rng);

  double field_min = *std::min_element(field.begin(), field.end());
  double field_max = *std::max_element(field.begin(), field.end());

  for (double r : {2.0, 5.0, 10.0}) {
    auto result = disk_average(g, loc, field, r);
    double res_min = *std::min_element(result.begin(), result.end());
    double res_max = *std::max_element(result.begin(), result.end());

    CHECK(res_min >= field_min - 1e-10,
          "min bound r=" + std::to_string(r) + " res_min=" + std::to_string(res_min));
    CHECK(res_max <= field_max + 1e-10,
          "max bound r=" + std::to_string(r) + " res_max=" + std::to_string(res_max));
  }
}

// ---------------------------------------------------------------------------
// 4.  Smoothing reduces range
// ---------------------------------------------------------------------------

static void test_smoothing_reduces_range() {
  std::cout << "  Smoothing reduces range ...\n";

  Grid g;
  g.build(5.0, 5.0);
  RegularGridLocator loc(g.lats, g.lons, HAVERSINE);

  // Random noisy field
  std::mt19937 rng(77);
  std::uniform_real_distribution<double> val_d(0.0, 100.0);
  std::vector<double> field(g.size());
  for (auto &v : field)
    v = val_d(rng);

  double orig_range =
      *std::max_element(field.begin(), field.end()) -
      *std::min_element(field.begin(), field.end());

  // A large enough averaging radius should reduce the range
  auto result = disk_average(g, loc, field, 15.0);
  double new_range =
      *std::max_element(result.begin(), result.end()) -
      *std::min_element(result.begin(), result.end());

  CHECK(new_range < orig_range,
        "range reduced: " + std::to_string(orig_range) + " -> " +
        std::to_string(new_range));
}

// ---------------------------------------------------------------------------
// 5.  Grid convention independence
// ---------------------------------------------------------------------------

static void test_grid_convention_independence() {
  std::cout << "  Grid convention independence ...\n";

  // Build two grids: [0,360) and [-180,180)
  Grid g1, g2;
  g1.build(5.0, 5.0, 0.0);
  g2.build(5.0, 5.0, -180.0);

  RegularGridLocator loc1(g1.lats, g1.lons, HAVERSINE);
  RegularGridLocator loc2(g2.lats, g2.lons, HAVERSINE);

  // Field = cos(lat) * sin(lon)  (a smooth spherical harmonic-like field)
  std::vector<double> field1(g1.size()), field2(g2.size());
  for (size_t i = 0; i < g1.size(); ++i) {
    double lat = g1.lat_at(i) * DEG_TO_RAD;
    double lon = g1.lon_at(i) * DEG_TO_RAD;
    field1[i] = std::cos(lat) * std::sin(lon);
  }
  for (size_t i = 0; i < g2.size(); ++i) {
    double lat = g2.lat_at(i) * DEG_TO_RAD;
    double lon = g2.lon_at(i) * DEG_TO_RAD;
    field2[i] = std::cos(lat) * std::sin(lon);
  }

  auto result1 = disk_average(g1, loc1, field1, 8.0);
  auto result2 = disk_average(g2, loc2, field2, 8.0);

  // Compare results at corresponding grid points.  The grids have the same
  // lat values but shifted lon values.  For each (lat, lon) in g1, find the
  // matching point in g2.
  int mismatches = 0;
  for (size_t ilat = 0; ilat < g1.nlat; ++ilat) {
    for (size_t ilon = 0; ilon < g1.nlon; ++ilon) {
      double lon1 = g1.lons[ilon];
      // Find matching lon in g2: normalize lon1 to g2's range
      double lon2 = lon1;
      if (lon2 >= 180.0) lon2 -= 360.0;
      // Find index in g2
      size_t ilon2 = 0;
      double best = 999;
      for (size_t j = 0; j < g2.nlon; ++j) {
        double d = std::abs(g2.lons[j] - lon2);
        if (d < best) { best = d; ilon2 = j; }
      }
      if (best > 0.01) continue; // no matching point

      size_t idx1 = ilat * g1.nlon + ilon;
      size_t idx2 = ilat * g2.nlon + ilon2;
      if (std::abs(result1[idx1] - result2[idx2]) > 1e-10)
        ++mismatches;
    }
  }
  CHECK(mismatches == 0,
        "convention mismatches=" + std::to_string(mismatches));
}

// ---------------------------------------------------------------------------
// 6.  Coverage: every point gets a value
// ---------------------------------------------------------------------------

static void test_full_coverage() {
  std::cout << "  Full coverage (no NaN/inf) ...\n";

  Grid g;
  g.build(5.0, 5.0);
  RegularGridLocator loc(g.lats, g.lons, HAVERSINE);

  std::vector<double> field(g.size());
  std::mt19937 rng(22);
  std::uniform_real_distribution<double> val_d(1.0, 100.0);
  for (auto &v : field)
    v = val_d(rng);

  // Even with a tiny radius where some points might find no neighbors,
  // the fallback to kNN should ensure every point gets a value.
  for (double r : {0.01, 0.5, 5.0}) {
    auto result = disk_average(g, loc, field, r);
    int bad = 0;
    for (double v : result) {
      if (std::isnan(v) || std::isinf(v))
        ++bad;
    }
    CHECK(bad == 0,
          "NaN/inf count r=" + std::to_string(r) + " bad=" + std::to_string(bad));
  }
}

// ---------------------------------------------------------------------------
// 7.  Pole-region averaging
// ---------------------------------------------------------------------------

static void test_pole_averaging() {
  std::cout << "  Pole-region averaging ...\n";

  Grid g;
  g.build(2.0, 2.0);
  RegularGridLocator loc(g.lats, g.lons, HAVERSINE);

  // Constant field = 7.0
  std::vector<double> field(g.size(), 7.0);

  // Average at the north pole with a large radius
  PointType3D qpt = {0.0, 90.0, 0.0};
  std::vector<nanoflann::ResultItem<size_t, CoordinateType>> matches;
  loc.radiusSearch(qpt, 5.0, matches, false);

  // Should find points at lat=90 (all lons) and lat=88 (all lons)
  CHECK(!matches.empty(), "pole query returned results");

  // Average should still be 7.0 (constant field)
  double sum = 0;
  for (auto &m : matches)
    sum += field[m.first];
  double avg = sum / matches.size();
  CHECK(NEAR(avg, 7.0, 1e-12), "pole avg of constant field = 7.0");

  // For a latitude-dependent field, pole averaging should include
  // points from multiple latitudes
  std::vector<double> lat_field(g.size());
  for (size_t i = 0; i < g.size(); ++i)
    lat_field[i] = g.lat_at(i);

  sum = 0;
  for (auto &m : matches)
    sum += lat_field[m.first];
  avg = sum / matches.size();
  // Average should be between 85 and 90 (we average lat=88 and lat=90 points)
  CHECK(avg >= 85.0 && avg <= 90.0,
        "pole avg of lat field in [85,90], got " + std::to_string(avg));
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
  std::cout << "=== test_disk_averaging ===\n";

  test_constant_field();
  test_linear_field_lat();
  test_min_max_bounds();
  test_smoothing_reduces_range();
  test_grid_convention_independence();
  test_full_coverage();
  test_pole_averaging();

  std::cout << "\n--- Summary ---\n"
            << "Total : " << g_tests << "\n"
            << "Passed: " << g_pass << "\n"
            << "Failed: " << g_fail << "\n";

  if (g_fail == 0)
    std::cout << "All tests passed.\n";
  else
    std::cout << "TESTS FAILED\n";

  return (g_fail == 0) ? 0 : 1;
}

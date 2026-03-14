/**
 * @file test_kdtree_locator.cpp
 * @brief Validation test for nanoflann KD-tree radius search on unit sphere
 *
 * Validates that nanoflann's 3D KD-tree correctly finds all points within
 * a given angular radius on the unit sphere.  The reference is brute-force
 * 3D Euclidean distance (cannot fail).
 *
 * This exercises the same KD-tree code path used by ScalarRemapper for
 * unstructured point clouds.
 *
 * Test coverage:
 *   - Random point clouds on the unit sphere
 *   - Queries at mid-latitudes, near poles, at poles
 *   - Variable radii (small to large)
 *   - Consistency: KD-tree result == brute-force result (exact match)
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include "MBDAUtilities.hpp"
#include "nanoflann.hpp"

// ---------------------------------------------------------------------------
// nanoflann adaptor for a flat vector of 3D points
// ---------------------------------------------------------------------------

struct PointCloud3D {
  std::vector<std::array<double, 3>> pts;
  size_t kdtree_get_point_count() const { return pts.size(); }
  double kdtree_get_pt(size_t idx, int dim) const { return pts[idx][dim]; }
  template <class BBOX> bool kdtree_get_bbox(BBOX &) const { return false; }
};

using KDTree3D = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, PointCloud3D>, PointCloud3D, 3,
    size_t>;

// ---------------------------------------------------------------------------
// Brute-force reference
// ---------------------------------------------------------------------------

/// Find all points within squared Euclidean distance <= r2 of query.
static std::vector<size_t> brute_force_l2(const PointCloud3D &cloud,
                                           const double *q, double r2) {
  std::vector<size_t> out;
  for (size_t i = 0; i < cloud.pts.size(); ++i) {
    double dx = cloud.pts[i][0] - q[0];
    double dy = cloud.pts[i][1] - q[1];
    double dz = cloud.pts[i][2] - q[2];
    double d2 = dx * dx + dy * dy + dz * dz;
    if (d2 < r2) // nanoflann uses strict <
      out.push_back(i);
  }
  std::sort(out.begin(), out.end());
  return out;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::array<double, 3> lonlat_to_xyz(double lon_deg, double lat_deg) {
  double lon = lon_deg * DEG_TO_RAD;
  double lat = lat_deg * DEG_TO_RAD;
  double c = std::cos(lat);
  return {c * std::cos(lon), c * std::sin(lon), std::sin(lat)};
}

/// Convert angular radius (degrees) to squared chord distance on unit sphere.
static double deg_to_chord2(double r_deg) {
  double half = r_deg * DEG_TO_RAD / 2.0;
  double s = std::sin(half);
  return 4.0 * s * s;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
  // ---------------------------------------------------------------
  // 1.  Build a random point cloud on the unit sphere
  // ---------------------------------------------------------------
  constexpr size_t N = 50000;
  PointCloud3D cloud;
  cloud.pts.resize(N);

  std::mt19937 rng(123);
  std::uniform_real_distribution<double> uni(-1.0, 1.0);

  for (auto &p : cloud.pts) {
    double x, y, z, r;
    do {
      x = uni(rng);
      y = uni(rng);
      z = uni(rng);
      r = std::sqrt(x * x + y * y + z * z);
    } while (r < 1e-8); // reject origin
    p = {x / r, y / r, z / r};
  }

  // ---------------------------------------------------------------
  // 2.  Build KD-tree
  // ---------------------------------------------------------------
  KDTree3D tree(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(16));
  tree.buildIndex();

  std::cout << "KD-tree built with " << N << " unit-sphere points\n";

  // ---------------------------------------------------------------
  // 3.  Generate queries
  // ---------------------------------------------------------------
  struct Query {
    std::string label;
    double qx, qy, qz;
    double r_deg;
  };
  std::vector<Query> queries;

  // 3a. Random queries using existing points, variable radii
  {
    std::uniform_int_distribution<size_t> idx_d(0, N - 1);
    std::uniform_real_distribution<double> rad_d(0.5, 10.0);
    for (int i = 0; i < 500; ++i) {
      size_t idx = idx_d(rng);
      auto &p = cloud.pts[idx];
      queries.push_back({"random #" + std::to_string(i), p[0], p[1], p[2],
                         rad_d(rng)});
    }
  }

  // 3b. Pole queries
  for (double r : {0.5, 2.0, 5.0, 10.0, 20.0}) {
    queries.push_back({"north-pole r=" + std::to_string(r), 0, 0, 1, r});
    queries.push_back({"south-pole r=" + std::to_string(r), 0, 0, -1, r});
  }

  // 3c. Equator / dateline queries
  for (double lon : {0.0, 90.0, 180.0, 270.0}) {
    auto xyz = lonlat_to_xyz(lon, 0.0);
    for (double r : {1.0, 5.0, 15.0})
      queries.push_back({"equator lon=" + std::to_string((int)lon), xyz[0],
                         xyz[1], xyz[2], r});
  }

  // 3d. Near-pole queries
  for (double lat : {85.0, 89.0, -85.0, -89.0}) {
    for (double lon : {0.0, 90.0, 180.0}) {
      auto xyz = lonlat_to_xyz(lon, lat);
      for (double r : {1.0, 3.0, 8.0})
        queries.push_back(
            {"near-pole lat=" + std::to_string((int)lat), xyz[0], xyz[1],
             xyz[2], r});
    }
  }

  // ---------------------------------------------------------------
  // 4.  Run and compare
  // ---------------------------------------------------------------
  size_t total = 0, failures = 0;
  nanoflann::SearchParameters params;
  params.sorted = false;

  for (auto &q : queries) {
    double qpt[3] = {q.qx, q.qy, q.qz};
    double r2 = deg_to_chord2(q.r_deg);

    // KD-tree result
    std::vector<nanoflann::ResultItem<size_t, double>> kd_matches;
    tree.radiusSearch(qpt, r2, kd_matches, params);
    std::vector<size_t> kd_idx;
    kd_idx.reserve(kd_matches.size());
    for (auto &m : kd_matches)
      kd_idx.push_back(m.first);
    std::sort(kd_idx.begin(), kd_idx.end());

    // Brute-force reference
    auto ref = brute_force_l2(cloud, qpt, r2);

    ++total;
    if (kd_idx != ref) {
      ++failures;
      if (failures <= 5) {
        std::cerr << "[FAIL] " << q.label << "  r=" << q.r_deg
                  << "  KD=" << kd_idx.size() << "  ref=" << ref.size()
                  << "\n";
      }
    }
  }

  std::cout << "\n--- Summary ---\n"
            << "Total queries : " << total << "\n"
            << "Passed        : " << (total - failures) << "\n"
            << "Failed        : " << failures << "\n";

  if (failures == 0)
    std::cout << "All queries validated successfully.\n";
  else
    std::cout << "VALIDATION FAILED\n";

  return (failures == 0) ? 0 : 1;
}

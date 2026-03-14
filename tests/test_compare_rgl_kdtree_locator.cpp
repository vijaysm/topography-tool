/**
 * @file test_rgl_locator.cpp
 * @brief Validation test for RegularGridLocator::radiusSearch
 *
 * Compares RegularGridLocator against a brute-force haversine reference
 * that uses the identical distance metric and <= boundary convention.
 * Any disagreement is a real bug in the RegularGridLocator search logic.
 *
 * Test coverage:
 *   - Random mid-latitude queries with fractional radii
 *   - Near-pole queries (|lat| > 85°)
 *   - Exact pole queries (lat = ±90°) including radii that cross the pole
 *   - Dateline queries (lon near grid boundary)
 *   - Both [0,360) and [-180,180) grid longitude conventions
 *   - Range of radii (tiny → 10°)
 */

#define ELPP_DISABLE_LOGS

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include "easylogging.hpp"
INITIALIZE_EASYLOGGINGPP

#include "RegularGridLocator.hpp"

using namespace moab;

// ---------------------------------------------------------------------------
// Brute-force reference (ground truth)
// ---------------------------------------------------------------------------

/// Brute-force radius search using haversine_distance with <=.
/// Exactly the same metric and boundary convention as RegularGridLocator.
static std::vector<size_t>
brute_force_haversine(const std::vector<double> &lats,
                      const std::vector<double> &lons, double q_lon,
                      double q_lat, double r_deg) {
  std::vector<size_t> out;
  size_t nlon = lons.size();
  for (size_t ilat = 0; ilat < lats.size(); ++ilat) {
    for (size_t ilon = 0; ilon < nlon; ++ilon) {
      double dist =
          haversine_distance(q_lon, q_lat, lons[ilon], lats[ilat]);
      if (dist <= r_deg) {
        out.push_back(ilat * nlon + ilon);
      }
    }
  }
  std::sort(out.begin(), out.end());
  return out;
}

// ---------------------------------------------------------------------------
// RegularGridLocator wrapper
// ---------------------------------------------------------------------------

static std::vector<size_t> rgl_radius_search(RegularGridLocator &loc,
                                              double q_lon, double q_lat,
                                              double r_deg) {
  PointType3D qpt = {q_lon, q_lat, 0.0};
  std::vector<nanoflann::ResultItem<size_t, CoordinateType>> matches;
  loc.radiusSearch(qpt, r_deg, matches, /*sorted=*/false);

  std::vector<size_t> out;
  out.reserve(matches.size());
  for (auto &m : matches)
    out.push_back(m.first);
  std::sort(out.begin(), out.end());
  return out;
}

// ---------------------------------------------------------------------------
// Test infrastructure
// ---------------------------------------------------------------------------

struct TestCase {
  std::string label;
  double lon, lat, radius_deg;
};

/// Tolerance for boundary comparison.  haversine_distance is an inline
/// function; when inlined into two separate translation units the compiler
/// may generate slightly different FP rounding at the exact boundary.
/// Any discrepancy where the point's distance is within this tolerance
/// of the search radius is acceptable.
static constexpr double BOUNDARY_TOL = 1e-10; // degrees

static bool check_one(const TestCase &tc, RegularGridLocator &loc,
                      const std::vector<double> &lats,
                      const std::vector<double> &lons,
                      size_t &total_checked) {
  auto ref = brute_force_haversine(lats, lons, tc.lon, tc.lat, tc.radius_deg);
  auto got = rgl_radius_search(loc, tc.lon, tc.lat, tc.radius_deg);

  ++total_checked;

  if (ref == got)
    return true;

  // Check whether every disagreement is within FP tolerance of the boundary
  std::unordered_set<size_t> ref_set(ref.begin(), ref.end());
  std::unordered_set<size_t> got_set(got.begin(), got.end());

  size_t nlon = lons.size();
  size_t hard_missing = 0, hard_extra = 0;
  size_t boundary_missing = 0, boundary_extra = 0;

  for (size_t idx : ref_set) {
    if (!got_set.count(idx)) {
      size_t ilat = idx / nlon, ilon = idx % nlon;
      double dist = haversine_distance(tc.lon, tc.lat, lons[ilon], lats[ilat]);
      if (std::abs(dist - tc.radius_deg) < BOUNDARY_TOL)
        ++boundary_missing;
      else
        ++hard_missing;
    }
  }
  for (size_t idx : got_set) {
    if (!ref_set.count(idx)) {
      size_t ilat = idx / nlon, ilon = idx % nlon;
      double dist = haversine_distance(tc.lon, tc.lat, lons[ilon], lats[ilat]);
      if (std::abs(dist - tc.radius_deg) < BOUNDARY_TOL)
        ++boundary_extra;
      else
        ++hard_extra;
    }
  }

  // Boundary-only discrepancies are acceptable (FP noise)
  if (hard_missing == 0 && hard_extra == 0)
    return true;

  // Real failure — print diagnostics
  std::cerr << "\n[FAIL] " << tc.label << "\n"
            << "  query = (lon=" << tc.lon << ", lat=" << tc.lat
            << ", r=" << tc.radius_deg << ")\n"
            << "  Reference: " << ref.size() << " pts   RGL: " << got.size()
            << " pts\n"
            << "  hard_missing=" << hard_missing
            << " hard_extra=" << hard_extra
            << " boundary_missing=" << boundary_missing
            << " boundary_extra=" << boundary_extra << "\n";

  int shown = 0;
  for (size_t idx : ref_set) {
    if (!got_set.count(idx) && shown++ < 8) {
      size_t ilat = idx / nlon, ilon = idx % nlon;
      double dist = haversine_distance(tc.lon, tc.lat, lons[ilon], lats[ilat]);
      std::cerr << "  MISSING idx=" << idx << " (lat=" << lats[ilat]
                << " lon=" << lons[ilon] << " dist=" << std::setprecision(15)
                << dist << ")\n";
    }
  }
  shown = 0;
  for (size_t idx : got_set) {
    if (!ref_set.count(idx) && shown++ < 8) {
      size_t ilat = idx / nlon, ilon = idx % nlon;
      double dist = haversine_distance(tc.lon, tc.lat, lons[ilon], lats[ilat]);
      std::cerr << "  EXTRA   idx=" << idx << " (lat=" << lats[ilat]
                << " lon=" << lons[ilon] << " dist=" << std::setprecision(15)
                << dist << ")\n";
    }
  }

  return false;
}

// ---------------------------------------------------------------------------
// Grid builders
// ---------------------------------------------------------------------------

static void build_grid(double dlat, double dlon, double lon_start,
                       std::vector<double> &lats, std::vector<double> &lons) {
  lats.clear();
  lons.clear();
  for (double lat = -90.0; lat <= 90.0 + 1e-9; lat += dlat)
    lats.push_back(lat);
  double lon_end = lon_start + 360.0;
  for (double lon = lon_start; lon < lon_end - 1e-9; lon += dlon)
    lons.push_back(lon);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
  const double DLAT = 2.0;
  const double DLON = 2.0;

  struct GridSpec {
    std::string name;
    double lon_start;
  };
  std::vector<GridSpec> grids = {{"[0,360)", 0.0}, {"[-180,180)", -180.0}};

  size_t total_checked = 0;
  size_t total_failures = 0;

  for (auto &gs : grids) {
    std::vector<double> lats, lons;
    build_grid(DLAT, DLON, gs.lon_start, lats, lons);

    RegularGridLocator loc(lats, lons, HAVERSINE);

    std::cout << "Grid " << gs.name << ": " << lats.size() << " lats x "
              << lons.size() << " lons = " << (lats.size() * lons.size())
              << " points\n";

    std::vector<TestCase> cases;

    // 1. Random mid-latitude queries (fractional radii to avoid boundary ties)
    {
      std::mt19937 rng(42);
      std::uniform_real_distribution<double> lon_d(gs.lon_start,
                                                   gs.lon_start + 360.0 - DLON);
      std::uniform_real_distribution<double> lat_d(-70.0, 70.0);
      std::uniform_real_distribution<double> rad_d(0.3, 8.0);
      for (int i = 0; i < 200; ++i)
        cases.push_back({"mid-lat #" + std::to_string(i), lon_d(rng),
                         lat_d(rng), rad_d(rng)});
    }

    // 2. Near-pole queries
    for (double lat : {85.0, 87.0, 88.0, 89.0, -85.0, -87.0, -89.0}) {
      for (double lon : {0.0, 45.0, 90.0, 180.0, 270.0}) {
        double l = gs.lon_start >= 0 ? lon : lon - 180.0;
        for (double r : {0.5, 1.5, 3.0, 5.0, 8.0})
          cases.push_back({"near-pole lat=" + std::to_string((int)lat), l, lat,
                           r});
      }
    }

    // 3. Cross-pole queries
    for (double lat : {88.0, 89.0, 89.5, -88.0, -89.0, -89.5}) {
      for (double lon : {0.0, 90.0, 180.0}) {
        double l = gs.lon_start >= 0 ? lon : lon - 180.0;
        for (double r : {3.0, 5.0, 7.0, 10.0, 15.0})
          cases.push_back(
              {"cross-pole lat=" + std::to_string(lat), l, lat, r});
      }
    }

    // 4. Exact pole queries
    for (double lat : {90.0, -90.0}) {
      for (double r : {0.5, 1.0, 2.0, 5.0, 10.0, 20.0})
        cases.push_back(
            {"exact-pole lat=" + std::to_string((int)lat), 0.0, lat, r});
    }

    // 5. Dateline / boundary queries
    {
      // Query at first/last grid longitude
      double first_lon = lons.front();
      double last_lon = lons.back();
      double mid_lon = lons[lons.size() / 2];
      for (double lon : {first_lon, last_lon, mid_lon}) {
        for (double lat : {0.0, 45.0, -45.0, 80.0}) {
          for (double r : {1.0, 3.0, 5.0})
            cases.push_back({"dateline lon=" + std::to_string(lon), lon, lat, r});
        }
      }
    }

    // 6. Very small radii
    for (int i = 0; i < 20; ++i) {
      double lat = -80.0 + i * 8.0;
      double lon = gs.lon_start + i * 18.0;
      if (lon >= gs.lon_start + 360.0)
        lon -= 360.0;
      cases.push_back({"tiny-r", lon, lat, 0.1});
    }

    // 7. Large radii (should include many points)
    for (double lat : {0.0, 45.0, 80.0, -80.0}) {
      double lon = gs.lon_start + 60.0;
      cases.push_back({"large-r lat=" + std::to_string((int)lat), lon, lat,
                       20.0});
    }

    // Run all cases
    size_t grid_failures = 0;
    for (auto &tc : cases) {
      if (!check_one(tc, loc, lats, lons, total_checked))
        ++grid_failures;
    }
    total_failures += grid_failures;

    std::cout << "  Tested " << cases.size() << " queries  ->  "
              << (cases.size() - grid_failures) << " passed, " << grid_failures
              << " failed\n";
  }

  std::cout << "\n--- Summary ---\n"
            << "Total queries : " << total_checked << "\n"
            << "Passed        : " << (total_checked - total_failures) << "\n"
            << "Failed        : " << total_failures << "\n";

  if (total_failures == 0)
    std::cout << "All queries validated successfully.\n";
  else
    std::cout << "VALIDATION FAILED\n";

  return (total_failures == 0) ? 0 : 1;
}

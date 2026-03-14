/**
 * @file test_utilities.cpp
 * @brief Exhaustive tests for MBDAUtilities.hpp coordinate transforms,
 *        distance functions, and normalization routines.
 *
 * These functions are the mathematical bedrock of the remapper — any bug
 * here silently corrupts every downstream result.
 *
 * Test categories:
 *   1. RLLtoXYZ_Deg: unit-sphere constraint, known values, poles, equator
 *   2. XYZtoRLL_Deg: inverse transform, round-trip fidelity
 *   3. angular_to_cartesian / cartesian_to_angular: inverse pair
 *   4. normalize_longitude: range, idempotency, edge cases
 *   5. haversine_distance (lon/lat): symmetry, triangle inequality,
 *      known values, wraparound, poles, antipodes
 *   6. haversine_distance (3D): consistency with lon/lat variant
 *   7. euclidean_distance: monotone relationship with haversine
 *   8. compute_distance: dispatch correctness
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "MBDAUtilities.hpp"

// ---------------------------------------------------------------------------
// Tiny test harness
// ---------------------------------------------------------------------------

static int g_tests = 0, g_pass = 0, g_fail = 0;

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

static constexpr double EPS = 1e-12;
static constexpr double LOOSE = 1e-6;

// ---------------------------------------------------------------------------
// 1.  RLLtoXYZ_Deg
// ---------------------------------------------------------------------------

static void test_rll_to_xyz() {
  std::cout << "  RLLtoXYZ_Deg ...\n";

  // Unit-sphere constraint for a sweep of lon/lat
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> lon_d(0, 360);
  std::uniform_real_distribution<double> lat_d(-90, 90);
  for (int i = 0; i < 1000; ++i) {
    double lon = lon_d(rng), lat = lat_d(rng);
    PointType3D p;
    RLLtoXYZ_Deg(lon, lat, p);
    double r = std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
    CHECK(NEAR(r, 1.0, EPS), "unit sphere: r=" + std::to_string(r));
  }

  // Known values: north pole
  {
    PointType3D p;
    RLLtoXYZ_Deg(0.0, 90.0, p);
    CHECK(NEAR(p[0], 0.0, EPS) && NEAR(p[1], 0.0, EPS) && NEAR(p[2], 1.0, EPS),
          "north pole");
  }

  // Known values: south pole
  {
    PointType3D p;
    RLLtoXYZ_Deg(0.0, -90.0, p);
    CHECK(NEAR(p[0], 0.0, EPS) && NEAR(p[1], 0.0, EPS) &&
              NEAR(p[2], -1.0, EPS),
          "south pole");
  }

  // Known values: lon=0, lat=0 → (1, 0, 0)
  {
    PointType3D p;
    RLLtoXYZ_Deg(0.0, 0.0, p);
    CHECK(NEAR(p[0], 1.0, EPS) && NEAR(p[1], 0.0, EPS) && NEAR(p[2], 0.0, EPS),
          "origin");
  }

  // Known values: lon=90, lat=0 → (0, 1, 0)
  {
    PointType3D p;
    RLLtoXYZ_Deg(90.0, 0.0, p);
    CHECK(NEAR(p[0], 0.0, EPS) && NEAR(p[1], 1.0, EPS) && NEAR(p[2], 0.0, EPS),
          "lon=90 equator");
  }

  // Known values: lon=180, lat=0 → (-1, 0, 0)
  {
    PointType3D p;
    RLLtoXYZ_Deg(180.0, 0.0, p);
    CHECK(NEAR(p[0], -1.0, EPS) && NEAR(p[1], 0.0, EPS) &&
              NEAR(p[2], 0.0, EPS),
          "antimeridian equator");
  }

  // lon=360 should equal lon=0
  {
    PointType3D a, b;
    RLLtoXYZ_Deg(0.0, 45.0, a);
    RLLtoXYZ_Deg(360.0, 45.0, b);
    CHECK(NEAR(a[0], b[0], EPS) && NEAR(a[1], b[1], EPS) &&
              NEAR(a[2], b[2], EPS),
          "lon=0 == lon=360");
  }

  // Negative longitude should work
  {
    PointType3D a, b;
    RLLtoXYZ_Deg(-90.0, 0.0, a);
    RLLtoXYZ_Deg(270.0, 0.0, b);
    CHECK(NEAR(a[0], b[0], EPS) && NEAR(a[1], b[1], EPS) &&
              NEAR(a[2], b[2], EPS),
          "lon=-90 == lon=270");
  }
}

// ---------------------------------------------------------------------------
// 2.  XYZtoRLL_Deg and round-trip
// ---------------------------------------------------------------------------

static void test_xyz_to_rll() {
  std::cout << "  XYZtoRLL_Deg round-trip ...\n";

  // Round-trip: lon/lat → XYZ → lon/lat should recover original
  std::mt19937 rng(11);
  std::uniform_real_distribution<double> lon_d(0, 360);
  std::uniform_real_distribution<double> lat_d(-89.99, 89.99); // avoid exact poles
  for (int i = 0; i < 1000; ++i) {
    double lon = lon_d(rng), lat = lat_d(rng);
    PointType3D p;
    RLLtoXYZ_Deg(lon, lat, p);
    double lon2, lat2;
    XYZtoRLL_Deg(p.data(), lon2, lat2);
    // XYZtoRLL returns lon in [0,360), so normalize original
    double lon_n = normalize_longitude(lon);
    CHECK(NEAR(lon_n, lon2, LOOSE) && NEAR(lat, lat2, LOOSE),
          "round-trip lon=" + std::to_string(lon) + " lat=" + std::to_string(lat) +
          " -> lon2=" + std::to_string(lon2) + " lat2=" + std::to_string(lat2));
  }

  // Pole round-trip
  {
    PointType3D p;
    RLLtoXYZ_Deg(123.0, 90.0, p); // lon is arbitrary at pole
    double lon2, lat2;
    XYZtoRLL_Deg(p.data(), lon2, lat2);
    CHECK(NEAR(lat2, 90.0, EPS), "north pole round-trip lat");
  }
  {
    PointType3D p;
    RLLtoXYZ_Deg(45.0, -90.0, p);
    double lon2, lat2;
    XYZtoRLL_Deg(p.data(), lon2, lat2);
    CHECK(NEAR(lat2, -90.0, EPS), "south pole round-trip lat");
  }

  // Non-unit-sphere input should be normalized
  {
    double coords[3] = {2.0, 0.0, 0.0}; // 2x unit sphere
    double lon, lat;
    XYZtoRLL_Deg(coords, lon, lat);
    CHECK(NEAR(lon, 0.0, EPS) && NEAR(lat, 0.0, EPS),
          "non-unit sphere normalization");
  }
}

// ---------------------------------------------------------------------------
// 3.  angular_to_cartesian / cartesian_to_angular
// ---------------------------------------------------------------------------

static void test_angular_cartesian_conversion() {
  std::cout << "  angular_to_cartesian / cartesian_to_angular ...\n";

  // They should be mutual inverses
  for (double ang = 0.01; ang < M_PI; ang += 0.1) {
    double chord = angular_to_cartesian(ang);
    double ang2 = cartesian_to_angular(chord);
    CHECK(NEAR(ang, ang2, EPS),
          "inverse at ang=" + std::to_string(ang));
  }

  // Known values
  CHECK(NEAR(angular_to_cartesian(0.0), 0.0, EPS), "ang=0 -> chord=0");
  CHECK(NEAR(angular_to_cartesian(M_PI), 2.0, EPS), "ang=pi -> chord=2 (diameter)");
  CHECK(NEAR(angular_to_cartesian(M_PI / 2), std::sqrt(2.0), EPS),
        "ang=pi/2 -> chord=sqrt(2)");

  // Boundary: cartesian_to_angular(0) = 0
  CHECK(NEAR(cartesian_to_angular(0.0), 0.0, EPS), "chord=0 -> ang=0");
  CHECK(NEAR(cartesian_to_angular(2.0), M_PI, EPS), "chord=2 -> ang=pi");
}

// ---------------------------------------------------------------------------
// 4.  normalize_longitude
// ---------------------------------------------------------------------------

static void test_normalize_longitude() {
  std::cout << "  normalize_longitude ...\n";

  // Output always in [0, 360)
  for (double lon : {0.0, 180.0, 359.99, -1.0, -180.0, 360.0, 720.0, -360.0,
                     -0.001, 500.0, -500.0}) {
    double n = normalize_longitude(lon);
    CHECK(n >= 0.0 && n < 360.0,
          "range for lon=" + std::to_string(lon) + " got=" + std::to_string(n));
  }

  // Idempotent: f(f(x)) == f(x)
  for (double lon : {-123.456, 0.0, 180.0, 359.999, 400.0}) {
    double n1 = normalize_longitude(lon);
    double n2 = normalize_longitude(n1);
    CHECK(NEAR(n1, n2, EPS),
          "idempotent for lon=" + std::to_string(lon));
  }

  // Known mappings
  CHECK(NEAR(normalize_longitude(0.0), 0.0, EPS), "0 -> 0");
  CHECK(NEAR(normalize_longitude(-90.0), 270.0, EPS), "-90 -> 270");
  CHECK(NEAR(normalize_longitude(360.0), 0.0, EPS), "360 -> 0");
  CHECK(NEAR(normalize_longitude(450.0), 90.0, EPS), "450 -> 90");
}

// ---------------------------------------------------------------------------
// 5.  haversine_distance (lon/lat)
// ---------------------------------------------------------------------------

static void test_haversine_lonlat() {
  std::cout << "  haversine_distance (lon/lat) ...\n";

  // (a) Zero distance: same point
  CHECK(NEAR(haversine_distance(10, 20, 10, 20), 0.0, EPS), "same point");

  // (b) Symmetry: d(A,B) == d(B,A)
  {
    std::mt19937 rng(99);
    std::uniform_real_distribution<double> lon_d(0, 360), lat_d(-90, 90);
    for (int i = 0; i < 200; ++i) {
      double lon1 = lon_d(rng), lat1 = lat_d(rng);
      double lon2 = lon_d(rng), lat2 = lat_d(rng);
      double d1 = haversine_distance(lon1, lat1, lon2, lat2);
      double d2 = haversine_distance(lon2, lat2, lon1, lat1);
      CHECK(NEAR(d1, d2, EPS), "symmetry i=" + std::to_string(i));
    }
  }

  // (c) Non-negative
  {
    std::mt19937 rng(77);
    std::uniform_real_distribution<double> lon_d(0, 360), lat_d(-90, 90);
    for (int i = 0; i < 200; ++i) {
      double d = haversine_distance(lon_d(rng), lat_d(rng), lon_d(rng), lat_d(rng));
      CHECK(d >= 0.0, "non-negative i=" + std::to_string(i));
    }
  }

  // (d) Maximum distance: antipodal points = 180 degrees
  CHECK(NEAR(haversine_distance(0, 0, 180, 0), 180.0, LOOSE), "antipodal equator");
  CHECK(NEAR(haversine_distance(0, 90, 0, -90), 180.0, LOOSE), "pole-to-pole");

  // (e) Known value: latitude-only distance
  CHECK(NEAR(haversine_distance(0, 0, 0, 90), 90.0, LOOSE), "equator to north pole");
  CHECK(NEAR(haversine_distance(10, 45, 10, 50), 5.0, LOOSE), "pure latitude 5 deg");

  // (f) Longitude wraparound: d(0,0, 359,0) should be ~1 degree, not 359
  {
    double d = haversine_distance(0, 0, 359, 0);
    CHECK(NEAR(d, 1.0, LOOSE), "lon wraparound 0->359 = 1deg, got=" + std::to_string(d));
  }

  // (g) Wraparound with negative longitudes
  {
    double d = haversine_distance(-1, 0, 1, 0);
    CHECK(NEAR(d, 2.0, LOOSE), "lon -1 to 1 = 2deg");
  }

  // (h) Large longitude gap that wraps
  {
    double d = haversine_distance(10, 0, 350, 0);
    CHECK(NEAR(d, 20.0, LOOSE), "lon 10->350 wraps to 20deg");
  }

  // (i) Triangle inequality: d(A,C) <= d(A,B) + d(B,C)
  {
    std::mt19937 rng(33);
    std::uniform_real_distribution<double> lon_d(0, 360), lat_d(-90, 90);
    for (int i = 0; i < 200; ++i) {
      double la1 = lon_d(rng), la2 = lat_d(rng);
      double lb1 = lon_d(rng), lb2 = lat_d(rng);
      double lc1 = lon_d(rng), lc2 = lat_d(rng);
      double dAB = haversine_distance(la1, la2, lb1, lb2);
      double dBC = haversine_distance(lb1, lb2, lc1, lc2);
      double dAC = haversine_distance(la1, la2, lc1, lc2);
      CHECK(dAC <= dAB + dBC + LOOSE,
            "triangle inequality i=" + std::to_string(i));
    }
  }

  // (j) Pole distances: all longitudes at a pole are equidistant from a
  //     fixed point
  {
    double ref = haversine_distance(0, 45, 0, 90);
    for (double lon : {0.0, 90.0, 180.0, 270.0, 359.0}) {
      double d = haversine_distance(0, 45, lon, 90);
      CHECK(NEAR(d, ref, LOOSE),
            "pole equidistant lon=" + std::to_string(lon));
    }
  }
}

// ---------------------------------------------------------------------------
// 6.  haversine_distance (3D variant)
// ---------------------------------------------------------------------------

static void test_haversine_3d() {
  std::cout << "  haversine_distance (3D) ...\n";

  // Consistency: 3D variant should agree with lon/lat variant
  std::mt19937 rng(55);
  std::uniform_real_distribution<double> lon_d(0, 360), lat_d(-90, 90);
  for (int i = 0; i < 200; ++i) {
    double lon1 = lon_d(rng), lat1 = lat_d(rng);
    double lon2 = lon_d(rng), lat2 = lat_d(rng);

    // lon/lat variant returns degrees
    double d_deg = haversine_distance(lon1, lat1, lon2, lat2);

    // 3D variant returns radians (on unit sphere)
    PointType3D a, b;
    RLLtoXYZ_Deg(lon1, lat1, a);
    RLLtoXYZ_Deg(lon2, lat2, b);
    double d_rad = haversine_distance(a.data(), b.data());
    double d_deg_from_3d = d_rad * RAD_TO_DEG;

    CHECK(NEAR(d_deg, d_deg_from_3d, LOOSE),
          "3D vs lonlat consistency i=" + std::to_string(i) +
          " d1=" + std::to_string(d_deg) + " d2=" + std::to_string(d_deg_from_3d));
  }
}

// ---------------------------------------------------------------------------
// 7.  euclidean_distance
// ---------------------------------------------------------------------------

static void test_euclidean_distance() {
  std::cout << "  euclidean_distance ...\n";

  // (a) Same point → 0
  CHECK(NEAR(euclidean_distance(10, 20, 10, 20), 0.0, EPS), "same point");

  // (b) Monotone relationship with haversine: if haversine(A,B) < haversine(A,C)
  //     then euclidean(A,B) < euclidean(A,C) (chord is monotone with arc)
  {
    std::mt19937 rng(44);
    std::uniform_real_distribution<double> lon_d(0, 360), lat_d(-90, 90);
    int violations = 0;
    for (int i = 0; i < 500; ++i) {
      double la = lon_d(rng), lata = lat_d(rng);
      double lb = lon_d(rng), latb = lat_d(rng);
      double lc = lon_d(rng), latc = lat_d(rng);
      double hAB = haversine_distance(la, lata, lb, latb);
      double hAC = haversine_distance(la, lata, lc, latc);
      double eAB = euclidean_distance(la, lata, lb, latb);
      double eAC = euclidean_distance(la, lata, lc, latc);
      if (hAB < hAC - LOOSE && eAB > eAC + LOOSE)
        ++violations;
      if (hAB > hAC + LOOSE && eAB < eAC - LOOSE)
        ++violations;
    }
    CHECK(violations == 0,
          "monotone relationship violations=" + std::to_string(violations));
  }

  // (c) Antipodal chord distance = 2 (diameter of unit sphere)
  CHECK(NEAR(euclidean_distance(0, 0, 180, 0), 2.0, LOOSE), "antipodal chord=2");

  // (d) 90-degree separation → chord = sqrt(2)
  CHECK(NEAR(euclidean_distance(0, 0, 90, 0), std::sqrt(2.0), LOOSE),
        "90deg chord=sqrt2");
}

// ---------------------------------------------------------------------------
// 8.  compute_distance dispatcher
// ---------------------------------------------------------------------------

static void test_compute_distance() {
  std::cout << "  compute_distance dispatcher ...\n";

  double lon1 = 10, lat1 = 20, lon2 = 30, lat2 = 40;

  double h = haversine_distance(lon1, lat1, lon2, lat2);
  double e = euclidean_distance(lon1, lat1, lon2, lat2);
  double ch = compute_distance(lon1, lat1, lon2, lat2, HAVERSINE);
  double ce = compute_distance(lon1, lat1, lon2, lat2, CARTESIAN);

  CHECK(NEAR(h, ch, EPS), "HAVERSINE dispatch");
  CHECK(NEAR(e, ce, EPS), "CARTESIAN dispatch");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
  std::cout << "=== test_utilities ===\n";

  test_rll_to_xyz();
  test_xyz_to_rll();
  test_angular_cartesian_conversion();
  test_normalize_longitude();
  test_haversine_lonlat();
  test_haversine_3d();
  test_euclidean_distance();
  test_compute_distance();

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

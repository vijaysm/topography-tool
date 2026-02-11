
#ifndef MBDA_UTILITIES_HPP
#define MBDA_UTILITIES_HPP

//===========================================================================
// Type Definitions and Utility Functions
//===========================================================================

/// Type aliases for coordinate systems and data types
typedef double CoordinateType;
typedef std::array<CoordinateType, 3> PointType3D; // 3D Cartesian coordinates
typedef std::array<CoordinateType, 2> PointType;   // 2D lon/lat coordinates

// Constants
static constexpr double EARTH_RADIUS_KM = 6371.0;
static constexpr double DEG_TO_RAD = M_PI / 180.0;
static constexpr double RAD_TO_DEG = 180.0 / M_PI;
static constexpr double POLE_TOLERANCE = 1e-6; // Degrees from pole

/// Numerical tolerance for coordinate comparisons
static constexpr CoordinateType ReferenceTolerance = 1e-12;

enum DistanceMetric {
  HAVERSINE, // Great circle distance on sphere (accurate)
  CARTESIAN  // Euclidean distance in 3D Cartesian space (fast approximation)
};

/**
 * @brief Convert longitude/latitude (degrees) to 3D Cartesian coordinates

 * Projects geographic coordinates onto a unit sphere using spherical
 * coordinate transformation. This is useful for distance calculations
 * and spatial indexing.
 *
 * @param lon_deg Longitude in degrees
 * @param lat_deg Latitude in degrees
 * @param coordinates Output 3D Cartesian coordinates [x, y, z]
 * @return MB_SUCCESS on success
 */
template <typename T>
inline void RLLtoXYZ_Deg(T lon_deg, T lat_deg, T *coordinates) {
  // Convert to radians
  T lon_rad = lon_deg * M_PI / 180.0;
  T lat_rad = lat_deg * M_PI / 180.0;

  // Spherical to Cartesian transformation
  T cos_lat = cos(lat_rad);
  coordinates[0] = cos_lat * cos(lon_rad); // x = cos(lat) * cos(lon)
  coordinates[1] = cos_lat * sin(lon_rad); // y = cos(lat) * sin(lon)
  coordinates[2] = sin(lat_rad);           // z = sin(lat)

  return;
}

/**
 * @brief Convert longitude/latitude (degrees) to 3D Cartesian coordinates
 *
 * Convenience wrapper using std::array interface.
 *
 * @param lon_deg Longitude in degrees
 * @param lat_deg Latitude in degrees
 * @param coordinates Output 3D Cartesian coordinates array
 */
template <typename T>
inline void RLLtoXYZ_Deg(T lon_deg, T lat_deg, std::array<T, 3> &coordinates) {
  RLLtoXYZ_Deg(lon_deg, lat_deg, coordinates.data());
}

// Convert angular distance (radians) to Cartesian distance on unit sphere
inline double angular_to_cartesian(double angular_distance_radians) {
  return 2.0 * std::sin(angular_distance_radians / 2.0);
}

// Convert Cartesian distance to angular distance (radians) on unit sphere
inline double cartesian_to_angular(double cartesian_distance) {
  return 2.0 * std::asin(cartesian_distance / 2.0);
}

/**
 * @brief Calculate latitude and longitude from 3D Cartesian coordinates
 *
 * Performs inverse spherical projection from Cartesian coordinates back
 * to geographic coordinates (degrees). Handles pole special cases.
 *
 * @param coordinates Input 3D Cartesian coordinates
 * @param lon_deg Output longitude in degrees [0, 360)
 * @param lat_deg Output latitude in degrees [-90, 90]
 * @return MB_SUCCESS on success
 */
template <typename SType, typename T>
inline void XYZtoRLL_Deg(const SType *coordinates, T &lon_deg, T &lat_deg) {
  // Normalize to unit sphere
  SType dMag = std::sqrt(coordinates[0] * coordinates[0] +
                         coordinates[1] * coordinates[1] +
                         coordinates[2] * coordinates[2]);
  SType x = coordinates[0] / dMag;
  SType y = coordinates[1] / dMag;
  SType z = coordinates[2] / dMag;

  // Handle pole special cases
  if (fabs(z) < 1.0 - ReferenceTolerance) {
    // Standard case: use atan2 and asin
    lon_deg = (atan2(y, x)) * 180.0 / M_PI;
    lat_deg = (asin(z)) * 180.0 / M_PI;

    // Normalize longitude to [0, 360)
    if (lon_deg < 0.0) {
      lon_deg += 360.0;
    }
  } else if (z > 0.0) {
    // North pole
    lon_deg = 0.0;
    lat_deg = 90.0;
  } else {
    // South pole
    lon_deg = 0.0;
    lat_deg = -90.0;
  }

  return;
}

//===========================================================================
// Distance Computation Methods
//===========================================================================

/**
 * @brief Normalize longitude to [m_lon_min, m_lon_min + 360) range
 *
 * Ensures longitude values are within the valid range for the grid.
 * Handles wraparound for global grids.
 *
 * @param lon Input longitude in degrees
 * @return Normalized longitude in degrees
 */
inline double normalize_longitude(double lon) {
  // Normalize to [0, 360) range
  while (lon < 0)
    lon += 360.0;
  while (lon >= 360.0)
    lon -= 360.0;
  return lon;
}

/**
 * @brief Compute Euclidean distance between two points
 * @tparam T Point type (array-like with size() method)
 * @param p1 First point
 * @param p2 Second point
 * @return Euclidean distance
 */
template <typename T>
inline typename T::value_type private_compute_distance(const T &p1,
                                                       const T &p2) {
  typename T::value_type dMag = 0.0;
  for (size_t i = 0; i < p1.size(); ++i) {
    dMag += (p1[i] - p2[i]) * (p1[i] - p2[i]);
  }
  return std::sqrt(dMag);
}

/**
 * @brief Compute distance between 3D Cartesian point and 2D lon/lat point
 * @param p1 3D Cartesian point
 * @param p2 2D lon/lat point
 * @return Distance on unit sphere
 */
inline CoordinateType private_compute_distance(const PointType3D &p1,
                                               const PointType &p2) {
  PointType3D p2_3d;
  RLLtoXYZ_Deg(p2[0], p2[1], p2_3d);
  return private_compute_distance(p1, p2_3d);
}

/**
 * @brief Compute Haversine (great circle) distance in degrees
 *
 * Calculates the great circle distance between two points on a sphere
 * using the Haversine formula. This is the most accurate method for
 * spherical distances.
 *
 * @param lon1 Longitude of first point (degrees)
 * @param lat1 Latitude of first point (degrees)
 * @param lon2 Longitude of second point (degrees)
 * @param lat2 Latitude of second point (degrees)
 * @return Distance in degrees (angular distance)
 */
inline double haversine_distance(double lon1, double lat1, double lon2,
                                 double lat2) {
  // Convert to radians
  double lat1_rad = lat1 * DEG_TO_RAD;
  double lat2_rad = lat2 * DEG_TO_RAD;
  double dlon = (lon2 - lon1) * DEG_TO_RAD;
  double dlat = (lat2 - lat1) * DEG_TO_RAD;

  // Haversine formula: a = sin²(Δφ/2) + cos φ1 ⋅ cos φ2 ⋅ sin²(Δλ/2)
  double a = std::sin(dlat / 2.0) * std::sin(dlat / 2.0) +
             std::cos(lat1_rad) * std::cos(lat2_rad) * std::sin(dlon / 2.0) *
                 std::sin(dlon / 2.0);
  double c = 2.0 * std::atan2(std::sqrt(a), std::sqrt(1.0 - a));

  // Return distance in degrees (angular distance)
  return c / DEG_TO_RAD;
}

/**
 * @brief Compute Euclidean L2 distance in lat/lon space
 *
 * Fast approximation for small distances where the curvature of
 * the Earth can be ignored. Handles longitude wraparound by choosing
 * the shorter path around the sphere.
 *
 * @param lon1 Longitude of first point (degrees)
 * @param lat1 Latitude of first point (degrees)
 * @param lon2 Longitude of second point (degrees)
 * @param lat2 Latitude of second point (degrees)
 * @return Euclidean distance in degrees
 */
inline double euclidean_distance(double lon1, double lat1, double lon2,
                                 double lat2) {
  PointType3D p1, p2;
  RLLtoXYZ_Deg(lon1, lat1, p1);
  RLLtoXYZ_Deg(lon2, lat2, p2);

  return private_compute_distance(p1, p2);
}

/**
 * @brief Compute distance between two points based on selected metric
 *
 * Dispatch function that routes to the appropriate distance computation
 * method based on the configured distance metric.
 *
 * @param lon1 Longitude of first point (degrees)
 * @param lat1 Latitude of first point (degrees)
 * @param lon2 Longitude of second point (degrees)
 * @param lat2 Latitude of second point (degrees)
 * @return Distance in degrees
 */
inline double compute_distance(double lon1, double lat1, double lon2,
                               double lat2, DistanceMetric metric) {
  if (metric == HAVERSINE) {
    return haversine_distance(lon1, lat1, lon2, lat2);
  } else {
    return euclidean_distance(lon1, lat1, lon2, lat2);
  }
}

#endif // MBDA_UTILITIES_HPP

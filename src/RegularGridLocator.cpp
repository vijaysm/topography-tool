/**
 * @file RegularGridLocator.cpp
 * @brief Implementation of fast spatial queries for regular lat/lon grids
 *
 * This file implements spatial indexing for structured latitude/longitude
 * grids, providing O(1) radius and k-nearest neighbor searches without the
 * overhead of building a KD-tree. The implementation handles spherical
 * geometry, longitude wraparound, and pole special cases.
 *
 * Key Features:
 * - O(1) spatial queries for regular grids
 * - Haversine (great circle) and Euclidean distance metrics
 * - Thread-safe for parallel queries
 * - Handles longitude wraparound and pole special cases
 * - Proper cross-pole radius search handling
 * - Efficient spiral search for k-nearest neighbors
 *
 * Author: Vijay Mahadevan
 * Date: 2025-2026
 */

#include "RegularGridLocator.hpp"
#include "easylogging.hpp"
#include <cmath>
#include <iostream>
#include <limits>

namespace moab {

//===========================================================================
// Constructor and Initialization
//===========================================================================

/**
 * @brief Construct locator from explicit lat/lon coordinate vectors
 *
 * Initializes the spatial index for regular latitude/longitude grids.
 * The grid is assumed to be organized as nlat × nlon with longitude
 * changing fastest: Index(ilat, ilon) = ilat * nlon + ilon
 *
 * @param lats Latitude values (degrees, typically -90 to 90)
 * @param lons Longitude values (degrees, typically 0 to 360 or -180 to 180)
 * @param metric Distance metric to use for queries (HAVERSINE or EUCLIDEAN_L2)
 *
 * @throws std::runtime_error if lat/lon arrays are empty
 */
RegularGridLocator::RegularGridLocator(const std::vector<double> &lats,
                                       const std::vector<double> &lons,
                                       DistanceMetric metric)
    : m_lats(lats), m_lons(lons), m_nlat(lats.size()), m_nlon(lons.size()),
      m_metric(metric) {
  // Validate input
  if (m_nlat == 0 || m_nlon == 0) {
    throw std::runtime_error("RegularGridLocator: Empty lat/lon arrays");
  }

  // Compute grid bounds and spacing
  m_lat_min = *std::min_element(m_lats.begin(), m_lats.end());
  m_lat_max = *std::max_element(m_lats.begin(), m_lats.end());
  m_lon_min = *std::min_element(m_lons.begin(), m_lons.end());
  m_lon_max = *std::max_element(m_lons.begin(), m_lons.end());

  // Compute average spacing (for regular grids this should be uniform)
  m_dlat = (m_nlat > 1) ? (m_lat_max - m_lat_min) / (m_nlat - 1) : 0.0;
  m_dlon = (m_nlon > 1) ? (m_lon_max - m_lon_min) / (m_nlon - 1) : 0.0;

  // Log initialization details
  LOG(INFO) << "RegularGridLocator initialized: " << m_nlat << " x " << m_nlon
            << " grid, lat[" << m_lat_min << ", " << m_lat_max << "] lon["
            << m_lon_min << ", " << m_lon_max << "]";
  LOG(INFO) << "  Average spacing: dlat=" << m_dlat << " dlon=" << m_dlon;
  LOG(INFO) << "  Distance metric: "
            << (m_metric == HAVERSINE ? "Haversine" : "Euclidean L2");
}

//===========================================================================
// Utility Methods
//===========================================================================

/**
 * @brief Find nearest grid index for a given coordinate value
 *
 * Uses binary search to find the closest coordinate value in a sorted array.
 * This is O(log n) complexity and provides the most accurate index for
 * a given coordinate value.
 *
 * @param coords Sorted coordinate array (latitude or longitude)
 * @param value Coordinate value to find
 * @return Index of nearest coordinate
 */
size_t RegularGridLocator::find_nearest_index(const std::vector<double> &coords,
                                              double value) const {
  // Binary search for insertion point
  auto it = std::lower_bound(coords.begin(), coords.end(), value);

  // Handle edge cases
  if (it == coords.begin())
    return 0;
  if (it == coords.end())
    return coords.size() - 1;

  // Check which neighbor is closer
  size_t idx = std::distance(coords.begin(), it);
  if (std::abs(coords[idx] - value) < std::abs(coords[idx - 1] - value)) {
    return idx;
  } else {
    return idx - 1;
  }
}

/**
 * @brief Find index bounds for radius search
 *
 * Computes the latitude and longitude index ranges that should be searched
 * for a radius query. Handles longitude wraparound and pole crossing.
 *
 * Algorithm:
 * 1. Normalize query longitude to grid range
 * 2. Compute latitude bounds with pole-crossing detection
 * 3. Compute longitude bounds with wraparound detection
 * 4. Adjust longitude radius for latitude compression at poles
 *
 * For pole crossing: if query_lat + radius > 90 or query_lat - radius < -90,
 * the search region "wraps over" the pole. In spherical geometry, this means
 * points at lat=89 with lon=X and lat=89 with lon=X+180 are both close to the
 * pole. When crossing a pole, we must search all longitudes.
 *
 * @param query_lon Query longitude (degrees)
 * @param query_lat Query latitude (degrees)
 * @param radius Search radius (degrees, angular distance)
 * @param ilat_min Output: minimum latitude index
 * @param ilat_max Output: maximum latitude index
 * @param ilon_min Output: minimum longitude index
 * @param ilon_max Output: maximum longitude index
 * @param wraps_around Output: true if longitude range wraps around (including pole crossing)
 */
void RegularGridLocator::get_search_bounds(double query_lon, double query_lat,
                                           double radius, size_t &ilat_min,
                                           size_t &ilat_max, size_t &ilon_min,
                                           size_t &ilon_max,
                                           bool &wraps_around) const {
  wraps_around = false;

  // Normalize query longitude to grid range
  query_lon = normalize_longitude_to_grid(query_lon);

  // Conservative expansion factor to account for discrete grid sampling
  // and ensure we don't miss boundary cases
  double factor = 1.15;
  double expanded_radius = radius * factor;

  // Compute raw latitude bounds
  double lat_search_min = query_lat - expanded_radius;
  double lat_search_max = query_lat + expanded_radius;

  // Check for pole crossing
  // If radius extends past a pole, points on the "other side" of the pole
  // (180 degrees away in longitude) are actually within the radius
  bool crosses_north_pole = lat_search_max > 90.0;
  bool crosses_south_pole = lat_search_min < -90.0;

  if (crosses_north_pole || crosses_south_pole) {
    // Pole crossing: we must search all longitudes
    wraps_around = true;

    if (crosses_north_pole && crosses_south_pole) {
      // Radius spans both poles - search entire grid
      ilat_min = 0;
      ilat_max = m_nlat - 1;
    } else if (crosses_north_pole) {
      // Crossing north pole: the "reflected" latitude is 180 - lat_search_max
      // Points at lat > (180 - lat_search_max) with lon+180 are within radius
      double reflected_lat = 180.0 - lat_search_max;
      lat_search_min = std::min(lat_search_min, reflected_lat);
      lat_search_max = 90.0;
      lat_search_min = std::max(m_lat_min, lat_search_min);
      ilat_min = find_nearest_index(m_lats, lat_search_min);
      ilat_max = m_nlat - 1; // Include up to the pole
    } else {
      // Crossing south pole: reflected latitude is -180 - lat_search_min
      double reflected_lat = -180.0 - lat_search_min;
      lat_search_max = std::max(lat_search_max, reflected_lat);
      lat_search_min = -90.0;
      lat_search_max = std::min(m_lat_max, lat_search_max);
      ilat_min = 0; // Include down to the pole
      ilat_max = find_nearest_index(m_lats, lat_search_max);
    }

    // Ensure ilat_min <= ilat_max
    if (ilat_min > ilat_max)
      std::swap(ilat_min, ilat_max);

    // All longitudes for pole crossing
    ilon_min = 0;
    ilon_max = m_nlon - 1;
    return;
  }

  // No pole crossing - standard latitude clipping
  lat_search_min = std::max(m_lat_min, lat_search_min);
  lat_search_max = std::min(m_lat_max, lat_search_max);

  ilat_min = find_nearest_index(m_lats, lat_search_min);
  ilat_max = find_nearest_index(m_lats, lat_search_max);

  // Ensure ilat_min <= ilat_max (binary search may return them swapped)
  if (ilat_min > ilat_max)
    std::swap(ilat_min, ilat_max);

  // Longitude bounds with proper spherical geometry consideration
  // At high latitudes, equal angular distance spans more longitude degrees
  // because meridians converge toward the poles.
  //
  // The longitude span for a given angular radius at latitude φ is:
  //   Δλ = arcsin(sin(r) / cos(φ))  for Haversine
  // We use a conservative approximation: radius / cos(lat)
  // with safeguards near poles

  double lat_rad = std::abs(query_lat) * DEG_TO_RAD;
  double cos_lat = std::cos(lat_rad);

  // Near poles (within ~0.5 degrees), search all longitudes
  // This threshold corresponds to cos(89.5°) ≈ 0.0087
  const double NEAR_POLE_COS_THRESHOLD = 0.01;

  double lon_radius;
  if (cos_lat < NEAR_POLE_COS_THRESHOLD) {
    // Very close to pole - search all longitudes
    wraps_around = true;
    ilon_min = 0;
    ilon_max = m_nlon - 1;
    return;
  } else {
    // Standard case: expand longitude search based on latitude
    lon_radius = expanded_radius / cos_lat;
  }

  double lon_search_min = query_lon - lon_radius;
  double lon_search_max = query_lon + lon_radius;

  // Check if longitude range exceeds 180 degrees (would wrap more than halfway)
  // or crosses the grid boundaries
  bool exceeds_half_globe = (lon_search_max - lon_search_min) >= 180.0;
  bool crosses_lon_boundary = (lon_search_min < m_lon_min) ||
                               (lon_search_max > m_lon_max);

  if (exceeds_half_globe || crosses_lon_boundary) {
    wraps_around = true;
    ilon_min = 0;
    ilon_max = m_nlon - 1;
  } else {
    ilon_min = find_nearest_index(m_lons, lon_search_min);
    ilon_max = find_nearest_index(m_lons, lon_search_max);
    // Ensure ilon_min <= ilon_max
    if (ilon_min > ilon_max)
      std::swap(ilon_min, ilon_max);
  }
}

//===========================================================================
// Spatial Query Methods
//===========================================================================

/**
 * @brief Radius search - find all points within given radius
 *
 * Finds all grid points within a specified radius of the query point.
 * This is optimized for regular grids by computing search bounds and
 * only examining points within those bounds.
 *
 * Algorithm:
 * 1. Handle special case: query at pole
 * 2. Compute search bounds using get_search_bounds()
 * 3. Search within bounds, computing distances
 * 4. Sort results by distance
 *
 * @param query_point Query coordinates [lon, lat, z] in degrees (z ignored)
 * @param radius Search radius in degrees
 * @param matches Output vector of (index, distance_squared) pairs
 * @return Number of matches found
 */
size_t RegularGridLocator::radiusSearch(
    const PointType3D &query_point, CoordinateType radius,
    std::vector<nanoflann::ResultItem<size_t, CoordinateType>> &matches,
    bool sorted) const {
  matches.clear();

  CoordinateType query_lon = query_point[0];
  CoordinateType query_lat = query_point[1];

  // Normalize query longitude to grid range
  query_lon = normalize_longitude_to_grid(query_lon);

  // Check if grid actually has poles
  const bool grid_has_north_pole = is_at_pole(m_lat_max);
  const bool grid_has_south_pole = is_at_pole(m_lat_min);

  // Special case: if query is at pole and grid has that pole
  if (is_at_pole(query_lat) && (grid_has_north_pole || grid_has_south_pole)) {
    // Find the latitude index closest to the pole
    size_t pole_ilat = (query_lat > 0) ? m_nlat - 1 : 0;
    CoordinateType pole_lat = m_lats[pole_ilat];

    // Check if pole is within radius
    CoordinateType pole_dist = std::abs(query_lat - pole_lat);
    if (pole_dist <= radius) {
      // All longitudes at pole are valid (they're all the same point)
      for (size_t ilon = 0; ilon < m_nlon; ++ilon) {
        size_t idx = get_linear_index(pole_ilat, ilon);
        matches.push_back({idx, pole_dist * pole_dist});
      }
    }

    // Also search adjacent latitudes if radius extends beyond the pole point
    if (radius > pole_dist) {
      double remaining_radius = radius - pole_dist;
      // Search all latitudes within remaining_radius of the pole
      for (size_t ilat = 0; ilat < m_nlat; ++ilat) {
        if (ilat == pole_ilat)
          continue; // Already added

        CoordinateType grid_lat = m_lats[ilat];
        CoordinateType lat_dist_from_pole = std::abs(grid_lat - pole_lat);

        // For points near pole, search all longitudes
        if (lat_dist_from_pole <= remaining_radius) {
          for (size_t ilon = 0; ilon < m_nlon; ++ilon) {
            CoordinateType grid_lon = m_lons[ilon];
            CoordinateType dist = compute_distance(query_lon, query_lat,
                                                   grid_lon, grid_lat, m_metric);

            if (dist <= radius) {
              size_t idx = get_linear_index(ilat, ilon);
              matches.push_back({idx, dist * dist});
            }
          }
        }
      }
    }

    // Sort results by distance (ascending order)
    if (sorted) {
      std::sort(matches.begin(), matches.end(),
                [](const auto &a, const auto &b) { return a.second < b.second; });
    }

    return matches.size();
  }

  // Get search bounds for efficient querying
  size_t ilat_min, ilat_max, ilon_min, ilon_max;
  bool wraps_around;
  get_search_bounds(query_lon, query_lat, radius, ilat_min, ilat_max, ilon_min,
                    ilon_max, wraps_around);

  // Search within computed bounds
  for (size_t ilat = ilat_min; ilat <= ilat_max; ++ilat) {
    CoordinateType grid_lat = m_lats[ilat];
    bool lat_at_pole = is_at_pole(grid_lat);

    if (wraps_around) {
      // Search all longitudes (wraparound case)
      for (size_t ilon = 0; ilon < m_nlon; ++ilon) {
        CoordinateType grid_lon = m_lons[ilon];
        CoordinateType dist = compute_distance(query_lon, query_lat, grid_lon,
                                               grid_lat, m_metric);

        if (dist <= radius) {
          size_t idx = get_linear_index(ilat, ilon);
          matches.push_back({idx, dist * dist});

          // If this latitude is at pole, we only need one longitude value
          if (lat_at_pole)
            break;
        }
      }
    } else {
      // Search limited longitude range
      for (size_t ilon = ilon_min; ilon <= ilon_max; ++ilon) {
        CoordinateType grid_lon = m_lons[ilon];
        CoordinateType dist = compute_distance(query_lon, query_lat, grid_lon,
                                               grid_lat, m_metric);

        if (dist <= radius) {
          size_t idx = get_linear_index(ilat, ilon);
          matches.push_back({idx, dist * dist});

          // If this latitude is at pole, we only need one longitude value
          if (lat_at_pole)
            break;
        }
      }
    }
  }

  // Sort results by distance (ascending order)
  if (sorted) {
    std::sort(matches.begin(), matches.end(),
              [](const auto &a, const auto &b) { return a.second < b.second; });
  }

  return matches.size();
}

/**
 * @brief K nearest neighbor search
 *
 * Finds the k nearest grid points to the query point using an efficient
 * spiral search algorithm. This is optimized for regular grids where the
 * nearest points are likely to be close in index space.
 *
 * Algorithm:
 * 1. Handle special case: query at pole
 * 2. Find nearest grid point as starting position
 * 3. Use max-heap to maintain k nearest neighbors
 * 4. Spiral search outward from starting position
 * 5. Handle longitude wraparound during search
 * 6. Return sorted results
 *
 * @param query_point Query coordinates [lon, lat, z] in degrees (z ignored)
 * @param k Number of neighbors to find
 * @param indices Output array of k indices (filled with -1 if not found)
 * @param distances_sq Output array of k squared distances
 */
void RegularGridLocator::knnSearch(const PointType3D &query_point, size_t k,
                                   size_t *indices,
                                   CoordinateType *distances_sq,
                                   bool sorted) const {
  CoordinateType query_lon = query_point[0];
  CoordinateType query_lat = query_point[1];

  // Normalize query longitude to grid range
  query_lon = normalize_longitude_to_grid(query_lon);

  // Use a max-heap to keep track of k nearest neighbors
  // Pair: (distance_sq, linear_index)
  using HeapElement = std::pair<double, size_t>;
  std::priority_queue<HeapElement> max_heap;

  // Check if grid actually has poles
  const bool grid_has_north_pole = is_at_pole(m_lat_max);
  const bool grid_has_south_pole = is_at_pole(m_lat_min);

  // Special case: query at pole
  if (is_at_pole(query_lat) && (grid_has_north_pole || grid_has_south_pole)) {
    size_t pole_ilat = (query_lat > 0) ? m_nlat - 1 : 0;
    CoordinateType pole_lat = m_lats[pole_ilat];
    CoordinateType pole_dist = std::abs(query_lat - pole_lat);
    CoordinateType pole_dist_sq = pole_dist * pole_dist;

    // Add pole points (just one since they're all the same)
    size_t idx = get_linear_index(pole_ilat, 0);
    max_heap.push({pole_dist_sq, idx});

    // If we need more neighbors, expand search to adjacent latitudes
    if (k > 1) {
      // Use int for safe decrementing/incrementing
      if (pole_ilat == 0) {
        // South pole - search upward
        for (size_t ilat = 1; max_heap.size() < k && ilat < m_nlat; ++ilat) {
          for (size_t ilon = 0; ilon < m_nlon && max_heap.size() < k; ++ilon) {
            CoordinateType dist = compute_distance(
                query_lon, query_lat, m_lons[ilon], m_lats[ilat], m_metric);
            size_t index = get_linear_index(ilat, ilon);
            max_heap.push({dist * dist, index});
          }
        }
      } else {
        // North pole - search downward (use int to avoid underflow)
        for (int ilat = static_cast<int>(pole_ilat) - 1;
             max_heap.size() < k && ilat >= 0; --ilat) {
          for (size_t ilon = 0; ilon < m_nlon && max_heap.size() < k; ++ilon) {
            CoordinateType dist = compute_distance(
                query_lon, query_lat, m_lons[ilon], m_lats[ilat], m_metric);
            size_t index = get_linear_index(static_cast<size_t>(ilat), ilon);
            max_heap.push({dist * dist, index});
          }
        }
      }
    }
  } else {
    // Find starting position (nearest grid point)
    size_t ilat_start = find_nearest_index(m_lats, query_lat);
    size_t ilon_start = find_nearest_index(m_lons, query_lon);

    // Spiral search outward from starting position
    size_t search_radius = 0;
    const size_t max_radius = std::max(m_nlat, m_nlon);

    while (max_heap.size() < k && search_radius < max_radius) {
      // Search ring at current radius
      for (int dlat = -static_cast<int>(search_radius);
           dlat <= static_cast<int>(search_radius); ++dlat) {
        for (int dlon = -static_cast<int>(search_radius);
             dlon <= static_cast<int>(search_radius); ++dlon) {
          // Only process perimeter of ring (skip interior for efficiency)
          if (search_radius > 0 &&
              std::abs(dlat) < static_cast<int>(search_radius) &&
              std::abs(dlon) < static_cast<int>(search_radius)) {
            continue;
          }

          int ilat = static_cast<int>(ilat_start) + dlat;
          int ilon = static_cast<int>(ilon_start) + dlon;

          // Handle latitude bounds (no wraparound)
          if (ilat < 0 || ilat >= static_cast<int>(m_nlat))
            continue;

          // Handle longitude wraparound
          while (ilon < 0)
            ilon += m_nlon;
          while (ilon >= static_cast<int>(m_nlon))
            ilon -= m_nlon;

          CoordinateType grid_lon = m_lons[ilon];
          CoordinateType grid_lat = m_lats[ilat];
          CoordinateType dist = compute_distance(query_lon, query_lat, grid_lon,
                                                 grid_lat, m_metric);
          CoordinateType dist_sq = dist * dist;

          size_t idx = get_linear_index(ilat, ilon);

          // Maintain heap of k nearest neighbors
          if (max_heap.size() < k) {
            max_heap.push({dist_sq, idx});
          } else if (dist_sq < max_heap.top().first) {
            max_heap.pop();
            max_heap.push({dist_sq, idx});
          }
        }
      }
      search_radius++;
    }
  }

  // Extract results from heap (in reverse order - farthest first)
  std::vector<HeapElement> results;
  while (!max_heap.empty()) {
    results.push_back(max_heap.top());
    max_heap.pop();
  }

  // Reverse to get ascending order (nearest first)
  if (sorted) {
    std::reverse(results.begin(), results.end());
  }

  // Fill output arrays with found neighbors
  size_t n = std::min(k, results.size());
  for (size_t i = 0; i < n; ++i) {
    indices[i] = results[i].second;
    distances_sq[i] = results[i].first;
  }

  // Fill remaining slots if we didn't find k neighbors
  for (size_t i = n; i < k; ++i) {
    indices[i] = static_cast<size_t>(-1);
    distances_sq[i] = std::numeric_limits<double>::max();
  }
}

} // namespace moab

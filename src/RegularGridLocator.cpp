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
double RegularGridLocator::normalize_longitude(double lon) const {
    // Normalize to [m_lon_min, m_lon_min + 360) range
    while (lon < m_lon_min)
        lon += 360.0;
    while (lon >= m_lon_min + 360.0)
        lon -= 360.0;
    return lon;
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
double RegularGridLocator::haversine_distance(double lon1, double lat1,
                                              double lon2, double lat2) const {
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
double RegularGridLocator::euclidean_distance(double lon1, double lat1,
                                              double lon2, double lat2) const {
    PointType3D p1, p2;
    RLLtoXYZ_Deg(lon1, lat1, p1);
    RLLtoXYZ_Deg(lon2, lat2, p2);

    return std::sqrt((p2[0] - p1[0]) * (p2[0] - p1[0]) + (p2[1] - p1[1]) * (p2[1] - p1[1]) + (p2[2] - p1[2]) * (p2[2] - p1[2]));

    // double dlon = lon2 - lon1;
    // double dlat = lat2 - lat1;

    // // Handle longitude wraparound: choose shorter path
    // if (dlon > 180.0)
    //     dlon -= 360.0;
    // if (dlon < -180.0)
    //     dlon += 360.0;

    // return std::sqrt(dlon * dlon + dlat * dlat);
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
double RegularGridLocator::compute_distance(double lon1, double lat1,
                                            double lon2, double lat2) const {
    if (m_metric == HAVERSINE) {
        return haversine_distance(lon1, lat1, lon2, lat2);
    } else {
        return euclidean_distance(lon1, lat1, lon2, lat2);
    }
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
 * for a radius query. Handles longitude wraparound and pole special cases.
 *
 * Algorithm:
 * 1. Normalize query longitude
 * 2. Compute latitude bounds (simple clipping)
 * 3. Compute longitude bounds with wraparound detection
 * 4. Adjust longitude radius for latitude compression at poles
 *
 * @param query_lon Query longitude (degrees)
 * @param query_lat Query latitude (degrees)
 * @param radius Search radius (degrees)
 * @param ilat_min Output: minimum latitude index
 * @param ilat_max Output: maximum latitude index
 * @param ilon_min Output: minimum longitude index
 * @param ilon_max Output: maximum longitude index
 * @param wraps_around Output: true if longitude range wraps around
 */
void RegularGridLocator::get_search_bounds(double query_lon, double query_lat,
                                           double radius, size_t &ilat_min,
                                           size_t &ilat_max, size_t &ilon_min,
                                           size_t &ilon_max,
                                           bool &wraps_around) const {
    wraps_around = false;

    // Normalize query longitude to grid range
    query_lon = normalize_longitude(query_lon);

    // Latitude bounds (simple clipping, no wraparound)
    double lat_search_min = std::max(m_lat_min, query_lat - radius);
    double lat_search_max = std::min(m_lat_max, query_lat + radius);

    ilat_min = find_nearest_index(m_lats, lat_search_min);
    ilat_max = find_nearest_index(m_lats, lat_search_max);

    // Longitude bounds with wraparound handling
    // At high latitudes, longitude spacing becomes compressed due to
    // convergence Use a conservative estimate: radius / cos(lat) to account for
    // this
    double lat_rad = std::abs(query_lat) * DEG_TO_RAD;
    double lon_radius = (std::abs(query_lat) < 90.0)
                            ? radius / std::max(0.01, std::cos(lat_rad))
                            : 180.0;

    double lon_search_min = query_lon - lon_radius;
    double lon_search_max = query_lon + lon_radius;

    // Check for wraparound across the dateline (preserve periodicity)
    if (lon_search_min < m_lon_min || lon_search_max > m_lon_max) {
        wraps_around = true;
        // For wraparound case, we'll search entire longitude range
        ilon_min = 0;
        ilon_max = m_nlon - 1;
    } else {
        ilon_min = find_nearest_index(m_lons, lon_search_min);
        ilon_max = find_nearest_index(m_lons, lon_search_max);
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

    // Special case: if query is at pole, all longitude values at pole are
    // equidistant
    if (is_at_pole(query_lat)) {
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
        return matches.size();
    }

    // Get search bounds for efficient querying
    size_t ilat_min, ilat_max, ilon_min, ilon_max;
    bool wraps_around;
    get_search_bounds(query_lon, query_lat, radius, ilat_min, ilat_max,
                      ilon_min, ilon_max, wraps_around);

    // Search within computed bounds
    for (size_t ilat = ilat_min; ilat <= ilat_max; ++ilat) {
        CoordinateType grid_lat = m_lats[ilat];
        bool lat_at_pole = is_at_pole(grid_lat);

        if (wraps_around) {
            // Search all longitudes (wraparound case)
            for (size_t ilon = 0; ilon < m_nlon; ++ilon) {
                CoordinateType grid_lon = m_lons[ilon];
                CoordinateType dist =
                    compute_distance(query_lon, query_lat, grid_lon, grid_lat);

                if (dist <= radius) {
                    size_t idx = get_linear_index(ilat, ilon);
                    matches.push_back({idx, dist * dist});

                    // If this latitude is at pole, we only need one longitude
                    // value
                    if (lat_at_pole)
                        break;
                }
            }
        } else {
            // Search limited longitude range
            for (size_t ilon = ilon_min; ilon <= ilon_max; ++ilon) {
                CoordinateType grid_lon = m_lons[ilon];
                CoordinateType dist =
                    compute_distance(query_lon, query_lat, grid_lon, grid_lat);

                if (dist <= radius) {
                    size_t idx = get_linear_index(ilat, ilon);
                    matches.push_back({idx, dist * dist});

                    // If this latitude is at pole, we only need one longitude
                    // value
                    if (lat_at_pole)
                        break;
                }
            }
        }
    }

    // Sort results by distance (ascending order)
    if (sorted) {
        std::sort(
            matches.begin(), matches.end(),
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
    query_lon = normalize_longitude(query_lon);

    // Use a max-heap to keep track of k nearest neighbors
    // Pair: (distance_sq, linear_index)
    using HeapElement = std::pair<double, size_t>;
    std::priority_queue<HeapElement> max_heap;

    // Special case: query at pole
    if (is_at_pole(query_lat)) {
        size_t pole_ilat = (query_lat > 0) ? m_nlat - 1 : 0;
        CoordinateType pole_lat = m_lats[pole_ilat];
        CoordinateType pole_dist = std::abs(query_lat - pole_lat);
        CoordinateType pole_dist_sq = pole_dist * pole_dist;

        // Add pole points (just one since they're all the same)
        size_t idx = get_linear_index(pole_ilat, 0);
        max_heap.push({pole_dist_sq, idx});

        // If we need more neighbors, expand search to adjacent latitudes
        if (k > 1 && pole_ilat > 0) {
            for (size_t ilat = pole_ilat - 1;
                 max_heap.size() < k && ilat < m_nlat; --ilat) {
                for (size_t ilon = 0; ilon < m_nlon && max_heap.size() < k;
                     ++ilon) {
                    CoordinateType dist = compute_distance(
                        query_lon, query_lat, m_lons[ilon], m_lats[ilat]);
                    size_t index = get_linear_index(ilat, ilon);
                    max_heap.push({dist * dist, index});
                }
            }
        }
    } else {
        // Find starting position (nearest grid point)
        size_t ilat_start = find_nearest_index(m_lats, query_lat);
        size_t ilon_start = find_nearest_index(m_lons, query_lon);

        // Spiral search outward from starting position
        size_t radius = 0;
        const size_t max_radius = std::max(m_nlat, m_nlon);

        while (max_heap.size() < k && radius < max_radius) {
            // Search ring at current radius
            for (int dlat = -static_cast<int>(radius);
                 dlat <= static_cast<int>(radius); ++dlat) {
                for (int dlon = -static_cast<int>(radius);
                     dlon <= static_cast<int>(radius); ++dlon) {
                    // Only process perimeter of ring (skip interior for
                    // efficiency)
                    if (radius > 0 &&
                        std::abs(dlat) < static_cast<int>(radius) &&
                        std::abs(dlon) < static_cast<int>(radius)) {
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
                    CoordinateType dist = compute_distance(query_lon, query_lat,
                                                           grid_lon, grid_lat);
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
            radius++;
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

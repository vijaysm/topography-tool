#include "easylogging.hpp"

#include "RegularGridLocator.hpp"
#include <cmath>
#include <limits>
#include <iostream>

namespace moab {

RegularGridLocator::RegularGridLocator(const std::vector<double>& lats,
                                       const std::vector<double>& lons,
                                       DistanceMetric metric)
    : m_lats(lats), m_lons(lons), m_nlat(lats.size()), m_nlon(lons.size()), m_metric(metric)
{
    if (m_nlat == 0 || m_nlon == 0) {
        throw std::runtime_error("RegularGridLocator: Empty lat/lon arrays");
    }

    // Compute grid bounds
    m_lat_min = *std::min_element(m_lats.begin(), m_lats.end());
    m_lat_max = *std::max_element(m_lats.begin(), m_lats.end());
    m_lon_min = *std::min_element(m_lons.begin(), m_lons.end());
    m_lon_max = *std::max_element(m_lons.begin(), m_lons.end());

    // Compute average spacing (for regular grids this should be uniform)
    m_dlat = (m_nlat > 1) ? (m_lat_max - m_lat_min) / (m_nlat - 1) : 0.0;
    m_dlon = (m_nlon > 1) ? (m_lon_max - m_lon_min) / (m_nlon - 1) : 0.0;

    LOG(INFO) << "RegularGridLocator initialized: " << m_nlat << " x " << m_nlon
              << " grid, lat[" << m_lat_min << ", " << m_lat_max << "] lon["
              << m_lon_min << ", " << m_lon_max << "]";
    LOG(INFO) << "  Average spacing: dlat=" << m_dlat << " dlon=" << m_dlon;
    LOG(INFO) << "  Distance metric: " << (m_metric == HAVERSINE ? "Haversine" : "Euclidean L2");
}

double RegularGridLocator::normalize_longitude(double lon) const {
    // Normalize to [m_lon_min, m_lon_min + 360) range
    while (lon < m_lon_min) lon += 360.0;
    while (lon >= m_lon_min + 360.0) lon -= 360.0;
    return lon;
}

double RegularGridLocator::haversine_distance(double lon1, double lat1, double lon2, double lat2) const {
    // Convert to radians
    double lat1_rad = lat1 * DEG_TO_RAD;
    double lat2_rad = lat2 * DEG_TO_RAD;
    double dlon = (lon2 - lon1) * DEG_TO_RAD;
    double dlat = (lat2 - lat1) * DEG_TO_RAD;

    // Haversine formula
    double a = std::sin(dlat / 2.0) * std::sin(dlat / 2.0) +
               std::cos(lat1_rad) * std::cos(lat2_rad) *
               std::sin(dlon / 2.0) * std::sin(dlon / 2.0);
    double c = 2.0 * std::atan2(std::sqrt(a), std::sqrt(1.0 - a));

    // Return distance in degrees (angular distance)
    return c / DEG_TO_RAD;
}

double RegularGridLocator::euclidean_distance(double lon1, double lat1, double lon2, double lat2) const {
    double dlon = lon2 - lon1;
    double dlat = lat2 - lat1;

    // Handle longitude wraparound: choose shorter path
    if (dlon > 180.0) dlon -= 360.0;
    if (dlon < -180.0) dlon += 360.0;

    return std::sqrt(dlon * dlon + dlat * dlat);
}

double RegularGridLocator::compute_distance(double lon1, double lat1, double lon2, double lat2) const {
    if (m_metric == HAVERSINE) {
        return haversine_distance(lon1, lat1, lon2, lat2);
    } else {
        return euclidean_distance(lon1, lat1, lon2, lat2);
    }
}

size_t RegularGridLocator::find_nearest_index(const std::vector<double>& coords, double value) const {
    // Binary search for nearest value in sorted array
    auto it = std::lower_bound(coords.begin(), coords.end(), value);

    if (it == coords.begin()) return 0;
    if (it == coords.end()) return coords.size() - 1;

    // Check which neighbor is closer
    size_t idx = std::distance(coords.begin(), it);
    if (std::abs(coords[idx] - value) < std::abs(coords[idx - 1] - value)) {
        return idx;
    } else {
        return idx - 1;
    }
}

void RegularGridLocator::get_search_bounds(double query_lon, double query_lat, double radius,
                                           size_t& ilat_min, size_t& ilat_max,
                                           size_t& ilon_min, size_t& ilon_max,
                                           bool& wraps_around) const {
    wraps_around = false;

    // Normalize query longitude
    query_lon = normalize_longitude(query_lon);

    // Latitude bounds (simple clipping, no wraparound)
    double lat_search_min = std::max(m_lat_min, query_lat - radius);
    double lat_search_max = std::min(m_lat_max, query_lat + radius);

    ilat_min = find_nearest_index(m_lats, lat_search_min);
    ilat_max = find_nearest_index(m_lats, lat_search_max);

    // Longitude bounds (handle wraparound)
    // At high latitudes, longitude spacing becomes compressed
    // Use a conservative estimate: radius / cos(lat)
    double lat_rad = std::abs(query_lat) * DEG_TO_RAD;
    double lon_radius = (std::abs(query_lat) < 89.0) ? radius / std::max(0.01, std::cos(lat_rad)) : 180.0;

    double lon_search_min = query_lon - lon_radius;
    double lon_search_max = query_lon + lon_radius;

    // Check for wraparound
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

size_t RegularGridLocator::radiusSearch(const PointType3D& query_point,
                                       CoordinateType radius,
                                       std::vector<nanoflann::ResultItem<size_t, CoordinateType>>& matches) const {
    matches.clear();

    CoordinateType query_lon = query_point[0];
    CoordinateType query_lat = query_point[1];

    // Special case: if query is at pole, all longitude values at pole are equidistant
    if (is_at_pole(query_lat)) {
        // Find the latitude index closest to the pole
        size_t pole_ilat = (query_lat > 0) ? m_nlat - 1 : 0;
        CoordinateType pole_lat = m_lats[pole_ilat];

        // Check if pole is within radius
        CoordinateType pole_dist = std::abs(query_lat - pole_lat);
        if (pole_dist <= radius) {
            // All longitudes at pole are valid
            for (size_t ilon = 0; ilon < m_nlon; ++ilon) {
                size_t idx = get_linear_index(pole_ilat, ilon);
                matches.push_back({idx, pole_dist * pole_dist});
            }
        }
        return matches.size();
    }

    // Get search bounds
    size_t ilat_min, ilat_max, ilon_min, ilon_max;
    bool wraps_around;
    get_search_bounds(query_lon, query_lat, radius, ilat_min, ilat_max, ilon_min, ilon_max, wraps_around);

    // Search within bounds
    for (size_t ilat = ilat_min; ilat <= ilat_max; ++ilat) {
        CoordinateType grid_lat = m_lats[ilat];

        // Check if this latitude is at a pole
        bool lat_at_pole = is_at_pole(grid_lat);

        if (wraps_around) {
            // Search all longitudes
            for (size_t ilon = 0; ilon < m_nlon; ++ilon) {
                CoordinateType grid_lon = m_lons[ilon];
                CoordinateType dist = compute_distance(query_lon, query_lat, grid_lon, grid_lat);

                if (dist <= radius) {
                    size_t idx = get_linear_index(ilat, ilon);
                    matches.push_back({idx, dist * dist});

                    // If this latitude is at pole, we only need one longitude value
                    if (lat_at_pole) break;
                }
            }
        } else {
            // Search limited longitude range
            for (size_t ilon = ilon_min; ilon <= ilon_max; ++ilon) {
                CoordinateType grid_lon = m_lons[ilon];
                CoordinateType dist = compute_distance(query_lon, query_lat, grid_lon, grid_lat);

                if (dist <= radius) {
                    size_t idx = get_linear_index(ilat, ilon);
                    matches.push_back({idx, dist * dist});

                    // If this latitude is at pole, we only need one longitude value
                    if (lat_at_pole) break;
                }
            }
        }
    }

    // Sort by distance (optional, but nanoflann does this)
    std::sort(matches.begin(), matches.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    return matches.size();
}

void RegularGridLocator::knnSearch(const PointType3D& query_point,
                                  size_t k,
                                  size_t* indices,
                                  CoordinateType* distances_sq) const {
    CoordinateType query_lon = query_point[0];
    CoordinateType query_lat = query_point[1];

    // Normalize query longitude
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

        // If we need more neighbors, expand search
        if (k > 1 && pole_ilat > 0) {
            for (size_t ilat = pole_ilat - 1; max_heap.size() < k && ilat < m_nlat; --ilat) {
                for (size_t ilon = 0; ilon < m_nlon && max_heap.size() < k; ++ilon) {
                    CoordinateType dist = compute_distance(query_lon, query_lat, m_lons[ilon], m_lats[ilat]);
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
            for (int dlat = -static_cast<int>(radius); dlat <= static_cast<int>(radius); ++dlat) {
                for (int dlon = -static_cast<int>(radius); dlon <= static_cast<int>(radius); ++dlon) {
                    // Only process perimeter of ring (skip interior)
                    if (radius > 0 && std::abs(dlat) < static_cast<int>(radius) &&
                        std::abs(dlon) < static_cast<int>(radius)) {
                        continue;
                    }

                    int ilat = static_cast<int>(ilat_start) + dlat;
                    int ilon = static_cast<int>(ilon_start) + dlon;

                    // Handle latitude bounds (no wraparound)
                    if (ilat < 0 || ilat >= static_cast<int>(m_nlat)) continue;

                    // Handle longitude wraparound
                    while (ilon < 0) ilon += m_nlon;
                    while (ilon >= static_cast<int>(m_nlon)) ilon -= m_nlon;

                    CoordinateType grid_lon = m_lons[ilon];
                    CoordinateType grid_lat = m_lats[ilat];
                    CoordinateType dist = compute_distance(query_lon, query_lat, grid_lon, grid_lat);
                    CoordinateType dist_sq = dist * dist;

                    size_t idx = get_linear_index(ilat, ilon);

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

    // Extract results from heap (in reverse order)
    std::vector<HeapElement> results;
    while (!max_heap.empty()) {
        results.push_back(max_heap.top());
        max_heap.pop();
    }

    // Reverse to get ascending order
    std::reverse(results.begin(), results.end());

    // Fill output arrays
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

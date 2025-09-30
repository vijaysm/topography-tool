#ifndef REGULAR_GRID_LOCATOR_HPP
#define REGULAR_GRID_LOCATOR_HPP

#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <queue>
#include "nanoflann.hpp"
#include "ParallelPointCloudReader.hpp"

namespace moab {

/**
 * @brief Fast spatial query for regular lat/lon grids
 *
 * Exploits the structured nature of regular grids to provide O(1) spatial queries
 * without the overhead of building a KD-tree. Thread-safe for parallel queries.
 */
class RegularGridLocator {
public:
    enum DistanceMetric {
        HAVERSINE,      // Great circle distance on sphere (accurate)
        EUCLIDEAN_L2    // Euclidean distance in lat/lon space (fast approximation)
    };

    /**
     * @brief Construct locator from explicit lat/lon coordinate vectors
     *
     * @param lats Latitude values (degrees, typically -90 to 90)
     * @param lons Longitude values (degrees, typically 0 to 360 or -180 to 180)
     * @param metric Distance metric to use for queries
     *
     * Data is assumed to be organized as nlat*nlon with longitude changing fastest:
     * Index(ilat, ilon) = ilat * nlon + ilon
     */
    RegularGridLocator(const std::vector<double>& lats,
                       const std::vector<double>& lons,
                       DistanceMetric metric = HAVERSINE);

    /**
     * @brief Radius search - find all points within given radius
     *
     * @param query_point Query coordinates [lon, lat, z] in degrees (z ignored)
     * @param radius Search radius
     * @param matches Output vector of (index, distance_squared) pairs
     * @return Number of matches found
     */
    size_t radiusSearch(const ParallelPointCloudReader::PointType3D& query_point,
                       ParallelPointCloudReader::CoordinateType radius,
                       std::vector<nanoflann::ResultItem<size_t, ParallelPointCloudReader::CoordinateType>>& matches) const;

    /**
     * @brief K nearest neighbor search
     *
     * @param query_point Query coordinates [lon, lat, z] in degrees (z ignored)
     * @param k Number of neighbors to find
     * @param indices Output array of k indices
     * @param distances_sq Output array of k squared distances
     */
    void knnSearch(const ParallelPointCloudReader::PointType3D& query_point,
                  size_t k,
                  size_t* indices,
                  ParallelPointCloudReader::CoordinateType* distances_sq) const;

    /**
     * @brief Get total number of points in grid
     */
    size_t size() const { return m_nlat * m_nlon; }

    /**
     * @brief Get grid dimensions
     */
    void get_dimensions(size_t& nlat, size_t& nlon) const {
        nlat = m_nlat;
        nlon = m_nlon;
    }

private:
    // Grid data
    std::vector<double> m_lats;  // Latitude values
    std::vector<double> m_lons;  // Longitude values
    size_t m_nlat, m_nlon;

    // Grid bounds and spacing
    double m_lat_min, m_lat_max;
    double m_lon_min, m_lon_max;
    double m_dlat, m_dlon;  // Average spacing

    DistanceMetric m_metric;

    // Constants
    static constexpr double EARTH_RADIUS_KM = 6371.0;
    static constexpr double DEG_TO_RAD = M_PI / 180.0;
    static constexpr double POLE_TOLERANCE = 1e-6;  // Degrees from pole

    /**
     * @brief Compute distance between two points based on selected metric
     */
    double compute_distance(double lon1, double lat1, double lon2, double lat2) const;

    /**
     * @brief Compute Haversine (great circle) distance in degrees
     */
    double haversine_distance(double lon1, double lat1, double lon2, double lat2) const;

    /**
     * @brief Compute Euclidean L2 distance in lat/lon space
     */
    double euclidean_distance(double lon1, double lat1, double lon2, double lat2) const;

    /**
     * @brief Convert 2D grid indices to linear index
     */
    inline size_t get_linear_index(size_t ilat, size_t ilon) const {
        return ilat * m_nlon + ilon;
    }

    /**
     * @brief Convert linear index to 2D grid indices
     */
    inline void get_grid_indices(size_t linear_idx, size_t& ilat, size_t& ilon) const {
        ilat = linear_idx / m_nlon;
        ilon = linear_idx % m_nlon;
    }

    /**
     * @brief Normalize longitude to [m_lon_min, m_lon_min + 360) range
     */
    double normalize_longitude(double lon) const;

    /**
     * @brief Check if latitude is at or very close to a pole
     */
    bool is_at_pole(double lat) const {
        return std::abs(std::abs(lat) - 90.0) < POLE_TOLERANCE;
    }

    /**
     * @brief Find index bounds for radius search
     * Handles wraparound and pole special cases
     */
    void get_search_bounds(double query_lon, double query_lat, double radius,
                          size_t& ilat_min, size_t& ilat_max,
                          size_t& ilon_min, size_t& ilon_max,
                          bool& wraps_around) const;

    /**
     * @brief Find nearest grid index for a given coordinate value
     */
    size_t find_nearest_index(const std::vector<double>& coords, double value) const;
};

} // namespace moab

#endif // REGULAR_GRID_LOCATOR_HPP

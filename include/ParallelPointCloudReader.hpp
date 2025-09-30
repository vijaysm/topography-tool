#ifndef PARALLEL_POINT_CLOUD_READER_HPP
#define PARALLEL_POINT_CLOUD_READER_HPP

#include "moab/Core.hpp"
#include "moab/ParallelComm.hpp"
#include "moab/Range.hpp"
#include "moab/CartVect.hpp"
#ifdef MOAB_HAVE_PNETCDF
#include <pnetcdf>
#else
#error Need to build MOAB with Parallel-NetCDF for the example to work
#endif
#include <vector>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <memory>
#include <array>

namespace moab {

/**
 * @brief Efficient parallel NetCDF point cloud reader with MOAB mesh-based decomposition
 *
 * This class provides scalable reading of massive point cloud datasets from NetCDF files,
 * distributing points based on proximity to MOAB mesh elements with configurable buffer zones.
 */
class ParallelPointCloudReader {
public:
    // Type definitions for easier use - made configurable via typedefs
    typedef double CoordinateType;
    static constexpr int DIM = 2;
    static constexpr CoordinateType ReferenceTolerance = 1e-12;
    using PointType3D = std::array<CoordinateType, 3>;
    using PointType = std::array<CoordinateType, 2>;

    // Custom hash for PointType to detect duplicates
    struct PointHash {
        std::size_t operator()(const PointType& p) const {
            // Multiply by a large factor to preserve precision, then cast to integer
            long long lon_int = static_cast<long long>(p[0] * 1e7);
            long long lat_int = static_cast<long long>(p[1] * 1e7);
            // Combine the hashes of the integer representations
            return std::hash<long long>{}(lon_int) ^ (std::hash<long long>{}(lat_int) << 1);
        }
    };

    struct BoundingBox {
        PointType min_coords;  // [min_lon, min_lat]
        PointType max_coords;  // [max_lon, max_lat]

        BoundingBox() {
            min_coords.fill(std::numeric_limits<CoordinateType>::max());
            max_coords.fill(std::numeric_limits<CoordinateType>::lowest());
        }

        void expand(CoordinateType factor) {
            for (int i = 0; i < DIM; ++i) {
                CoordinateType range = max_coords[i] - min_coords[i];
                CoordinateType expansion = range * factor;
                min_coords[i] -= expansion;
                max_coords[i] += expansion;
            }
        }

        bool contains(const PointType& lonlat_point) const {
            for (int i = 0; i < DIM; ++i) {
                if (lonlat_point[i] < min_coords[i] || lonlat_point[i] > max_coords[i]) {
                    return false;
                }
            }
            return true;
        }
    };

    struct PointData {
        // For unstructured/irregular meshes: explicit coordinate storage
        std::vector<PointType> lonlat_coordinates;
        
        // For structured grids (USGS format): 1D lat/lon arrays
        std::vector<CoordinateType> longitudes;  // 1D longitude array
        std::vector<CoordinateType> latitudes;   // 1D latitude array
        bool is_structured_grid = false;         // Flag to indicate grid type
        
        // Scalar data
        std::unordered_map<std::string, std::vector<double>> d_scalar_variables;
        std::unordered_map<std::string, std::vector<int>> i_scalar_variables;

        /**
         * @brief Get total number of points
         * For structured grids: nlat * nlon
         * For unstructured: lonlat_coordinates.size()
         */
        size_t size() const { 
            if (is_structured_grid && !latitudes.empty() && !longitudes.empty()) {
                return latitudes.size() * longitudes.size();
            }
            return lonlat_coordinates.size(); 
        }

        /**
         * @brief Get longitude at global index
         * For structured grids: computes from 1D arrays (lon moves fastest)
         * For unstructured: reads from explicit storage
         */
        CoordinateType longitude(size_t global_idx) const { 
            if (is_structured_grid && !longitudes.empty()) {
                size_t nlon = longitudes.size();
                size_t ilon = global_idx % nlon;
                return longitudes[ilon];
            }
            return lonlat_coordinates[global_idx][0]; 
        }
        
        /**
         * @brief Get latitude at global index
         * For structured grids: computes from 1D arrays (lon moves fastest)
         * For unstructured: reads from explicit storage
         */
        CoordinateType latitude(size_t global_idx) const { 
            if (is_structured_grid && !latitudes.empty()) {
                size_t nlon = longitudes.size();
                size_t ilat = global_idx / nlon;
                return latitudes[ilat];
            }
            return lonlat_coordinates[global_idx][1]; 
        }
        
        /**
         * @brief Get lon/lat coordinate pair at global index
         * Convenience method that returns both coordinates
         */
        PointType get_lonlat(size_t global_idx) const {
            return {longitude(global_idx), latitude(global_idx)};
        }
        
        /**
         * @brief Get grid dimensions (only for structured grids)
         */
        void get_grid_dimensions(size_t& nlat, size_t& nlon) const {
            if (is_structured_grid) {
                nlat = latitudes.size();
                nlon = longitudes.size();
            } else {
                nlat = 0;
                nlon = 0;
            }
        }

        void reserve(size_t n) {
            if (!is_structured_grid) {
                lonlat_coordinates.reserve(n);
            }
        }

        void clear() {
            lonlat_coordinates.clear();
            longitudes.clear();
            latitudes.clear();
            d_scalar_variables.clear();
            i_scalar_variables.clear();
            is_structured_grid = false;
        }
    };

    struct ReadConfig {
        std::string netcdf_filename;
        std::vector<std::string> coord_var_names = {"x", "y"};
        std::vector<std::string> scalar_var_names;
        float buffer_factor = 0.05; // 5% buffer around bounding box
        bool use_collective_io = true;
        bool convert_lonlat_to_xyz = false;  // Convert geographic to Cartesian coordinates
        double sphere_radius = 1.0;  // Radius for coordinate conversion
        bool auto_detect_format = true;  // Automatically detect USGS vs other formats
        bool use_distributed_reading = false;  // Use distributed reading with MPI redistribution
        bool print_statistics = false; // Print details statistics about the reading and communication workflows
        bool verbose = false; // Print verbose output
    };

private:
    Interface* m_interface;
    ParallelComm* m_pcomm;
    EntityHandle m_mesh_set;
    ReadConfig m_config;

    // NetCDF file handle
    PnetCDF::NcmpiFile* m_ncfile;

    // NetCDF variables
    std::unordered_map<std::string, PnetCDF::NcmpiVar> m_vars;

    // Spatial decomposition
    BoundingBox m_local_bbox;
    std::vector<BoundingBox> m_all_bboxes;

    // Point cloud data
    size_t m_total_points;
    bool m_is_usgs_format;
    size_t nlats, nlons;
    MPI_Offset nlats_start, nlons_start;
    MPI_Offset nlats_count, nlons_count;
    std::string lon_var_name, lat_var_name, topo_var_name, fract_var_name;

    // For tracking unique points to avoid duplicates
    std::unordered_set<PointType, PointHash> m_unique_points;

    // Root-based distribution structures
    // struct PointBuffer {
    //     std::vector<PointType3D> coordinates;
    //     std::unordered_map<std::string, std::vector<double>> scalar_variables;

    //     void clear() {
    //         coordinates.clear();
    //         scalar_variables.clear();
    //     }

    //     size_t memory_size() const {
    //         size_t size = coordinates.size() * sizeof(PointType3D);
    //         for (const auto& var : scalar_variables) {
    //             size += var.second.size() * sizeof(double);
    //         }
    //         return size;
    //     }
    // };

public:
    ParallelPointCloudReader(Interface* interface, ParallelComm* pcomm, EntityHandle mesh_set);
    ~ParallelPointCloudReader();

    // Configuration
    ErrorCode configure(const ReadConfig& config);

    // Main reading interface
    ErrorCode read_points(PointData& local_points);
    const ReadConfig& get_config() const { return m_config; }

    // Utility functions
    size_t get_point_count() const { return m_total_points; }
    ErrorCode get_variable_info(std::vector<std::string>& var_names, std::vector<size_t>& var_sizes);
    ErrorCode gather_all_bounding_boxes(std::vector<BoundingBox>& all_bboxes);
    const std::vector<BoundingBox>& get_all_bounding_boxes() const { return m_all_bboxes; };
    bool is_usgs_format() const { return m_is_usgs_format; }

private:

    // Initialization
    ErrorCode initialize_netcdf();
    ErrorCode compute_mesh_bounding_boxes();

    // I/O operations
    ErrorCode read_chunk_parallel(size_t start_idx, size_t count, PointData& chunk_data);
    ErrorCode read_coordinates_chunk(size_t start_idx, size_t count,
                                   std::vector<PointType>& coords);

    template<typename T>
    ErrorCode read_scalar_variable_chunk(const std::string& var_name, size_t start_idx,
                                        size_t count, std::vector<T>& data);

    // Root-based distribution methods
    // ErrorCode read_and_distribute_root_based(PointData& local_points);
    // ErrorCode gather_all_bounding_boxes_on_root();
    // ErrorCode root_read_and_distribute_points(PointData& root_local_points);
    ErrorCode determine_point_owners(const PointType& point, std::vector<int>& owner_ranks);
    // ErrorCode send_buffered_data_to_ranks(std::vector<PointBuffer>& rank_buffers);
    // ErrorCode receive_distributed_data(PointData& local_points);

    // Distributed reading and redistribution methods
    ErrorCode read_and_redistribute_distributed(PointData& local_points);
    ErrorCode read_local_chunk_distributed(size_t start_idx, size_t count, PointData& chunk_data);
    // ErrorCode redistribute_points_by_ownership(PointData& initial_points, PointData& final_points);

    // Point distribution
    // ErrorCode distribute_points_by_proximity(PointData& all_points, PointData& local_points);
    std::vector<int> get_target_ranks(const PointType3D& point);

    // Memory management
    ErrorCode estimate_memory_requirements(size_t& estimated_memory);
    ErrorCode optimize_chunk_size();

    // Utility functions
    void cleanup_netcdf();
    ErrorCode check_netcdf_error(const std::exception& e, const std::string& operation);
    ErrorCode convert_lonlat_to_cartesian(std::vector<PointType3D>& coordinates);
    void convert_single_lonlat_to_cartesian(PointType3D& coord);
    // ErrorCode convert_cartesian_bbox_to_lonlat(BoundingBox& bbox);
    ErrorCode compute_lonlat_bounding_box_from_mesh(BoundingBox& lonlat_bbox);

    // ErrorCode find_spatial_index_range(const std::vector<PointType3D>& coords, double min_val, double max_val,
    //                                   size_t& start_idx, size_t& count);
    ErrorCode detect_netcdf_format();
    ErrorCode read_usgs_format(PointData& points);
    ErrorCode read_standard_format(PointData& points);
    ErrorCode read_standard_format_impl(PointData& points);
    ErrorCode read_points_chunked(PointData& points);
};



template<typename T>
inline moab::ErrorCode RLLtoXYZ_Deg(T lon_deg, T lat_deg, T* coordinates) {
    // Convert lon/lat (in degrees) to Cartesian coordinates on unit sphere
    T lon_rad = lon_deg * M_PI / 180.0;
    T lat_rad = lat_deg * M_PI / 180.0;

    T cos_lat = cos(lat_rad);
    coordinates[0] = cos_lat * cos(lon_rad);  // x
    coordinates[1] = cos_lat * sin(lon_rad);  // y
    coordinates[2] = sin(lat_rad);            // z
    return MB_SUCCESS;
}

template<typename T>
inline moab::ErrorCode RLLtoXYZ_Deg(T lon_deg, T lat_deg, std::array<T, 3>& coordinates) {
    return RLLtoXYZ_Deg(lon_deg, lat_deg, coordinates.data());
}

/// <summary>
///   Calculate latitude and longitude from normalized 3D Cartesian
///   coordinates, in degrees.
/// </summary>
template<typename SType, typename T>
inline moab::ErrorCode XYZtoRLL_Deg(const SType* coordinates, T & lon_deg, T & lat_deg) {
    // SType dMag2 = coordinates[0] * coordinates[0] + coordinates[1] * coordinates[1] + coordinates[2] * coordinates[2];

    // always project to the unit sphere
    SType dMag = std::sqrt(coordinates[0] * coordinates[0] + coordinates[1] * coordinates[1] + coordinates[2] * coordinates[2]);
    SType x = coordinates[0] / dMag;
    SType y = coordinates[1] / dMag;
    SType z = coordinates[2] / dMag;

    if (fabs(z) < 1.0 - ParallelPointCloudReader::ReferenceTolerance) {
        lon_deg = (atan2(y, x)) * 180.0 / M_PI;
        lat_deg = (asin(z)) * 180.0 / M_PI;

        if (lon_deg < 0.0) {
            lon_deg += 360.0;
        }
    } else if (z > 0.0) {
        lon_deg = 0.0;
        lat_deg = 90.0;
    } else {
        lon_deg = 0.0;
        lat_deg = -90.0;
    }
    return moab::MB_SUCCESS;
}



} // namespace moab

#endif // PARALLEL_POINT_CLOUD_READER_HPP

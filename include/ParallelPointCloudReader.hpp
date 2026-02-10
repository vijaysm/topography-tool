/**
 * @file ParallelPointCloudReader.hpp
 * @brief Efficient parallel NetCDF point cloud reader for MOAB
 *
 * This header defines a high-performance parallel NetCDF reader for point cloud
 * datasets, supporting both structured grids (USGS format) and unstructured
 * point clouds. The implementation uses Parallel-NetCDF for scalable I/O and
 * includes sophisticated spatial decomposition for efficient data distribution.
 *
 * Key Features:
 * - Parallel NetCDF I/O with PnetCDF for scalable reading
 * - Automatic format detection (USGS vs standard NetCDF)
 * - Spatial decomposition for efficient data distribution
 * - Support for both structured and unstructured point clouds
 * - Memory-efficient chunked reading
 * - Coordinate system transformations (lon/lat to Cartesian)
 * - Thread-safe operations for OpenMP parallelism
 *
 * Author: Vijay Mahadevan
 * Date: 2025-2026
 */

#ifndef PARALLEL_POINT_CLOUD_READER_HPP
#define PARALLEL_POINT_CLOUD_READER_HPP

#include "moab/CartVect.hpp"
#include "moab/Core.hpp"
#include "moab/Range.hpp"
#ifdef MOAB_HAVE_PNETCDF
#include <pnetcdf>
#else
#error Need to build MOAB with Parallel-NetCDF for NetCDF I/O
#endif

#include <array>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "MBDAUtilities.hpp"

namespace moab {

//===========================================================================
// Main ParallelPointCloudReader Class
//===========================================================================

/**
 * @brief Efficient NetCDF point cloud reader for MOAB meshes
 *
 * This class provides high-performance parallel reading of point cloud datasets
 * from NetCDF files. It supports both structured grids (USGS format) and
 * unstructured point clouds, with automatic format detection and spatial
 * decomposition for efficient data distribution across MPI ranks.
 *
 * Key Features:
 * - Parallel NetCDF I/O with PnetCDF for scalable reading
 * - Automatic format detection (USGS vs standard NetCDF)
 * - Spatial decomposition for efficient data distribution
 * - Memory-efficient chunked reading for large datasets
 * - Support for both structured and unstructured point clouds
 * - Coordinate system transformations (lon/lat to Cartesian)
 * - Thread-safe operations for OpenMP parallelism
 * - Caching and reuse of point cloud data
 *
 * Supported Formats:
 * - USGS Format: Structured lat/lon grids with htopo/landfract variables
 * - Standard Format: Unstructured point clouds with coordinate arrays
 *
 * Performance Characteristics:
 * - O(N/P) I/O complexity where N=points, P=MPI ranks
 * - Memory usage proportional to local spatial region
 * - Scalable to billions of points with appropriate MPI configuration
 *
 * Thread Safety:
 * - All public methods are thread-safe
 * - Internal data structures use proper synchronization
 * - OpenMP parallelism can be exploited in data processing sections
 */
class ParallelPointCloudReader {
public:
  //=======================================================================
  // Type Definitions and Data Structures
  //=======================================================================

  /// Dimension constant for 2D coordinate systems
  static constexpr int DIM = 2;

  /**
   * @brief Custom hash function for PointType to detect duplicates
   *
   * Uses integer quantization to handle floating-point precision issues
   * when detecting duplicate points in unstructured point clouds.
   */
  struct PointHash {
    std::size_t operator()(const PointType &p) const {
      // Quantize to 7 decimal places (~1cm precision)
      long long lon_int = static_cast<long long>(p[0] * 1e7);
      long long lat_int = static_cast<long long>(p[1] * 1e7);
      // Combine hashes using bit manipulation
      return std::hash<long long>{}(lon_int) ^
             (std::hash<long long>{}(lat_int) << 1);
    }
  };

  /**
   * @brief Axis-aligned bounding box for spatial queries
   *
   * Used for spatial decomposition and determining point ownership
   * across MPI ranks. Supports expansion for overlapping regions.
   */
  struct BoundingBox {
    PointType min_coords; // [min_lon, min_lat]
    PointType max_coords; // [max_lon, max_lat]

    BoundingBox() {
      min_coords.fill(std::numeric_limits<CoordinateType>::max());
      max_coords.fill(std::numeric_limits<CoordinateType>::lowest());
    }

    /**
     * @brief Expand bounding box by a factor
     *
     * Expands the bounding box symmetrically by the specified factor
     * to create overlapping regions for robust spatial queries.
     *
     * @param factor Expansion factor (0.1 = 10% expansion)
     */
    void expand(CoordinateType factor) {
      for (int i = 0; i < DIM; ++i) {
        CoordinateType range = max_coords[i] - min_coords[i];
        CoordinateType expansion = range * factor;
        min_coords[i] -= expansion;
        max_coords[i] += expansion;
      }
    }

    /**
     * @brief Check if point is within bounding box
     *
     * @param lonlat_point Point to test [lon, lat]
     * @return true if point is inside bounding box
     */
    bool contains(const PointType &lonlat_point) const {
      for (int i = 0; i < DIM; ++i) {
        if (lonlat_point[i] < min_coords[i] ||
            lonlat_point[i] > max_coords[i]) {
          return false;
        }
      }
      return true;
    }
  };

  /**
   * @brief Container for point cloud data with dual storage modes
   *
   * This structure supports both structured grids (USGS format) and
   * unstructured point clouds, providing a unified interface for different
   * data layouts. The storage mode is automatically detected and optimized
   * for the specific format.
   *
   * Storage Modes:
   * - Structured Grid: 1D lat/lon arrays + implicit indexing
   * - Unstructured: Explicit coordinate arrays
   *
   * Memory Layout:
   * - Structured: lat[ilat], lon[ilon] with Index(ilat,ilon) = ilat*nlon +
   * ilon
   * - Unstructured: coords[idx] = [lon, lat] for each point
   */
  struct PointData {
    //===================================================================
    // Coordinate Storage
    //===================================================================

    /// Unstructured point clouds: explicit [lon, lat] for each point
    std::vector<PointType> lonlat_coordinates;

    /// Structured grids: 1D coordinate arrays (USGS format)
    std::vector<CoordinateType> longitudes; // 1D longitude array
    std::vector<CoordinateType> latitudes;  // 1D latitude array

    //===================================================================
    // Scalar Data Storage
    //===================================================================

    /// Double precision scalar variables (e.g., temperature, elevation)
    std::unordered_map<std::string, std::vector<double>> d_scalar_variables;

    /// Integer scalar variables (e.g., masks, classifications)
    std::unordered_map<std::string, std::vector<int>> i_scalar_variables;

    /// Area data for each point (used for weighted operations)
    std::vector<double> areas;

    //===================================================================
    // Format Detection
    //===================================================================

    /// True if data represents a structured grid
    bool is_structured_grid = false;

    //===================================================================
    // Access Methods
    //===================================================================

    /**
     * @brief Get total number of points
     *
     * For structured grids: nlat * nlon
     * For unstructured: lonlat_coordinates.size()
     *
     * @return Total point count
     */
    size_t size() const {
      if (is_structured_grid && !latitudes.empty() && !longitudes.empty()) {
        return latitudes.size() * longitudes.size();
      }
      return lonlat_coordinates.size();
    }

    /**
     * @brief Check if point cloud is empty
     *
     * @return true if no points are stored
     */
    bool empty() const {
      if (is_structured_grid) {
        return latitudes.empty() && longitudes.empty();
      }
      return lonlat_coordinates.empty();
    }

    /**
     * @brief Get longitude at global index
     *
     * For structured grids: computes from 1D arrays (longitude moves
     * fastest) For unstructured: reads from explicit storage
     *
     * @param global_idx Global point index
     * @return Longitude in degrees
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
     *
     * For structured grids: computes from 1D arrays (longitude moves
     * fastest) For unstructured: reads from explicit storage
     *
     * @param global_idx Global point index
     * @return Latitude in degrees
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
     *
     * Convenience method that returns both coordinates as a pair.
     *
     * @param global_idx Global point index
     * @return Coordinate pair [longitude, latitude]
     */
    PointType get_lonlat(size_t global_idx) const {
      return {longitude(global_idx), latitude(global_idx)};
    }

    /**
     * @brief Get grid dimensions (structured grids only)
     *
     * @param nlat Output: number of latitude points
     * @param nlon Output: number of longitude points
     */
    void get_grid_dimensions(size_t &nlat, size_t &nlon) const {
      if (is_structured_grid) {
        nlat = latitudes.size();
        nlon = longitudes.size();
      } else {
        nlat = 0;
        nlon = 0;
      }
    }

    //===================================================================
    // Memory Management
    //===================================================================

    /**
     * @brief Reserve memory for specified number of points
     *
     * Only affects unstructured storage mode.
     *
     * @param n Number of points to reserve space for
     */
    void reserve(size_t n) {
      if (!is_structured_grid) {
        lonlat_coordinates.reserve(n);
      }
    }

    /**
     * @brief Clear all stored data
     */
    void clear() {
      lonlat_coordinates.clear();
      longitudes.clear();
      latitudes.clear();
      d_scalar_variables.clear();
      i_scalar_variables.clear();
      areas.clear();
      is_structured_grid = false;
    }
  };

  // struct PointCloudMeshView {

  //     explicit PointCloudMeshView(const PointData& refpoints, double
  //     default_area) : points(refpoints), area(default_area) {

  //     }

  //     ErrorCode cartesian_coords(size_t index, PointType3D& point3d) const
  //     {
  //         auto lon = points.longitude( index );
  //         auto lat = points.latitude( index );
  //         MB_CHK_SET_ERR( RLLtoXYZ_Deg( lon, lat, point3d ),
  //                         "Failed to convert lon/lat to Cartesian" );
  //         return MB_SUCCESS;
  //     }

  //     size_t size() const { return points.size(); }
  //     bool empty() const { return points.empty(); }

  //     const PointData& points;
  //     const double area;
  // };

  //=======================================================================
  // Configuration Structure
  //=======================================================================

  /**
   * @brief Configuration parameters for point cloud reading
   *
   * This structure contains all configurable parameters for the
   * ParallelPointCloudReader, including file paths, variable names,
   * and processing options.
   */
  struct ReadConfig {
    /// Input NetCDF filename
    std::string netcdf_filename;

    /// Coordinate variable names (default: ["lon", "lat"])
    std::vector<std::string> coord_var_names = {"lon", "lat"};

    /// Scalar variable names to read from the file
    std::vector<std::string> scalar_var_names;

    /// Fields to compute squares for (e.g., velocity components)
    std::vector<std::string> square_field_names;

    /// Convert geographic coordinates to 3D Cartesian
    bool convert_lonlat_to_xyz = false;

    /// Sphere radius for coordinate conversion (default: 1.0)
    double sphere_radius = 1.0;

    /// Print detailed statistics about the reading process
    bool print_statistics = false;

    /// Enable verbose logging output
    bool verbose = false;

    /// Keep point cloud data cached for reuse
    bool retain_point_cloud = true;
  };

  //=======================================================================
  // Public Interface
  //=======================================================================

  /**
   * @brief Constructor
   *
   * @param interface MOAB interface instance
   * @param mesh_set Mesh set containing target elements
   */
  ParallelPointCloudReader(Interface *interface, EntityHandle mesh_set);

  /**
   * @brief Destructor - cleans up NetCDF resources
   */
  ~ParallelPointCloudReader();

  /**
   * @brief Configure the reader with user parameters
   *
   * @param config Configuration parameters
   * @return MB_SUCCESS on success, error code otherwise
   */
  ErrorCode configure(const ReadConfig &config);

  /**
   * @brief Main reading interface - loads point cloud data
   *
   * This is the primary method for reading point cloud data from NetCDF
   * files. It handles format detection, parallel I/O, and data distribution.
   *
   * @param points Output container for point cloud data
   * @return MB_SUCCESS on success, error code otherwise
   */
  ErrorCode read_points(PointData &points);

  /**
   * @brief Get current configuration
   *
   * @return Current ReadConfig instance
   */
  const ReadConfig &get_config() const { return m_config; }

  /**
   * @brief Check if cached point cloud data is available
   *
   * @return true if cached data exists
   */
  bool has_cached_point_cloud() const { return m_have_cached_points; }

  /**
   * @brief Get cached point cloud data
   *
   * @return Reference to cached PointData
   */
  const PointData &cached_point_cloud() const { return m_cached_points; }

  /**
   * @brief Build MOAB mesh from point cloud data
   *
   * Creates MOAB vertices from point cloud coordinates and assigns
   * scalar data as MOAB tags.
   *
   * @param points Input point cloud data
   * @param mb MOAB interface instance
   * @param mesh_set Target mesh set
   * @param entities Output vector of created entity handles
   * @param default_area Default area for each point
   * @return MB_SUCCESS on success, error code otherwise
   */
  ErrorCode build_mesh_from_point_cloud(const PointData &points, Interface *mb,
                                        EntityHandle mesh_set,
                                        std::vector<EntityHandle> &entities,
                                        double default_area) const;

  //=======================================================================
  // Utility Methods
  //=======================================================================

  /**
   * @brief Get total point count in the dataset
   *
   * @return Total number of points (across all MPI ranks)
   */
  size_t get_point_count() const { return m_total_points; }

  /**
   * @brief Get information about available variables
   *
   * @param var_names Output: list of variable names
   * @param var_sizes Output: list of variable sizes
   * @return MB_SUCCESS on success, error code otherwise
   */
  ErrorCode get_variable_info(std::vector<std::string> &var_names,
                              std::vector<size_t> &var_sizes);

  /**
   * @brief Check if the loaded file is USGS format
   *
   * @return true if USGS format, false otherwise
   */
  bool is_usgs_format() const { return m_is_usgs_format; }

private:
  //=======================================================================
  // Member Variables
  //=======================================================================

  /// MOAB interface instance
  Interface *m_interface;

  /// Mesh set containing target elements
  EntityHandle m_mesh_set;

  /// Configuration parameters
  ReadConfig m_config;

  /// NetCDF file handle for parallel I/O
  PnetCDF::NcmpiFile *m_ncfile;

  /// NetCDF variable handles for efficient access
  std::unordered_map<std::string, PnetCDF::NcmpiVar> m_vars;

  /// Cached point cloud data for reuse
  PointData m_cached_points;
  bool m_have_cached_points = false;

  /// Dataset metadata
  size_t m_total_points; // Total points in dataset
  bool m_is_usgs_format; // True for USGS structured format

  /// USGS format specific metadata
  size_t nlats, nlons;             // Grid dimensions
  size_t nlats_start, nlons_start; // Local decomposition start indices
  size_t nlats_count, nlons_count; // Local decomposition counts
  std::string lon_var_name, lat_var_name, topo_var_name, fract_var_name;

  /// For tracking unique points to avoid duplicates in unstructured data
  std::unordered_set<PointType, PointHash> m_unique_points;

  //=======================================================================
  // Initialization Methods
  //=======================================================================

  /**
   * @brief Initialize NetCDF file for parallel reading
   *
   * Opens the NetCDF file with PnetCDF and performs format detection.
   *
   * @return MB_SUCCESS on success, error code otherwise
   */
  ErrorCode initialize_netcdf();

  /**
   * @brief Compute bounding boxes for spatial decomposition
   *
   * Analyzes the mesh to determine spatial regions for each MPI rank.
   *
   * @return MB_SUCCESS on success, error code otherwise
   */
  ErrorCode compute_mesh_bounding_boxes();

  //=======================================================================
  // I/O Operations
  //=======================================================================

  /**
   * @brief Read a chunk of data in parallel
   *
   * Reads a specified range of points using parallel NetCDF I/O.
   *
   * @param start_idx Starting index
   * @param count Number of points to read
   * @param chunk_data Output container for chunk data
   * @return MB_SUCCESS on success, error code otherwise
   */
  ErrorCode read_chunk_parallel(size_t start_idx, size_t count,
                                PointData &chunk_data);

  /**
   * @brief Read coordinate data chunk
   *
   * @param start_idx Starting index
   * @param count Number of points
   * @param coords Output coordinate array
   * @return MB_SUCCESS on success, error code otherwise
   */
  ErrorCode read_coordinates_chunk(size_t start_idx, size_t count,
                                   std::vector<PointType> &coords);

  /**
   * @brief Template method to read scalar variable chunk
   *
   * @tparam var_type Data type (double or int)
   * @param var_name Variable name
   * @param start_idx Starting index
   * @param count Number of values
   * @param data Output data array
   * @return MB_SUCCESS on success, error code otherwise
   */
  template <typename T>
  ErrorCode read_scalar_variable_chunk(const std::string &var_name,
                                       size_t start_idx, size_t count,
                                       std::vector<T> &data);

  //=======================================================================
  // Data Reading Methods
  //=======================================================================

  /**
   * @brief Read all data using appropriate strategy
   *
   * @param points Output point cloud data
   * @return MB_SUCCESS on success, error code otherwise
   */
  ErrorCode read_all_data(PointData &points);

  /**
   * @brief Read local chunk with distributed processing
   *
   * @param start_idx Starting index
   * @param count Number of points
   * @param chunk_data Output chunk data
   * @return MB_SUCCESS on success, error code otherwise
   */
  ErrorCode read_local_chunk_distributed(size_t start_idx, size_t count,
                                         PointData &chunk_data);

  //=======================================================================
  // Memory Management
  //=======================================================================

  /**
   * @brief Estimate memory requirements for reading
   *
   * @param estimated_memory Output: estimated memory in bytes
   * @return MB_SUCCESS on success, error code otherwise
   */
  ErrorCode estimate_memory_requirements(size_t &estimated_memory);

  /**
   * @brief Optimize chunk size for memory efficiency
   *
   * @return MB_SUCCESS on success, error code otherwise
   */
  ErrorCode optimize_chunk_size();

  //=======================================================================
  // Utility Functions
  //=======================================================================

  /**
   * @brief Clean up NetCDF resources
   */
  void cleanup_netcdf();

  /**
   * @brief Handle NetCDF errors with proper logging
   *
   * @param e Exception that occurred
   * @param operation Description of the operation
   * @return MB_FAILURE
   */
  ErrorCode check_netcdf_error(const std::exception &e,
                               const std::string &operation);

  /**
   * @brief Convert lon/lat coordinates to Cartesian
   *
   * @param coordinates Input/output coordinates
   * @return MB_SUCCESS on success, error code otherwise
   */
  ErrorCode convert_lonlat_to_cartesian(std::vector<PointType3D> &coordinates);

  /**
   * @brief Convert single lon/lat coordinate to Cartesian
   *
   * @param coord Input/output coordinate
   */
  void convert_single_lonlat_to_cartesian(PointType3D &coord);

  /**
   * @brief Compute lon/lat bounding box from mesh
   *
   * @param lonlat_bbox Output bounding box
   * @return MB_SUCCESS on success, error code otherwise
   */
  ErrorCode compute_lonlat_bounding_box_from_mesh(BoundingBox &lonlat_bbox);

  //=======================================================================
  // Format Detection and Reading
  //=======================================================================

  /**
   * @brief Detect NetCDF file format (USGS vs standard)
   *
   * @return MB_SUCCESS on success, error code otherwise
   */
  ErrorCode detect_netcdf_format();

  /**
   * @brief Read USGS format data (structured grid)
   *
   * @param points Output point cloud data
   * @return MB_SUCCESS on success, error code otherwise
   */
  ErrorCode read_usgs_format(PointData &points);

  /**
   * @brief Read standard format data (unstructured)
   *
   * @param points Output point cloud data
   * @return MB_SUCCESS on success, error code otherwise
   */
  ErrorCode read_standard_format(PointData &points);

  /**
   * @brief Implementation of standard format reading
   *
   * @param points Output point cloud data
   * @return MB_SUCCESS on success, error code otherwise
   */
  ErrorCode read_standard_format_impl(PointData &points);

  /**
   * @brief Read data using chunked approach
   *
   * @param points Output point cloud data
   * @return MB_SUCCESS on success, error code otherwise
   */
  ErrorCode read_points_chunked(PointData &points);
};

} // namespace moab

#endif // PARALLEL_POINT_CLOUD_READER_HPP

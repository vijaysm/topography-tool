/**
 * @file ScalarRemapper.cpp
 * @brief Implementation of scalar data remapping from point clouds to mesh
 * elements
 *
 * This file implements multiple algorithms for mapping scalar data from NetCDF
 * point clouds to H5M mesh element centroids, supporting both regular and
 * spectral element meshes.
 *
 * Supported Algorithms:
 * 1. Nearest Neighbor - Fast, simple mapping to closest point
 * 2. Disk-Averaged Projection - Area-weighted averaging within search radius
 * 3. Spectral Element Projection - High-order GLL quadrature projection
 *
 * Features:
 * - OpenMP parallelization for multicore performance
 * - KD-tree and RegularGridLocator spatial indexing
 * - Support for both USGS and standard NetCDF formats
 * - Automatic handling of integer and double scalar fields
 * - Robust error handling and validation
 *
 * Author: Vijay Mahadevan
 * Date: 2025-2026
 */

#include "ScalarRemapper.hpp"
#include "GaussLobattoQuadrature.hpp"
#include "easylogging.hpp"
#include "moab/CartVect.hpp"
#include "moab/IntxMesh/IntxUtils.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <set>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace moab {

//===========================================================================
// ScalarRemapper Base Class Implementation
//===========================================================================

/**
 * @brief Constructor for ScalarRemapper base class
 * @param interface MOAB interface instance
 * @param mesh_set Mesh set containing target elements
 */
ScalarRemapper::ScalarRemapper(Interface *interface, EntityHandle mesh_set)
    : m_interface(interface), m_mesh_set(mesh_set), m_target_is_spectral(false),
      m_self_remapping(false), m_user_search_area(0.0), m_kdtree_built(false),
      m_grid_locator_built(false) {
  // Initialize member variables to default values
}

/**
 * @brief Configure the remapper with user parameters
 * @param config Configuration parameters
 * @return MB_SUCCESS on success, error code otherwise
 */
ErrorCode ScalarRemapper::configure(const RemapConfig &config) {
  m_config = config;
  m_self_remapping = config.reuse_source_mesh;

  if (m_self_remapping) {
    // For self-remapping, target is the same as source
    m_target_is_spectral = false;
    m_user_search_area = config.user_search_area;
  } else {
    // Extract mesh element centroids for target mesh
    MB_CHK_ERR(extract_mesh_centroids());

    LOG(INFO) << "Configured scalar remapper with " << m_mesh_data.size()
              << " mesh elements";
  }

  // Log the variables that will be remapped
  LOG(INFO) << "Variables to remap: ";
  for (const auto &var : m_config.scalar_var_names) {
    LOG(INFO) << "\t" << var;
  }

  return MB_SUCCESS;
}

/**
 * @brief Main entry point for scalar remapping
 *
 * This method orchestrates the entire remapping process:
 * 1. Initializes output scalar fields
 * 2. Calls the appropriate remapping algorithm
 * 3. Validates results and prints statistics
 *
 * @param point_data Input point cloud data with scalar fields
 * @return MB_SUCCESS on success, error code otherwise
 */
ErrorCode ScalarRemapper::remap_scalars(
    const ParallelPointCloudReader::PointData &point_data) {
  LOG(INFO) << "";
  LOG(INFO) << "Starting scalar remapping with " << point_data.size()
            << " point cloud points";

  auto start_time = std::chrono::high_resolution_clock::now();
  size_t target_size =
      m_self_remapping ? point_data.size() : m_mesh_data.size();

  // Initialize output scalar fields for all configured variables
  for (const auto &var_name : m_config.scalar_var_names) {
    if (m_config.is_usgs_format)
      m_mesh_data.i_scalar_fields[var_name].resize(target_size, 0);
    else
      m_mesh_data.d_scalar_fields[var_name].resize(target_size, 0.0);
  }

  // Perform the actual remapping using the appropriate algorithm
  if (m_self_remapping) {
    MB_CHK_ERR(perform_self_remapping(point_data));
  } else {
    MB_CHK_ERR(perform_remapping(point_data));
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);

  LOG(INFO) << "Remapping completed in " << duration.count() / 1000.0
            << " seconds";

  // Validate results and provide statistics
  MB_CHK_ERR(validate_remapping_results());
  print_remapping_statistics();

  return MB_SUCCESS;
}

/**
 * @brief Perform self-remapping for field smoothing
 *
 * This method is used when the source mesh is also the target mesh,
 * typically for smoothing operations on the same point cloud.
 *
 * @param point_data Input point cloud data
 * @return MB_SUCCESS on success, error code otherwise
 */
ErrorCode ScalarRemapper::perform_self_remapping(
    const ParallelPointCloudReader::PointData &point_data) {
  if (point_data.size() == 0) {
    LOG(INFO) << "No point cloud data available for remapping";
    return MB_SUCCESS;
  }

  // Build spatial index (KD-tree or RegularGridLocator) for efficient queries
  if (m_config.is_usgs_format && !m_config.use_kd_tree) {
    if (!m_grid_locator_built) {
      MB_CHK_ERR(build_grid_locator(point_data));
    }
  } else {
    if (!m_kdtree_built) {
      MB_CHK_ERR(this->build_kdtree(point_data));
    }
  }
  LOG(INFO) << "";

  // Perform disk-area averaged smoothing for each point
  assert(!m_target_is_spectral);
  LOG(INFO) << "Starting PC disk-area averaged self-projection with "
            << point_data.size() << " point cloud points";
  MB_CHK_ERR(
      smoothen_field_constant_area_averaging(point_data, m_user_search_area));

  return MB_SUCCESS;
}

/**
 * @brief Perform field smoothing using constant area averaging
 *
 * This algorithm smooths scalar fields by averaging values within a specified
 * search radius around each point. The search radius is derived from the
 * constant area parameter: radius = sqrt(area / (4π))
 *
 * Algorithm Steps:
 * 1. For each point, compute search radius from constant area
 * 2. Find all neighboring points within search radius using KD-tree
 * 3. If no neighbors found, fallback to nearest neighbor
 * 4. Compute simple average of all scalar values from neighbors
 * 5. Store averaged values back to the mesh data
 *
 * @param point_data Input point cloud data with scalar fields
 * @param constant_area Search area for finding neighbors (in radians^2)
 * @return MB_SUCCESS on success, error code otherwise
 */
moab::ErrorCode ScalarRemapper::smoothen_field_constant_area_averaging(
    const ParallelPointCloudReader::PointData &point_data,
    double constant_area) {

  LOG(INFO) << "Performing topography smoothing with area averaging using "
            << point_data.size() << " point cloud points";

  // Parallel processing of points using OpenMP
  std::vector<moab::ErrorCode> element_errors(point_data.size(), MB_SUCCESS);
  std::cout.precision(10);

  // Compute search radius from constant area:
  // A = (1.0/α²) × πr² → r = α × √(A/π)
  const double search_radius =
      m_config.user_alpha * std::sqrt(constant_area / M_PI);

  VLOG(2) << "Self smoothing area = " << constant_area
          << ", Search radius = " << search_radius;

  // Main parallel loop over all points
#pragma omp parallel for schedule(dynamic, 1)                                  \
    shared(m_kdtree, element_errors, point_data, m_mesh_data, m_config,        \
               m_interface, search_radius)
  for (size_t elem_idx = 0; elem_idx < point_data.size(); ++elem_idx) {
    // Progress reporting
    if ((elem_idx * 100) % point_data.size() == 0) {
#pragma omp critical
      LOG(INFO) << "Processing element " + std::to_string(elem_idx) + " of " +
                       std::to_string(point_data.size());
    }

    // Convert lon/lat to 3D Cartesian for KD-tree search
    PointType3D gll_point;
    RLLtoXYZ_Deg(point_data.longitude(elem_idx), point_data.latitude(elem_idx),
                 gll_point);

    // Thread-local storage for KD-tree search results to avoid allocations
    thread_local std::vector<size_t> neighbor_indices;
    thread_local std::vector<CoordinateType> distances_sq;
    thread_local std::vector<nanoflann::ResultItem<size_t, CoordinateType>>
        matches;

    // Find neighbors within search radius
    auto nearest_points =
        find_nearest_point(gll_point, point_data, nullptr, &search_radius);

    // Check if radius search found valid results
    bool has_valid_results = !nearest_points.empty() &&
                             nearest_points[0].first != static_cast<size_t>(-1);

    if (!has_valid_results) {
      // Fallback to nearest neighbor without radius constraint
      const size_t single_neighbor = 1;
      const CoordinateType no_radius = 0.0; // 0.0 means no radius constraint
      nearest_points = find_nearest_point(gll_point, point_data,
                                          &single_neighbor, &no_radius);
    }

    // Skip if no points found even after fallback
    if (nearest_points.empty()) {
      if (elem_idx < 1) {
#pragma omp critical
        LOG(WARNING) << "WARNING: Element " << elem_idx
                     << ") has ZERO neighbors even after fallback!";
        element_errors[elem_idx] = MB_FAILURE;
      }
      continue;
    }

    // Process each scalar variable with simple averaging
    for (const auto &var_name : m_config.scalar_var_names) {
      double weighted_sum = 0.0;
      bool is_double = false;

      // Handle double precision variables
      auto var_it = point_data.d_scalar_variables.find(var_name);
      if (!m_config.is_usgs_format &&
          var_it != point_data.d_scalar_variables.end()) {
        const auto &values = var_it->second;

        // Simple average: sum all neighbor values
        for (size_t k = 0; k < nearest_points.size(); ++k) {
          size_t pt_idx = nearest_points[k].first;
          if (pt_idx < values.size()) {
            weighted_sum += values[pt_idx];
          }
        }
        is_double = true;
      } else {
        // Handle integer variables
        auto ivar_it = point_data.i_scalar_variables.find(var_name);
        if (ivar_it != point_data.i_scalar_variables.end()) {
          const auto &values = ivar_it->second;

          // Simple average: sum all neighbor values
          for (size_t k = 0; k < nearest_points.size(); ++k) {
            size_t pt_idx = nearest_points[k].first;
            if (pt_idx < values.size()) {
              weighted_sum += static_cast<double>(values[pt_idx]);
            }
          }
        }
      }

      // Store averaged value (simple mean, not weighted)
      if (is_double)
        m_mesh_data.d_scalar_fields[var_name][elem_idx] =
            weighted_sum / nearest_points.size();
      else
        m_mesh_data.i_scalar_fields[var_name][elem_idx] =
            static_cast<int>(weighted_sum / nearest_points.size());
    }

  } // End of parallel region

  // Error reporting and statistics
  size_t error_count = 0;
  for (size_t i = 0; i < element_errors.size(); ++i) {
    if (element_errors[i] != MB_SUCCESS) {
      error_count++;
    }
  }

  if (error_count > 0) {
    LOG(ERROR) << "Warning: " << error_count
               << " elements failed processing out of " << point_data.size()
               << " total elements";
  }

  // Debug output: show sample values for verification
  for (const auto &var_name : m_config.scalar_var_names) {
    std::array<double, 5> values;
    for (size_t i = 0; i < std::min(size_t(5), point_data.size()); ++i) {
      auto d_it = m_mesh_data.d_scalar_fields.find(var_name);
      auto i_it = m_mesh_data.i_scalar_fields.find(var_name);
      if (d_it != m_mesh_data.d_scalar_fields.end()) {
        values[i] = d_it->second[i];
      } else if (i_it != m_mesh_data.i_scalar_fields.end()) {
        values[i] = i_it->second[i];
      }
    }
    VLOG(2) << "Sample values for " << var_name << ": " << values;
  }

  return MB_SUCCESS;
}

//===========================================================================
// Mesh Processing Methods
//===========================================================================

/**
 * @brief Extract mesh element centroids for remapping targets
 *
 * This method processes the target mesh to extract element centroids:
 * 1. Gets all 2D elements (quadrilaterals) or falls back to 0D points
 * 2. Determines if mesh contains spectral elements (all quads)
 * 3. Computes centroid for each element and projects to unit sphere
 *
 * @return MB_SUCCESS on success, error code otherwise
 */
ErrorCode ScalarRemapper::extract_mesh_centroids() {
  // Get all 2D elements from the mesh set
  Range elements;
  ErrorCode rval =
      m_interface->get_entities_by_dimension(m_mesh_set, 2, elements);

  if (rval != MB_SUCCESS || elements.empty()) {
    // Fallback to 0D elements (point cloud)
    rval = m_interface->get_entities_by_dimension(m_mesh_set, 0, elements);
    if (rval != MB_SUCCESS || elements.empty()) {
      LOG(ERROR) << "No 2D QUADS or 0-D (point-cloud) elements found in "
                    "mesh. Aborting...";
      return MB_FAILURE;
    }
    m_target_is_spectral = false;
  } else {
    // Check if all elements are quadrilaterals for spectral remapping
    m_target_is_spectral = elements.all_of_type(MBQUAD);
  }

  // Reserve memory for efficiency
  m_mesh_data.elements.clear();
  m_mesh_data.centroids.clear();
  m_mesh_data.elements.reserve(elements.size());
  m_mesh_data.centroids.reserve(elements.size());

  // Compute centroids for each element
  for (EntityHandle element : elements) {
    PointType3D centroid;
    MB_CHK_ERR(compute_element_centroid(element, centroid));

    m_mesh_data.elements.push_back(element);
    m_mesh_data.centroids.push_back(centroid);
  }

  LOG(INFO) << "Extracted " << m_mesh_data.elements.size()
            << " element centroids";

  return MB_SUCCESS;
}

/**
 * @brief Compute centroid of a mesh element and project to unit sphere
 *
 * For a single vertex (0D element), uses the vertex coordinates directly.
 * For higher-dimensional elements, computes the average of vertex coordinates.
 * The result is normalized to lie on the unit sphere.
 *
 * @param element Entity handle of the element
 * @param centroid Output 3D centroid coordinates
 * @return MB_SUCCESS on success, error code otherwise
 */
ErrorCode ScalarRemapper::compute_element_centroid(EntityHandle element,
                                                   PointType3D &centroid) {
  // Get element coordinates (works for both vertices and elements)
  CartVect centroid_data;
  MB_CHK_ERR(m_interface->get_coords(&element, 1, centroid_data.array()));

  // Project to unit sphere (normalize to unit length)
  centroid_data.normalize();

  // Copy to output array
  for (size_t i = 0; i < 3; ++i) {
    centroid[i] = centroid_data[i];
  }

  return MB_SUCCESS;
}

//===========================================================================
// Validation and Statistics Methods
//===========================================================================

/**
 * @brief Validate remapping results for numerical issues
 *
 * Checks for NaN or infinite values in all remapped scalar fields.
 * This helps catch numerical issues early in the processing pipeline.
 *
 * @return MB_SUCCESS if all values are valid, MB_FAILURE otherwise
 */
ErrorCode ScalarRemapper::validate_remapping_results() {
  // Check double precision scalar fields
  for (const auto &field_pair : m_mesh_data.d_scalar_fields) {
    const auto &field_data = field_pair.second;
    for (double value : field_data) {
      if (std::isnan(value) || std::isinf(value)) {
        std::cerr << "Invalid value detected in remapped field "
                  << field_pair.first;
        return MB_FAILURE;
      }
    }
  }

  // Check integer scalar fields
  for (const auto &field_pair : m_mesh_data.i_scalar_fields) {
    const auto &field_data = field_pair.second;
    for (int value : field_data) {
      if (std::isnan(value) || std::isinf(value)) {
        std::cerr << "Invalid value detected in remapped field "
                  << field_pair.first;
        return MB_FAILURE;
      }
    }
  }

  return MB_SUCCESS;
}

/**
 * @brief Print statistics about remapping results
 *
 * Computes and displays min/max/average statistics for all remapped
 * scalar fields to help users verify the results look reasonable.
 */
void ScalarRemapper::print_remapping_statistics() {
  LOG(INFO) << "\n=== Remapping Statistics ===";

  // Statistics for double precision fields
  for (const auto &field_pair : m_mesh_data.d_scalar_fields) {
    const std::string &var_name = field_pair.first;
    const auto &field_data = field_pair.second;

    if (field_data.empty())
      continue;

    double min_val = *std::min_element(field_data.begin(), field_data.end());
    double max_val = *std::max_element(field_data.begin(), field_data.end());
    double sum = std::accumulate(field_data.begin(), field_data.end(), 0.0);
    double avg = sum / field_data.size();

    LOG(INFO) << var_name << " - Min: " << min_val << ", Max: " << max_val
              << ", Avg: " << avg << " (on " << field_data.size()
              << " elements)";
  }

  // Statistics for integer fields
  for (const auto &field_pair : m_mesh_data.i_scalar_fields) {
    const std::string &var_name = field_pair.first;
    const auto &field_data = field_pair.second;

    if (field_data.empty())
      continue;

    double min_val = *std::min_element(field_data.begin(), field_data.end());
    double max_val = *std::max_element(field_data.begin(), field_data.end());
    double sum = std::accumulate(field_data.begin(), field_data.end(), 0);
    double avg = sum / field_data.size();

    LOG(INFO) << var_name << " - Min: " << min_val << ", Max: " << max_val
              << ", Avg: " << avg << " (on " << field_data.size()
              << " elements)";
  }
}

//===========================================================================
// MOAB Tag Writing Methods
//===========================================================================

/**
 * @brief Template function to write scalar data to MOAB tags
 *
 * Creates dense tags of appropriate type and writes data to mesh elements.
 * This template handles both double and integer data types.
 *
 * @tparam T Data type (double or int)
 * @param interface MOAB interface instance
 * @param tag_name Name of the tag to create/write
 * @param elements Vector of element handles
 * @param data Vector of data values
 * @return MB_SUCCESS on success, error code otherwise
 */
template <typename T>
static ErrorCode write_to_tag(moab::Interface *interface,
                              const std::string &tag_name,
                              const std::vector<EntityHandle> &elements,
                              const std::vector<T> &data) {
  // Create or get the tag with appropriate data type
  Tag tag;
  if (sizeof(T) == sizeof(double)) {
    MB_CHK_ERR(interface->tag_get_handle(tag_name.c_str(), 1, MB_TYPE_DOUBLE,
                                         tag, MB_TAG_DENSE | MB_TAG_CREAT));
  } else {
    MB_CHK_ERR(interface->tag_get_handle(tag_name.c_str(), 1, MB_TYPE_INTEGER,
                                         tag, MB_TAG_DENSE | MB_TAG_CREAT));
  }

  // Set tag data on all elements
  if (!elements.empty() && !data.empty()) {
    MB_CHK_ERR(interface->tag_set_data(tag, elements.data(), elements.size(),
                                       data.data()));
  }

  return MB_SUCCESS;
}

/**
 * @brief Write all remapped scalar fields to MOAB tags
 *
 * Creates MOAB tags for all remapped scalar variables and writes the data
 * to the corresponding mesh elements. Tags are created with a prefix
 * to avoid naming conflicts.
 *
 * @param tag_prefix Prefix to add to all tag names
 * @return MB_SUCCESS on success, error code otherwise
 */
ErrorCode ScalarRemapper::write_to_tags(const std::string &tag_prefix) {
  // Write double precision scalar fields
  for (const auto &field_pair : m_mesh_data.d_scalar_fields) {
    const std::string &var_name = field_pair.first;
    const auto &field_data = field_pair.second;

    std::string tag_name = tag_prefix + var_name;
    MB_CHK_ERR(write_to_tag<double>(m_interface, tag_name, m_mesh_data.elements,
                                    field_data));

    LOG(INFO) << "Created tag '" << tag_name << "' with " << field_data.size()
              << " values";
  }

  // Write integer scalar fields
  for (const auto &field_pair : m_mesh_data.i_scalar_fields) {
    const std::string &var_name = field_pair.first;
    const auto &field_data = field_pair.second;

    std::string tag_name = tag_prefix + var_name;
    MB_CHK_ERR(write_to_tag<int>(m_interface, tag_name, m_mesh_data.elements,
                                 field_data));

    LOG(INFO) << "Created tag '" << tag_name << "' with " << field_data.size()
              << " values";
  }

  return MB_SUCCESS;
}

//===========================================================================
// Nearest Neighbor Remapper Implementation
//===========================================================================

/**
 * @brief Constructor for NearestNeighborRemapper
 * @param interface MOAB interface instance
 * @param mesh_set Mesh set containing target elements
 */
NearestNeighborRemapper::NearestNeighborRemapper(Interface *interface,
                                                 EntityHandle mesh_set)
    : ScalarRemapper(interface, mesh_set) {}

/**
 * @brief Destructor - KD-tree cleanup handled automatically by unique_ptr
 */
NearestNeighborRemapper::~NearestNeighborRemapper() {
  // KD-tree cleanup handled automatically by unique_ptr
}

//===========================================================================
// Spatial Index Construction Methods
//===========================================================================

/**
 * @brief Build RegularGridLocator for USGS format data
 *
 * For USGS format data with structured lat/lon grids, this method creates
 * a RegularGridLocator that provides fast spatial queries without the
 * overhead of building a KD-tree.
 *
 * @param point_data Input point cloud data
 * @return MB_SUCCESS on success, error code otherwise
 */
ErrorCode ScalarRemapper::build_grid_locator(
    const ParallelPointCloudReader::PointData &point_data) {
  if (point_data.size() == 0) {
    return MB_FAILURE;
  }

  try {
    LOG(INFO) << "Building RegularGridLocator for USGS format data...";

    // For USGS format, use the 1D lat/lon arrays directly
    if (!point_data.latitudes.empty() && !point_data.longitudes.empty()) {
      LOG(INFO) << "Using stored 1D arrays: " << point_data.latitudes.size()
                << " latitudes x " << point_data.longitudes.size()
                << " longitudes";

      // Convert to std::vector<double> if needed
      std::vector<double> lats(point_data.latitudes.begin(),
                               point_data.latitudes.end());
      std::vector<double> lons(point_data.longitudes.begin(),
                               point_data.longitudes.end());

      auto build_start = std::chrono::high_resolution_clock::now();
      m_grid_locator = std::unique_ptr<RegularGridLocator>(
          new RegularGridLocator(lats, lons, m_config.distance_metric));
      m_grid_locator_built = true;

      auto build_end = std::chrono::high_resolution_clock::now();
      auto build_duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(build_end -
                                                                build_start);
      LOG(INFO) << "RegularGridLocator build time: " << build_duration.count()
                << " ms";
    } else {
      // Fallback: extract unique lat/lon values from point cloud
      std::vector<double> lats, lons;
      std::set<double> unique_lats, unique_lons;

      for (const auto &pt : point_data.lonlat_coordinates) {
        unique_lons.insert(pt[0]);
        unique_lats.insert(pt[1]);
      }

      lons.assign(unique_lons.begin(), unique_lons.end());
      lats.assign(unique_lats.begin(), unique_lats.end());

      LOG(INFO) << "Extracted grid dimensions: " << lats.size() << " x "
                << lons.size();

      auto build_start = std::chrono::high_resolution_clock::now();
      m_grid_locator = std::unique_ptr<RegularGridLocator>(
          new RegularGridLocator(lats, lons, m_config.distance_metric));
      m_grid_locator_built = true;

      auto build_end = std::chrono::high_resolution_clock::now();
      auto build_duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(build_end -
                                                                build_start);
      LOG(INFO) << "RegularGridLocator build time: " << build_duration.count()
                << " ms";
    }

    return MB_SUCCESS;
  } catch (const std::exception &e) {
    std::cerr << "Error building RegularGridLocator: " << e.what();
    m_grid_locator_built = false;
    return MB_FAILURE;
  }
}

/**
 * @brief Build KD-tree for fast spatial queries
 *
 * Constructs a nanoflann KD-tree from the point cloud data for efficient
 * nearest neighbor and radius searches. Supports both structured and
 * unstructured point clouds.
 *
 * @param point_data Input point cloud data
 * @return MB_SUCCESS on success, error code otherwise
 */
ErrorCode ScalarRemapper::build_kdtree(
    const ParallelPointCloudReader::PointData &point_data) {
  if (point_data.size() == 0) {
    return MB_FAILURE;
  }

  // Determine number of threads for parallel KD-tree construction
  int num_threads = 1;
#ifdef _OPENMP
#pragma omp parallel
  {
    num_threads = omp_get_max_threads();
  }
#endif

  try {
    // For structured grids, materialize coordinates for KD-tree
    std::vector<PointType> coords_for_kdtree;
    if (point_data.is_structured_grid) {
      LOG(INFO) << "Materializing " << point_data.size()
                << " coordinates for KD-tree from structured grid...";
      coords_for_kdtree.resize(point_data.size());
      for (size_t i = 0; i < point_data.size(); ++i) {
        coords_for_kdtree[i] = point_data.get_lonlat(i);
      }
    }

    // Create adapter for point cloud data
    const auto &coords = point_data.is_structured_grid
                             ? coords_for_kdtree
                             : point_data.lonlat_coordinates;
    m_adapter =
        std::unique_ptr<PointCloudAdapter>(new PointCloudAdapter(coords));

    // Create KD-tree with optimized parameters for performance
    LOG(INFO) << "Building KD-tree index now with " << num_threads
              << " threads...";

    auto build_start = std::chrono::high_resolution_clock::now();
    m_kdtree = std::unique_ptr<KDTree>(new KDTree(
        3, *m_adapter,
        nanoflann::KDTreeSingleIndexAdaptorParams(
            16, nanoflann::KDTreeSingleIndexAdaptorFlags::SkipInitialBuildIndex,
            num_threads /* number of threads */)));

    // Build the index
    m_kdtree->buildIndex();
    m_kdtree_built = true;

    auto build_end = std::chrono::high_resolution_clock::now();
    auto build_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        build_end - build_start);
    LOG(INFO) << "KD-tree build time: " << build_duration.count() << " ms";
    LOG(INFO) << "Built KD-tree index for " << point_data.size() << " points";

    return MB_SUCCESS;
  } catch (const std::exception &e) {
    std::cerr << "Error building KD-tree: " << e.what();
    m_kdtree_built = false;
    return MB_FAILURE;
  }
}

/**
 * @brief Perform nearest neighbor remapping
 *
 * Algorithm:
 * 1. Build spatial index (KD-tree or RegularGridLocator)
 * 2. For each mesh element centroid, find the nearest point cloud point
 * 3. Copy scalar values from nearest point to mesh element
 *
 * This is the simplest and fastest remapping method, suitable for
 * dense point clouds where high accuracy is not required.
 *
 * @param point_data Input point cloud data with scalar fields
 * @return MB_SUCCESS on success, error code otherwise
 */
ErrorCode NearestNeighborRemapper::perform_remapping(
    const ParallelPointCloudReader::PointData &point_data) {
  if (point_data.size() == 0) {
    LOG(INFO) << "No point cloud data available for remapping";
    return MB_SUCCESS;
  }

  // Build spatial index for efficient nearest neighbor queries
  if (m_config.is_usgs_format && !m_config.use_kd_tree) {
    if (!m_grid_locator_built) {
      MB_CHK_ERR(build_grid_locator(point_data));
    }
  } else {
    if (!m_kdtree_built) {
      MB_CHK_ERR(build_kdtree(point_data));
    }
  }

  constexpr size_t max_neighbors = 1;

  // Parallel processing of mesh elements
#pragma omp parallel for shared(m_mesh_data, point_data, m_kdtree, m_config)
  for (size_t elem_idx = 0; elem_idx < m_mesh_data.centroids.size();
       ++elem_idx) {
    const PointType3D &centroid = m_mesh_data.centroids[elem_idx];

    // Find the nearest point cloud point
    auto nearest_point =
        find_nearest_point(centroid, point_data, &max_neighbors);
    size_t nearest_point_idx = nearest_point.size() > 0
                                   ? nearest_point[0].first
                                   : static_cast<size_t>(-1);

    // Debug output for first few elements
    if (elem_idx < 10) {
      LOG(INFO) << "Nearest point for element " << elem_idx << " is "
                << nearest_point_idx;
    }

    if (nearest_point_idx != static_cast<size_t>(-1)) {
      // Copy scalar values from nearest point to mesh element
      for (const auto &var_name : m_config.scalar_var_names) {
        // Handle double precision variables
        auto var_it = point_data.d_scalar_variables.find(var_name);
        if (var_it != point_data.d_scalar_variables.end() &&
            nearest_point_idx < var_it->second.size()) {
          m_mesh_data.d_scalar_fields[var_name][elem_idx] =
              var_it->second[nearest_point_idx];
          if (elem_idx < 10) {
            LOG(INFO) << "Double Data (" << var_name
                      << ") at nearest neighbor element: " << nearest_point_idx
                      << " is " << var_it->second[nearest_point_idx];
          }
        } else {
          // Handle integer variables
          auto ivar_it = point_data.i_scalar_variables.find(var_name);
          if (ivar_it != point_data.i_scalar_variables.end() &&
              nearest_point_idx < ivar_it->second.size()) {
            m_mesh_data.i_scalar_fields[var_name][elem_idx] =
                ivar_it->second[nearest_point_idx];
            if (elem_idx < 10) {
              LOG(INFO) << "Integer Data (" << var_name
                        << ") at nearest neighbor element: "
                        << nearest_point_idx << " is "
                        << ivar_it->second[nearest_point_idx];
            }
          }
        }
      }
    } else {
      LOG(INFO) << "No nearest point found for element " << elem_idx;
    }
  }

  return MB_SUCCESS;
}

/**
 * @brief Find nearest points to a target location using spatial indexing
 *
 * This method provides a unified interface for spatial queries using either:
 * - RegularGridLocator (for USGS format structured grids)
 * - KD-tree (for unstructured point clouds)
 * - Linear search (fallback when spatial indices are unavailable)
 *
 * Supports both k-nearest neighbor and radius search modes.
 *
 * @param target_point 3D Cartesian target point
 * @param point_data Point cloud data to search within
 * @param user_max_neighbors Optional maximum number of neighbors to return
 * @param user_search_radius Optional search radius (0.0 = no radius constraint)
 * @return Vector of ResultItem containing point indices and distances
 */
std::vector<nanoflann::ResultItem<size_t, CoordinateType>>
ScalarRemapper::find_nearest_point(
    const PointType3D &target_point,
    const ParallelPointCloudReader::PointData &point_data,
    const size_t *user_max_neighbors,
    const CoordinateType *user_search_radius) {

  // Set default search parameters
  size_t max_neighbors = 1;
  CoordinateType search_radius = 0.0;
  if (user_max_neighbors)
    max_neighbors = *user_max_neighbors;
  if (user_search_radius)
    search_radius = *user_search_radius;

  std::vector<nanoflann::ResultItem<size_t, CoordinateType>> ret_matches;

  // Use RegularGridLocator for USGS format if enabled
  if (m_config.is_usgs_format && !m_config.use_kd_tree &&
      m_grid_locator_built && m_grid_locator) {
    // Convert 3D Cartesian to lon/lat for grid query
    CoordinateType lon, lat;
    XYZtoRLL_Deg(target_point.data(), lon, lat);
    PointType3D query_pt = {lon, lat, 0.0};

    if (search_radius > 0.0) {
      // Here, search radius is in radians. Need to convert to degrees
      search_radius = search_radius * RAD_TO_DEG;
      // Radius search within specified distance
      size_t num_matches = m_grid_locator->radiusSearch(query_pt, search_radius,
                                                        ret_matches, false);
      //   std::cout << "Search radius = " << search_radius
      //             << " radians, Number of matches: " << num_matches <<
      //             std::endl;
      if (num_matches == 0) {
        ret_matches.push_back(nanoflann::ResultItem<size_t, CoordinateType>(
            static_cast<size_t>(-1), 0.0));
      }
    } else {
      // k-nearest neighbor search
      std::vector<size_t> indices(max_neighbors);
      std::vector<CoordinateType> distances_sq(max_neighbors);
      m_grid_locator->knnSearch(query_pt, max_neighbors, indices.data(),
                                distances_sq.data());

      for (size_t i = 0; i < max_neighbors; ++i) {
        ret_matches.push_back({indices[i], distances_sq[i]});
      }
    }
    return ret_matches;
  }

  // Use KD-tree for fast spatial queries
  if (m_kdtree_built && m_kdtree) {
    return kdtree_search(target_point, angular_to_cartesian(search_radius / 12),
                         max_neighbors);
  } else {
    // Fallback to linear search when KD-tree is unavailable
    return linear_search_fallback(target_point, point_data, search_radius);
  }
}

//===========================================================================
// Spatial Search Helper Methods
//===========================================================================

/**
 * @brief Linear search fallback when spatial indices are unavailable
 *
 * Performs a brute-force O(n) search through all points. This is used as
 * a fallback when KD-tree construction fails or is not available.
 *
 * @param target_point 3D Cartesian target point
 * @param point_data Point cloud data to search within
 * @param search_radius Search radius constraint (0.0 = no constraint)
 * @return Vector of ResultItem containing point indices and distances
 */
std::vector<nanoflann::ResultItem<size_t, CoordinateType>>
ScalarRemapper::linear_search_fallback(
    const PointType3D &target_point,
    const ParallelPointCloudReader::PointData &point_data,
    CoordinateType search_radius) {
  std::vector<nanoflann::ResultItem<size_t, CoordinateType>> ret_matches;

  size_t nearest_idx = static_cast<size_t>(-1);
  CoordinateType min_distance = std::numeric_limits<CoordinateType>::max();

  // Linear search through all points
  for (size_t i = 0; i < point_data.size(); ++i) {
    auto coord = point_data.get_lonlat(i);
    CoordinateType distance = private_compute_distance(target_point, coord);

    // Check search radius constraint
    if (search_radius > 0.0 && distance > search_radius) {
      continue;
    }

    if (distance < min_distance) {
      min_distance = distance;
      nearest_idx = i;
    }
  }

  ret_matches.push_back(
      nanoflann::ResultItem<size_t, CoordinateType>(nearest_idx, min_distance));
  return ret_matches;
}

/**
 * @brief KD-tree based spatial search
 *
 * Performs fast O(log n) spatial queries using the nanoflann KD-tree.
 * Supports both radius search and k-nearest neighbor search.
 *
 * @param target_point 3D Cartesian target point
 * @param search_radius Search radius (0.0 = kNN search)
 * @param max_neighbors Maximum number of neighbors to return
 * @return Vector of ResultItem containing point indices and distances
 */
std::vector<nanoflann::ResultItem<size_t, CoordinateType>>
ScalarRemapper::kdtree_search(const PointType3D &target_point,
                              CoordinateType search_radius,
                              size_t max_neighbors) {
  std::vector<nanoflann::ResultItem<size_t, CoordinateType>> ret_matches;

  try {
    nanoflann::SearchParameters params;
    params.sorted = true;
    params.eps = std::numeric_limits<CoordinateType>::epsilon() *
                 10; // 10x machine epsilon

    if (search_radius > 0.0) {
      // Radius search with distance constraint
      const size_t num_matches = m_kdtree->radiusSearch(
          target_point.data(), search_radius, ret_matches, params);

      if (num_matches == 0) {
        ret_matches.push_back(nanoflann::ResultItem<size_t, CoordinateType>(
            static_cast<size_t>(-1), 0.0));
      }
    } else {
      // k-nearest neighbor search
      std::vector<size_t> ret_index(max_neighbors);
      std::vector<CoordinateType> out_dist_sqr(max_neighbors);

      m_kdtree->knnSearch(target_point.data(), max_neighbors, ret_index.data(),
                          out_dist_sqr.data());

      // Return the nearest point indices and distances
      for (size_t i = 0; i < max_neighbors; ++i) {
        ret_matches.push_back(nanoflann::ResultItem<size_t, CoordinateType>(
            ret_index[i], out_dist_sqr[i]));
      }
    }
  } catch (const std::exception &e) {
    std::cerr << "Error in KD-tree search: " << e.what();
    ret_matches.push_back(nanoflann::ResultItem<size_t, CoordinateType>(
        static_cast<size_t>(-1), 0.0));
  }

  return ret_matches;
}

//===========================================================================
// Factory Implementation
//===========================================================================

/**
 * @brief Create a remapper instance using the factory pattern
 *
 * This factory method provides a clean interface for creating different
 * types of remappers without exposing the specific class implementations.
 *
 * @param method The remapping algorithm to use
 * @param interface MOAB interface instance
 * @param mesh_set Mesh set containing target elements
 * @return Unique pointer to the created remapper instance
 */
std::unique_ptr<ScalarRemapper>
RemapperFactory::create_remapper(RemapMethod method, Interface *interface,
                                 EntityHandle mesh_set) {

  switch (method) {
  case ALG_DISKAVERAGE:
    // Default: Point cloud disk averaged projection with spectral element
    // support
    return std::unique_ptr<ScalarRemapper>(
        new PCDiskAveragedProjectionRemapper(interface, mesh_set));

  case ALG_NEAREST_NEIGHBOR:
    // Fast nearest neighbor mapping
    return std::unique_ptr<ScalarRemapper>(
        new NearestNeighborRemapper(interface, mesh_set));

  default:
    std::cerr << "Unknown remapping method";
    return std::unique_ptr<ScalarRemapper>();
  }
}

static moab::ErrorCode
Remapper_ApplyLocalMap(Interface * /* mb */, const double nodes[12],
                       double dAlpha, double dBeta, CartVect &node,
                       CartVect *dDx1G = nullptr, CartVect *dDx2G = nullptr) {
  // Bilinear interpolation on reference element
  double dXc = nodes[0 * 3 + 0] * (1.0 - dAlpha) * (1.0 - dBeta) +
               nodes[1 * 3 + 0] * dAlpha * (1.0 - dBeta) +
               nodes[2 * 3 + 0] * dAlpha * dBeta +
               nodes[3 * 3 + 0] * (1.0 - dAlpha) * dBeta;

  double dYc = nodes[0 * 3 + 1] * (1.0 - dAlpha) * (1.0 - dBeta) +
               nodes[1 * 3 + 1] * dAlpha * (1.0 - dBeta) +
               nodes[2 * 3 + 1] * dAlpha * dBeta +
               nodes[3 * 3 + 1] * (1.0 - dAlpha) * dBeta;

  double dZc = nodes[0 * 3 + 2] * (1.0 - dAlpha) * (1.0 - dBeta) +
               nodes[1 * 3 + 2] * dAlpha * (1.0 - dBeta) +
               nodes[2 * 3 + 2] * dAlpha * dBeta +
               nodes[3 * 3 + 2] * (1.0 - dAlpha) * dBeta;

  // Project to unit sphere
  double dR = sqrt(dXc * dXc + dYc * dYc + dZc * dZc);
  node[0] = dXc / dR;
  node[1] = dYc / dR;
  node[2] = dZc / dR;

  // Compute derivatives if requested
  if (dDx1G || dDx2G) {
    // Derivatives in Cartesian coordinates
    CartVect dDx1F((1.0 - dBeta) * (nodes[1 * 3 + 0] - nodes[0 * 3 + 0]) +
                       dBeta * (nodes[2 * 3 + 0] - nodes[3 * 3 + 0]),
                   (1.0 - dBeta) * (nodes[1 * 3 + 1] - nodes[0 * 3 + 1]) +
                       dBeta * (nodes[2 * 3 + 1] - nodes[3 * 3 + 1]),
                   (1.0 - dBeta) * (nodes[1 * 3 + 2] - nodes[0 * 3 + 2]) +
                       dBeta * (nodes[2 * 3 + 2] - nodes[3 * 3 + 2]));

    CartVect dDx2F((1.0 - dAlpha) * (nodes[3 * 3 + 0] - nodes[0 * 3 + 0]) +
                       dAlpha * (nodes[2 * 3 + 0] - nodes[1 * 3 + 0]),
                   (1.0 - dAlpha) * (nodes[3 * 3 + 1] - nodes[0 * 3 + 1]) +
                       dAlpha * (nodes[2 * 3 + 1] - nodes[1 * 3 + 1]),
                   (1.0 - dAlpha) * (nodes[3 * 3 + 2] - nodes[0 * 3 + 2]) +
                       dAlpha * (nodes[2 * 3 + 2] - nodes[1 * 3 + 2]));

    // Convert to spherical derivatives
    double dDenomTerm = 1.0 / (dR * dR * dR);

    if (dDx1G) {
      *dDx1G = dDenomTerm * CartVect(-dXc * (dYc * dDx1F[1] + dZc * dDx1F[2]) +
                                         (dYc * dYc + dZc * dZc) * dDx1F[0],
                                     -dYc * (dXc * dDx1F[0] + dZc * dDx1F[2]) +
                                         (dXc * dXc + dZc * dZc) * dDx1F[1],
                                     -dZc * (dXc * dDx1F[0] + dYc * dDx1F[1]) +
                                         (dXc * dXc + dYc * dYc) * dDx1F[2]);
    }

    if (dDx2G) {
      *dDx2G = dDenomTerm * CartVect(-dXc * (dYc * dDx2F[1] + dZc * dDx2F[2]) +
                                         (dYc * dYc + dZc * dZc) * dDx2F[0],
                                     -dYc * (dXc * dDx2F[0] + dZc * dDx2F[2]) +
                                         (dXc * dXc + dZc * dZc) * dDx2F[1],
                                     -dZc * (dXc * dDx2F[0] + dYc * dDx2F[1]) +
                                         (dXc * dXc + dYc * dYc) * dDx2F[2]);
    }
  }

  return moab::MB_SUCCESS;
}

/**
 * @brief Generate finite element metadata for spectral element remapping
 *
 * This function computes the Jacobian and other metadata required for
 * high-order spectral element projection. It evaluates the mapping from
 * reference to physical coordinates at GLL quadrature points.
 *
 * Algorithm:
 * 1. Get element connectivity and coordinates
 * 2. For each GLL point (i,j), compute physical coordinates using bilinear
 * mapping
 * 3. Compute Jacobian determinant at each GLL point
 * 4. Apply optional bubble correction for mass conservation
 *
 * @param mb MOAB interface instance
 * @param face Entity handle of the quadrilateral element
 * @param dG GLL quadrature points in [0,1]
 * @param dW GLL quadrature weights
 * @param dataGLLJacobian Output Jacobian values at each GLL point
 * @param apply_bubble_correction Whether to apply mass-conserving bubble
 * correction
 * @return Total numerical area of the element
 */
double Remapper_GenerateFEMetaData(Interface *mb, EntityHandle face,
                                   const std::vector<double> &dG,
                                   const std::vector<double> &dW,
                                   std::vector<double> &dataGLLJacobian,
                                   const bool apply_bubble_correction = false) {
  const int nP = dG.size();

  // Initialize output Jacobian array
  dataGLLJacobian.resize(nP * nP);

  // Get element connectivity and validate it's a quadrilateral
  const EntityHandle *connectivity;
  int nConnectivity;
  MB_CHK_SET_ERR(mb->get_connectivity(face, connectivity, nConnectivity, true),
                 "Failed to get element connectivity");

  if (nP != 4 || nConnectivity != 4) {
    MB_CHK_SET_ERR(MB_FAILURE, "Mesh must only contain quadrilateral elements");
  }

  // Get node coordinates
  double nConnCoords[12]; // 4 nodes × 3 coordinates
  MB_CHK_SET_ERR(mb->get_coords(connectivity, nConnectivity, nConnCoords),
                 "Failed to get element coordinates");

  double dFaceNumericalArea = 0.0;

  // Loop over all GLL quadrature points
  for (int j = 0; j < nP; j++) {
    for (int i = 0; i < nP; i++) {
      // Map reference coordinates to physical coordinates
      CartVect nodeGLL;
      CartVect dDx1G, dDx2G; // Derivatives

      Remapper_ApplyLocalMap(mb, nConnCoords, dG[i], dG[j], nodeGLL, &dDx1G,
                             &dDx2G);

      // Compute Jacobian determinant: |∂(x,y,z)/∂(ξ,η)|
      CartVect nodeCross = dDx1G * dDx2G;
      const double dJacobian = nodeCross.length() * dW[i] * dW[j];

      if (dJacobian <= 0.0) {
        MB_CHK_SET_ERR(
            MB_FAILURE,
            "Nonpositive Jacobian detected - invalid element geometry");
      }

      dFaceNumericalArea += dJacobian;
      dataGLLJacobian[j * nP + i] = dJacobian;
    }
  }

  // Apply bubble correction for mass conservation (optional)
  if (apply_bubble_correction) {
    // Calculate exact spherical polygon area using l'Huilier's method
    moab::IntxAreaUtils areaAdaptor(moab::IntxAreaUtils::lHuiller);
    const double vecFaceArea =
        areaAdaptor.area_spherical_polygon(nConnCoords, 4, 1.0);

    if (dFaceNumericalArea != vecFaceArea) {
      const double dMassDifference = vecFaceArea - dFaceNumericalArea;

      if (nP < 3) {
        // Linear elements: apply uniform correction to all GLL points
        for (int j = 0; j < nP; j++) {
          for (int i = 0; i < nP; i++) {
            dataGLLJacobian[j * nP + i] += dMassDifference * dW[i] * dW[j];
          }
        }
        dFaceNumericalArea += dMassDifference;
      } else {
        // Higher-order elements: apply HOMME-style bubble correction to
        // interior points only
        double dInteriorMassSum = 0.0;
        for (int i = 1; i < nP - 1; i++) {
          for (int j = 1; j < nP - 1; j++) {
            dInteriorMassSum += dataGLLJacobian[j * nP + i];
          }
        }

        // Check for numerical issues
        if (std::abs(dInteriorMassSum) < 1e-15) {
          MB_CHK_SET_ERR(MB_FAILURE, "Bubble correction cannot be performed: "
                                     "sum of interior weights is too small");
        }

        // Scale interior points to conserve mass
        const double dScalingFactor = dMassDifference / dInteriorMassSum;
        for (int j = 1; j < nP - 1; j++) {
          for (int i = 1; i < nP - 1; i++) {
            dataGLLJacobian[j * nP + i] *= (1.0 + dScalingFactor);
          }
        }
        dFaceNumericalArea += dMassDifference;
      }
    }
  }

  // Return the computed element area
  return dFaceNumericalArea;
}

///////////////////////////////////////////////////////////////////////////////
// ----------------------
// KD-tree point cloud
// ----------------------
struct MOABCentroidCloud {
  constexpr static int DIM = 3;
  std::vector<std::array<double, DIM>> points;
  std::vector<moab::EntityHandle> elements;

  inline void init(size_t length) {
    points.reserve(length);
    elements.reserve(length);
  }

  inline size_t kdtree_get_point_count() const { return points.size(); }

  inline double kdtree_get_pt(const size_t idx, const size_t dim) const {
    return points[idx][dim];
  }

  template <class BBOX> bool kdtree_get_bbox(BBOX &) const { return false; }
};

using MOABKDTree = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<CoordinateType, MOABCentroidCloud,
                                 CoordinateType>,
    MOABCentroidCloud, // DatasetAdaptor
    3,                 // 3D
    size_t             // IndexType
    >;

// ----------------------
// Radius search wrapper
// ----------------------
std::vector<size_t>
radius_search_kdtree(const MOABKDTree &tree, const MOABCentroidCloud &cloud,
                     const std::array<CoordinateType, 3> &query_pt,
                     CoordinateType radius) {
  CoordinateType radius_sq = radius * radius;
  // CoordinateType radius_sq = radius;
  std::vector<nanoflann::ResultItem<size_t, CoordinateType>> matches;
  nanoflann::SearchParameters params;
  params.sorted = true;

  const CoordinateType query_pt_sq = query_pt[0] * query_pt[0] +
                                     query_pt[1] * query_pt[1] +
                                     query_pt[2] * query_pt[2];
  if (query_pt_sq > 1.0 - 1e-10 &&
      query_pt_sq < 1.0 + 1.0e-10) // Check if the point is on the unit sphere
  {
    LOG(FATAL) << "Point " << query_pt[0] << ", " << query_pt[1] << ", "
               << query_pt[2] << " is not on the unit sphere";
    throw std::runtime_error("Point is not on the unit sphere");
  }

  tree.radiusSearch(query_pt.data(), radius_sq, matches, params);

  std::vector<size_t> found_elements;
  if (!matches.empty()) {
    // Return all matches within the radius
    for (const auto &match : matches)
      found_elements.emplace_back(cloud.elements[match.first]);
  } else {
    // Fallback to nearest neighbor
    size_t nearest_index;
    CoordinateType nearest_dist_sq;
    tree.knnSearch(query_pt.data(), 1, &nearest_index, &nearest_dist_sq);
    found_elements.emplace_back(cloud.elements[nearest_index]);
  }

  return found_elements;
}

///////////////////////////////////////////////////////////////////////////////

// PCDiskAveragedProjectionRemapper Implementation
PCDiskAveragedProjectionRemapper::PCDiskAveragedProjectionRemapper(
    Interface *interface, EntityHandle mesh_set)
    : ScalarRemapper(interface, mesh_set) {}

ErrorCode PCDiskAveragedProjectionRemapper::perform_remapping(
    const ParallelPointCloudReader::PointData &point_data) {
  if (point_data.size() == 0) {
    LOG(INFO) << "No point cloud data available for remapping";
    return MB_SUCCESS;
  }

  // Validate that mesh contains only quadrilaterals
  // MB_CHK_ERR(validate_quadrilateral_mesh());

  // Build spatial index (KD-tree or RegularGridLocator)
  if (m_config.is_usgs_format && !m_config.use_kd_tree) {
    if (!m_grid_locator_built) {
      MB_CHK_ERR(build_grid_locator(point_data));
    }
  } else {
    if (!m_kdtree_built) {
      MB_CHK_ERR(this->build_kdtree(point_data));
    }
  }

  LOG(INFO) << "";

  // Project point cloud data to all spectral elements

  if (m_target_is_spectral) {
    LOG(INFO) << "Starting PC averaged spectral projection with "
              << point_data.size() << " point cloud points and spectral order "
              << spectral_order;
    MB_CHK_ERR(project_point_cloud_to_spectral_elements(point_data));
  } else {
    LOG(INFO) << "Starting PC disk-area averaged projection with "
              << point_data.size() << " point cloud points";
    MB_CHK_ERR(project_point_cloud_with_area_averaging(point_data));
  }

  return MB_SUCCESS;
}

ErrorCode PCDiskAveragedProjectionRemapper::validate_quadrilateral_mesh() {
  // Check that all elements are quadrilaterals
  for (EntityHandle element : m_mesh_data.elements) {
    const EntityHandle *connectivity;
    int nConnectivity;
    MB_CHK_ERR(m_interface->get_connectivity(element, connectivity,
                                             nConnectivity, true));

    if (nConnectivity != 4) {
      std::cerr << "Error: Spectral element projection requires "
                   "quadrilateral mesh elements. "
                << "Found element with " << nConnectivity << " vertices.";
      return MB_FAILURE;
    }
  }

  LOG(INFO) << "Validated " << m_mesh_data.elements.size()
            << " quadrilateral elements for spectral projection";

  return MB_SUCCESS;
}

ErrorCode
PCDiskAveragedProjectionRemapper::project_point_cloud_to_spectral_elements(
    const ParallelPointCloudReader::PointData &point_data) {

  LOG(INFO) << "Performing spectral element projection with "
            << point_data.lonlat_coordinates.size() << " point cloud points";

  const int nP = spectral_order;

  LOG(INFO) << "Processing " << m_mesh_data.elements.size()
            << " quadrilateral elements in parallel";

  // Get GLL quadrature points and weights
  std::vector<double> dG, dW;
  MB_CHK_SET_ERR(GaussLobattoQuadrature::GetPoints(nP, dG, dW, 0.0, 1.0),
                 "Failed to get GLL quadrature points and weights");

  // Parallel processing of mesh elements
  std::vector<ErrorCode> element_errors(m_mesh_data.elements.size(),
                                        MB_SUCCESS);

  std::cout.precision(10);

  // const size_t nvars = m_config.scalar_var_names.size();

#pragma omp parallel for schedule(dynamic, 8) firstprivate(dG, dW)             \
    shared(m_kdtree, element_errors, point_data, m_mesh_data, m_config,        \
               m_interface)
  for (size_t elem_idx = 0; elem_idx < m_mesh_data.elements.size();
       ++elem_idx) {
    if ((elem_idx % (m_mesh_data.elements.size() / 10)) == 0) {
#pragma omp critical
      LOG(INFO) << "Processing element " + std::to_string(elem_idx) + " of " +
                       std::to_string(m_mesh_data.elements.size());
    }

    ErrorCode rval;
    EntityHandle face = m_mesh_data.elements[elem_idx];
    const EntityHandle *connectivity;
    int nConnectivity;
    rval =
        m_interface->get_connectivity(face, connectivity, nConnectivity, true);
    if (MB_SUCCESS != rval) {
      element_errors[elem_idx] = MB_FAILURE;
      continue;
    }

    double nConnCoords[12]; // 4 nodes, 3D coordinates
    rval = m_interface->get_coords(connectivity, nConnectivity, nConnCoords);
    if (MB_SUCCESS != rval) {
      element_errors[elem_idx] = MB_FAILURE;
      continue;
    }

    // Generate GLL metadata for this face (thread-safe)
    std::vector<double> dataGLLJacobian;

    // Generate finite element metadata using existing function
    /* double dNumericalArea = */ Remapper_GenerateFEMetaData(
        m_interface, face, dG, dW, dataGLLJacobian, apply_bubble_correction);

    // Initialize element-averaged values (thread-local)
    std::unordered_map<std::string, double> element_averages_d;
    // std::unordered_map<std::string, int> element_averages_i;
    std::unordered_map<std::string, double>
        total_weights; // Per-variable weight tracking

    // Pre-compute all GLL node coordinates for this element (vectorizable)
    std::vector<bool> gll_valid(nP * nP, false);

    // Process all GLL nodes for this element
    for (int j = 0; j < nP; j++) {
      for (int i = 0; i < nP; i++) {
        int gll_idx = j * nP + i;
        CartVect nodeGLL;
        rval = Remapper_ApplyLocalMap(m_interface, nConnCoords, dG[i], dG[j],
                                      nodeGLL);
        if (MB_SUCCESS != rval) {
          gll_valid[gll_idx] = false;
          continue;
        }

        // GLL point in 3D Cartesian coordinates (matches KD-tree
        // coordinate system)
        PointType3D gll_point = {static_cast<CoordinateType>(nodeGLL[0]),
                                 static_cast<CoordinateType>(nodeGLL[1]),
                                 static_cast<CoordinateType>(nodeGLL[2])};

        // Optimized KD-tree search with thread-local storage:
        thread_local std::vector<size_t> neighbor_indices;
        thread_local std::vector<CoordinateType> distances_sq;
        thread_local std::vector<nanoflann::ResultItem<size_t, CoordinateType>>
            matches;

        neighbor_indices.clear();
        distances_sq.clear();
        matches.clear();

        // Compute weighted average from neighboring points (vectorized)
        const CoordinateType search_radius = dataGLLJacobian[gll_idx];
        // const size_t max_neighbors = 5000; // if it exceeds, perhaps
        // SE mesh resolution is too coarse?

        auto nearest_points =
            find_nearest_point(gll_point, point_data, nullptr, &search_radius);

        // Check if radius search found valid results (not just dummy -1
        // index)
        bool has_valid_results =
            !nearest_points.empty() &&
            nearest_points[0].first != static_cast<size_t>(-1);

        if (!has_valid_results) {
          // Fallback to nearest neighbor WITHOUT radius constraint
          const size_t single_neighbor = 1;
          const CoordinateType no_radius =
              0.0; // 0.0 means no radius constraint
          nearest_points = find_nearest_point(gll_point, point_data,
                                              &single_neighbor, &no_radius);
        }

        // Check if we still have zero points after fallback
        if (nearest_points.empty()) {
          if (elem_idx < 10) {
#pragma omp critical
            LOG(WARNING) << "WARNING: Element " << elem_idx << ", GLL node ("
                         << i << "," << j
                         << ") has ZERO neighbors even after fallback!";
          }
          continue; // Skip this GLL node
        }

        // Pre-compute inverse distance weights (vectorizable)
        thread_local std::vector<CoordinateType> inv_weights;
        inv_weights.resize(nearest_points.size());
        std::fill(inv_weights.begin(), inv_weights.end(), 1.0);

        // Vectorized distance and weight computation
        // #pragma omp simd
        // for (size_t k = 0; k < nearest_points.size(); ++k) {
        //     CoordinateType distance =
        //     std::sqrt(nearest_points[k].second); inv_weights[k] = 1.0
        //     / (distance + 1e-10);
        // }

        // const double gll_weight_extensive =
        // dW[i]*dW[j]*dataGLLJacobian[gll_idx]; // For extensive
        // properties (area, mass) const double gll_weight_intensive =
        // dW[i]*dW[j]; // For intensive properties (fractions, ratios)
        // - NO Jacobian!
        const double gll_weight = dW[i] * dW[j];

        // Process each scalar variable
        for (const auto &var_name : m_config.scalar_var_names) {
          double weighted_sum = 0.0;
          double weight_sum = 0.0;

          // Determine if this is an extensive property (needs
          // Jacobian) bool is_extensive = (var_name == "area"); const
          // double gll_weight = is_extensive ? gll_weight_extensive :
          // gll_weight_intensive;

          // Check if it's a double variable
          auto var_it = point_data.d_scalar_variables.find(var_name);
          if (!m_config.is_usgs_format &&
              var_it != point_data.d_scalar_variables.end()) {
            const auto &values = var_it->second;

            // Vectorized weighted sum computation
            for (size_t k = 0; k < nearest_points.size(); ++k) {
              size_t pt_idx = nearest_points[k].first;
              if (pt_idx < values.size()) {
                weighted_sum += inv_weights[k] * values[pt_idx];
                weight_sum += inv_weights[k];
              }
            }
          } else {
            // Check if it's an integer variable
            auto ivar_it = point_data.i_scalar_variables.find(var_name);
            if (ivar_it != point_data.i_scalar_variables.end()) {
              const auto &values = ivar_it->second;

              // Vectorized weighted sum computation for integers
              for (size_t k = 0; k < nearest_points.size(); ++k) {
                size_t pt_idx = nearest_points[k].first;
                if (pt_idx < values.size()) {
                  weighted_sum +=
                      inv_weights[k] * static_cast<double>(values[pt_idx]);
                  weight_sum += inv_weights[k];
                }
              }
            }
          }

          // Add GLL-weighted contribution to element average
          if (fabs(weight_sum) > 0.0) {
            double gll_value = weighted_sum / weight_sum;
            element_averages_d[var_name] += gll_weight * gll_value;
            total_weights[var_name] += gll_weight; // Track weight per variable
          }
        }
      }
    }

    // Normalize element averages and store in mesh data (thread-safe
    // writes) Use per-variable weights for proper normalization
    for (const auto &var_name : m_config.scalar_var_names) {
      auto it = element_averages_d.find(var_name);
      auto wt_it = total_weights.find(var_name);

      if (it != element_averages_d.end() && wt_it != total_weights.end() &&
          fabs(wt_it->second) > 0.0) {
        const double inv_total_weight = 1.0 / wt_it->second;

        auto var_it = point_data.d_scalar_variables.find(var_name);
        if (var_it != point_data.d_scalar_variables.end()) {
          m_mesh_data.d_scalar_fields[var_name][elem_idx] =
              it->second * inv_total_weight;
        } else {
          auto ivar_it = point_data.i_scalar_variables.find(var_name);
          if (ivar_it != point_data.i_scalar_variables.end()) {
            m_mesh_data.i_scalar_fields[var_name][elem_idx] =
                static_cast<int>(it->second * inv_total_weight);
          }
        }
      }
    }

    // Check if any variable had data
    if (total_weights.empty()) {
      element_errors[elem_idx] = MB_FAILURE;
    }

  } // End of parallel region

  // Check for errors after parallel processing
  size_t error_count = 0;
  for (size_t i = 0; i < element_errors.size(); ++i) {
    if (element_errors[i] != MB_SUCCESS) {
      error_count++;
    }
  }

  if (error_count > 0) {
    LOG(ERROR) << "Warning: " << error_count
               << " elements failed processing out of "
               << m_mesh_data.elements.size() << " total elements";
  }

  // Thread-safe debugging output
  for (const auto &var_name : m_config.scalar_var_names) {
    std::array<double, 5> values;
    for (size_t i = 0; i < std::min(size_t(5), m_mesh_data.elements.size());
         ++i) {
      auto d_it = m_mesh_data.d_scalar_fields.find(var_name);
      auto i_it = m_mesh_data.i_scalar_fields.find(var_name);
      if (d_it != m_mesh_data.d_scalar_fields.end()) {
        values[i] = d_it->second[i];
      } else if (i_it != m_mesh_data.i_scalar_fields.end()) {
        values[i] = i_it->second[i];
      }
    }
    VLOG(2) << "Sample values for " << var_name << ": " << values;
  }

  return MB_SUCCESS;
}

ErrorCode
PCDiskAveragedProjectionRemapper::project_point_cloud_with_area_averaging(
    const ParallelPointCloudReader::PointData &point_data) {

  LOG(INFO) << "Performing topography projection with area averaging using "
            << point_data.size() << " point cloud points";

  const auto &elements = m_mesh_data.elements;

  // Parallel processing of mesh elements
  std::vector<ErrorCode> element_errors(elements.size(), MB_SUCCESS);

  std::cout.precision(10);

  std::vector<double> vertex_coords(elements.size() * 3, 0.0);
  std::vector<double> vertex_areas(elements.size(), 0.0);

  MB_CHK_ERR(m_interface->get_coords(elements.data(), elements.size(),
                                     vertex_coords.data()));

  Tag areaTag;
  MB_CHK_ERR(m_interface->tag_get_handle("area", areaTag));
  MB_CHK_ERR(m_interface->tag_get_data(areaTag, elements.data(),
                                       elements.size(), vertex_areas.data()));

  const size_t output_frequency = elements.size() / 10;
#pragma omp parallel for schedule(dynamic, 1)                                  \
    shared(m_kdtree, element_errors, point_data, m_mesh_data, m_config,        \
               m_interface, vertex_coords, vertex_areas)
  for (size_t elem_idx = 0; elem_idx < elements.size(); ++elem_idx) {
    if ((elem_idx + 1) % output_frequency == 0) {
#pragma omp critical
      LOG(INFO) << "Processing element " + std::to_string(elem_idx) + " of " +
                       std::to_string(elements.size());
    }

    // EntityHandle vertex = elements[elem_idx];
    PointType3D gll_point;
    std::copy_n(vertex_coords.data() + elem_idx * 3, 3, gll_point.begin());

    // Compute search radius from constant area:
    // A = (1.0/α²) × πr² → r = α × √(A/π)
    const double search_radius =
        m_config.user_alpha * std::sqrt(vertex_areas[elem_idx] / M_PI);

    if (elem_idx == 0) {
      VLOG(2) << "First element: Area = " << vertex_areas[elem_idx]
              << ", Search radius = " << search_radius;
    }

    // Initialize element-averaged values (thread-local)
    std::unordered_map<std::string, double> element_averages_d;
    // std::unordered_map<std::string, int> element_averages_i;
    std::unordered_map<std::string, double>
        total_weights; // Per-variable weight tracking

    // Optimized KD-tree search with thread-local storage:
    thread_local std::vector<size_t> neighbor_indices;
    thread_local std::vector<CoordinateType> distances_sq;
    thread_local std::vector<nanoflann::ResultItem<size_t, CoordinateType>>
        matches;

    neighbor_indices.clear();
    distances_sq.clear();
    matches.clear();

    // const size_t max_neighbors = 5000; // if it exceeds, perhaps SE mesh
    // resolution is too coarse?

    auto nearest_points =
        find_nearest_point(gll_point, point_data, nullptr, &search_radius);

    // Check if radius search found valid results (not just dummy -1 index)
    bool has_valid_results = !nearest_points.empty() &&
                             nearest_points[0].first != static_cast<size_t>(-1);

    if (!has_valid_results) {
      // Fallback to nearest neighbor WITHOUT radius constraint
      const size_t single_neighbor = 1;
      const CoordinateType no_radius = 0.0; // 0.0 means no radius constraint
      nearest_points = find_nearest_point(gll_point, point_data,
                                          &single_neighbor, &no_radius);
    }

    // Check if we still have zero points after fallback
    if (nearest_points.empty()) {
      if (elem_idx < 1) {
#pragma omp critical
        LOG(WARNING) << "WARNING: Element " << elem_idx
                     << ") has ZERO neighbors even after fallback!";
        element_errors[elem_idx] = MB_FAILURE;
      }
      continue; // Skip this GLL node
    }

    // Pre-compute inverse distance weights (vectorizable)
    thread_local std::vector<CoordinateType> inv_weights;
    inv_weights.resize(nearest_points.size(), 1.0);

    // Process each scalar variable
    for (const auto &var_name : m_config.scalar_var_names) {
      double weighted_sum = 0.0;
      double weight_sum = 0.0;

      // Check if it's a double variable
      auto var_it = point_data.d_scalar_variables.find(var_name);
      if (!m_config.is_usgs_format &&
          var_it != point_data.d_scalar_variables.end()) {
        const auto &values = var_it->second;

        // Vectorized weighted sum computation
        for (size_t k = 0; k < nearest_points.size(); ++k) {
          size_t pt_idx = nearest_points[k].first;
          if (pt_idx < values.size()) {
            weighted_sum += inv_weights[k] * values[pt_idx];
            weight_sum += inv_weights[k];
          }
        }
      } else {
        // Check if it's an integer variable
        auto ivar_it = point_data.i_scalar_variables.find(var_name);
        if (ivar_it != point_data.i_scalar_variables.end()) {
          const auto &values = ivar_it->second;

          // Vectorized weighted sum computation for integers
          for (size_t k = 0; k < nearest_points.size(); ++k) {
            size_t pt_idx = nearest_points[k].first;
            if (pt_idx < values.size()) {
              weighted_sum +=
                  inv_weights[k] * static_cast<double>(values[pt_idx]);
              weight_sum += inv_weights[k];
            }
          }
        }
      }

      // Add GLL-weighted contribution to element average
      if (fabs(weight_sum) > 0.0) {
        element_averages_d[var_name] = weighted_sum / weight_sum;
      }
    }

    // Normalize element averages and store in mesh data (thread-safe
    // writes) Use per-variable weights for proper normalization
    for (const auto &var_name : m_config.scalar_var_names) {
      auto it = element_averages_d.find(var_name);

      if (it != element_averages_d.end()) {
        auto var_it = point_data.d_scalar_variables.find(var_name);
        if (var_it != point_data.d_scalar_variables.end()) {
          m_mesh_data.d_scalar_fields[var_name][elem_idx] = it->second;
        } else {
          auto ivar_it = point_data.i_scalar_variables.find(var_name);
          if (ivar_it != point_data.i_scalar_variables.end()) {
            m_mesh_data.i_scalar_fields[var_name][elem_idx] =
                static_cast<int>(it->second);
          }
        }
      }
    }

  } // End of parallel region

  // Check for errors after parallel processing
  size_t error_count = 0;
  for (size_t i = 0; i < element_errors.size(); ++i) {
    if (element_errors[i] != MB_SUCCESS) {
      error_count++;
    }
  }

  if (error_count > 0) {
    LOG(ERROR) << "Warning: " << error_count
               << " elements failed processing out of "
               << m_mesh_data.elements.size() << " total elements";
  }

  // Thread-safe debugging output
  for (const auto &var_name : m_config.scalar_var_names) {
    std::array<double, 5> values;
    for (size_t i = 0; i < std::min(size_t(5), m_mesh_data.elements.size());
         ++i) {
      auto d_it = m_mesh_data.d_scalar_fields.find(var_name);
      auto i_it = m_mesh_data.i_scalar_fields.find(var_name);
      if (d_it != m_mesh_data.d_scalar_fields.end()) {
        values[i] = d_it->second[i];
      } else if (i_it != m_mesh_data.i_scalar_fields.end()) {
        values[i] = i_it->second[i];
      }
    }
    VLOG(2) << "Sample values for " << var_name << ": " << values;
  }

  return MB_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////

} // namespace moab

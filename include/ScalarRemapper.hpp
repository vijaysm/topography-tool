#ifndef SCALAR_REMAPPER_HPP
#define SCALAR_REMAPPER_HPP

#include "ParallelPointCloudReader.hpp"
#include "RegularGridLocator.hpp"
#include "moab/ErrorHandler.hpp"
#include "moab/Interface.hpp"
#include "nanoflann.hpp"
#include <array>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace moab {

// KD-tree functionality enabled using nanoflann for fast nearest neighbor
// queries

/**
 * @brief Point cloud adapter for nanoflann KD-tree
 */
struct PointCloudAdapter {
  // const std::vector<PointType>& latlon_points;
  std::vector<CoordinateType> points;

  PointCloudAdapter(const std::vector<PointType> &pts) //: latlon_points(pts)
  {
    points.resize(pts.size() * 3);
#pragma omp parallel for shared(pts, points)
    for (size_t i = 0; i < pts.size(); ++i) {
      size_t offset = i * 3;
      RLLtoXYZ_Deg(pts[i][0], pts[i][1], points.data() + offset);
    }
  }

  // Must return the number of data points
  inline size_t kdtree_get_point_count() const { return points.size() / 3; }

  // Returns the dim'th component of the idx'th point in the class
  inline CoordinateType kdtree_get_pt(const size_t idx,
                                      const size_t dim) const {
    return points[idx * 3 + dim];
  }

  // Optional bounding-box computation: return false to default to a standard
  // bbox computation loop
  template <class BBOX> bool kdtree_get_bbox(BBOX & /* bb */) const {
    return false;
  }
};

// Define the KD-tree type with explicit template parameters
typedef nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<CoordinateType, PointCloudAdapter,
                                 CoordinateType, size_t>,
    // nanoflann::SO2_Adaptor<CoordinateType, PointCloudAdapter, CoordinateType,
    // size_t>,
    PointCloudAdapter, 3, /* dim */
    size_t                /* IndexType */
    >
    KDTree;

/**
 * @brief Abstract base class for remapping scalar data from point clouds to
 * mesh elements
 *
 * This class provides a general interface for different remapping methods
 * (nearest neighbor, inverse distance weighting, etc.) to map scalar data from
 * NetCDF point clouds to H5M mesh element centroids. OpenMP parallelism is
 * exploited for multicore performance.
 */
class ScalarRemapper {
public:
  struct RemapConfig {
    std::vector<std::string> scalar_var_names; // Variables to remap
    bool is_usgs_format = false; // Use USGS format for point cloud
    bool use_kd_tree = false;    // Use KD-tree (true) or RegularGridLocator
                                 // (false) for USGS format
    DistanceMetric distance_metric = HAVERSINE; // Distance metric for
                                                // RegularGridLocator
    // const ParallelPointCloudReader::PointCloudMeshView*
    // target_point_cloud_view = nullptr;
    bool reuse_source_mesh = false; // Reuse source mesh as target so that
                                    // we can smoothen the data
    double user_search_area = 0.0;  // User-specified search area for smoothing
  };

  struct MeshData {

    size_t size() const { return elements.size(); }

    const PointType3D &centroid(size_t index) const { return centroids[index]; }

    std::vector<EntityHandle> elements; // Local mesh elements
    std::vector<PointType3D> centroids; // Element centroids

  public:
    std::unordered_map<std::string, std::vector<double>>
        d_scalar_fields; // Remapped data
    std::unordered_map<std::string, std::vector<int>>
        i_scalar_fields; // Remapped data
  };

  // Spectral element projection parameters: immutable for now
  const int spectral_order = 4;     // Spectral element order (nP)
  const bool continuous_gll = true; // Use continuous GLL nodes
  const bool apply_bubble_correction =
      false; // Apply bubble correction for mass conservation
protected:
  Interface *m_interface;
  EntityHandle m_mesh_set;
  RemapConfig m_config;
  MeshData m_mesh_data;
  bool m_target_is_spectral;
  bool m_self_remapping = false;
  double m_user_search_area = 0.0; // User-specified search area for smoothing

public:
  ScalarRemapper(Interface *interface, EntityHandle mesh_set);
  virtual ~ScalarRemapper() = default;

  // Configuration
  ErrorCode configure(const RemapConfig &config);

  // Main remapping interface
  ErrorCode
  remap_scalars(const ParallelPointCloudReader::PointData &point_data);

  // Access results
  const MeshData &get_mesh_data() const { return m_mesh_data; }
  ErrorCode write_to_tags(const std::string &tag_prefix = "");

protected:
  // Abstract method to be implemented by derived classes
  virtual ErrorCode
  perform_remapping(const ParallelPointCloudReader::PointData &point_data) = 0;

  // Abstract method to be implemented by derived classes
  virtual ErrorCode
  perform_self_remapping(const ParallelPointCloudReader::PointData &point_data);
  ErrorCode smoothen_field_constant_area_averaging(
      const ParallelPointCloudReader::PointData &point_data,
      double constant_area);

  // Utility methods
  ErrorCode extract_mesh_centroids();
  ErrorCode compute_element_centroid(EntityHandle element,
                                     PointType3D &centroid);

  // Validation and statistics
  ErrorCode validate_remapping_results();
  void print_remapping_statistics();

  std::vector<nanoflann::ResultItem<size_t, CoordinateType>>
  find_nearest_point(const PointType3D &target_point,
                     const ParallelPointCloudReader::PointData &point_data,
                     const size_t *max_neighbors = nullptr,
                     const CoordinateType *search_radius = nullptr);

  // Spatial query members: KD-tree or RegularGridLocator
  std::unique_ptr<PointCloudAdapter> m_adapter;
  std::unique_ptr<KDTree> m_kdtree;
  std::unique_ptr<RegularGridLocator> m_grid_locator;
  bool m_kdtree_built;
  bool m_grid_locator_built;

  ErrorCode build_kdtree(const ParallelPointCloudReader::PointData &point_data);
  ErrorCode
  build_grid_locator(const ParallelPointCloudReader::PointData &point_data);

  // Spatial search helper methods
  std::vector<nanoflann::ResultItem<size_t, CoordinateType>>
  linear_search_fallback(const PointType3D &target_point,
                         const ParallelPointCloudReader::PointData &point_data,
                         CoordinateType search_radius);

  std::vector<nanoflann::ResultItem<size_t, CoordinateType>>
  kdtree_search(const PointType3D &target_point, CoordinateType search_radius,
                size_t max_neighbors);
};

/**
 * @brief Nearest neighbor remapping implementation
 *
 * Maps each mesh element centroid to the value of the nearest point cloud
 * point. This is the simplest and fastest remapping method, suitable for dense
 * point clouds.
 */
class NearestNeighborRemapper : public ScalarRemapper {
public:
  NearestNeighborRemapper(Interface *interface, EntityHandle mesh_set);
  ~NearestNeighborRemapper();

protected:
  ErrorCode perform_remapping(
      const ParallelPointCloudReader::PointData &point_data) override;
};

/**
 * @brief Point cloud to spectral element projection remapper
 *
 * Implements the LinearRemapFVtoGLL_Averaged algorithm for projecting point
 * cloud data onto spectral element meshes using GLL quadrature points and
 * KD-tree spatial searches. NOTE: OpenMP parallelization is already implemented
 * in this class.
 */
class PCDiskAveragedProjectionRemapper : public ScalarRemapper {
public:
  PCDiskAveragedProjectionRemapper(Interface *interface, EntityHandle mesh_set);

protected:
  ErrorCode perform_remapping(
      const ParallelPointCloudReader::PointData &point_data) override;

private:
  // Spectral element mesh validation
  ErrorCode validate_quadrilateral_mesh();

  // Point cloud to spectral element projection
  ErrorCode project_point_cloud_to_spectral_elements(
      const ParallelPointCloudReader::PointData &point_data);

  // Point cloud to target projection using disk-area averaging
  ErrorCode project_point_cloud_with_area_averaging(
      const ParallelPointCloudReader::PointData &point_data);
};

/**
 * @brief Factory class for creating remapping instances
 */
class RemapperFactory {
public:
  enum RemapMethod {
    ALG_DISKAVERAGE, // Default: Point cloud disk averaged projection
    ALG_NEAREST_NEIGHBOR
  };

  static std::unique_ptr<ScalarRemapper> create_remapper(RemapMethod method,
                                                         Interface *interface,
                                                         EntityHandle mesh_set);
};

} // namespace moab

#endif // SCALAR_REMAPPER_HPP

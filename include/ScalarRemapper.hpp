#ifndef SCALAR_REMAPPER_HPP
#define SCALAR_REMAPPER_HPP

#include "ParallelPointCloudReader.hpp"
#include "moab/Interface.hpp"
#include "moab/ParallelComm.hpp"
#include "moab/ErrorHandler.hpp"
#include "nanoflann.hpp"
#include <array>
#include <vector>
#include <map>
#include <string>
#include <memory>

namespace moab {

// KD-tree functionality enabled using nanoflann for fast nearest neighbor queries

/**
 * @brief Point cloud adapter for nanoflann KD-tree
 */
struct PointCloudAdapter {
    // const std::vector<ParallelPointCloudReader::PointType>& latlon_points;
    std::vector<ParallelPointCloudReader::CoordinateType> points;

    PointCloudAdapter(const std::vector<ParallelPointCloudReader::PointType>& pts) //: latlon_points(pts)
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
    inline ParallelPointCloudReader::CoordinateType kdtree_get_pt(const size_t idx, const size_t dim) const {
        return points[idx * 3 + dim];
    }

    // Optional bounding-box computation: return false to default to a standard bbox computation loop
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }
};

// Define the KD-tree type with explicit template parameters
typedef nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<ParallelPointCloudReader::CoordinateType, PointCloudAdapter, ParallelPointCloudReader::CoordinateType, size_t>,
    // nanoflann::SO2_Adaptor<ParallelPointCloudReader::CoordinateType, PointCloudAdapter, ParallelPointCloudReader::CoordinateType, size_t>,
    PointCloudAdapter,
    3, /* dim */
    size_t /* IndexType */
> KDTree;

/**
 * @brief Abstract base class for remapping scalar data from point clouds to mesh elements
 *
 * This class provides a general interface for different remapping methods (nearest neighbor,
 * inverse distance weighting, etc.) to map scalar data from NetCDF point clouds to H5M mesh
 * element centroids. All operations are designed to be parallel without MPI communication
 * after the initial point cloud distribution.
 */
class ScalarRemapper {
public:
    struct RemapConfig {
        std::vector<std::string> scalar_var_names;  // Variables to remap
        ParallelPointCloudReader::CoordinateType search_radius = 0.0;                 // Search radius (0 = unlimited)
        int max_neighbors = 1;                      // Maximum neighbors to consider
        bool use_element_centroids = true;          // Use element centroids vs vertices
        bool normalize_weights = true;              // Normalize interpolation weights
        bool is_usgs_format = false;                 // Use USGS format for point cloud

        // Spectral element projection parameters
        int spectral_order = 4;                     // Spectral element order (nP)
        bool continuous_gll = true;                 // Use continuous GLL nodes
        bool apply_bubble_correction = false;       // Apply bubble correction for mass conservation
    };

    struct MeshData {
        std::vector<EntityHandle> elements;         // Local mesh elements
        std::vector<ParallelPointCloudReader::PointType3D> centroids; // Element centroids
        std::unordered_map<std::string, std::vector<double>> d_scalar_fields; // Remapped data
        std::unordered_map<std::string, std::vector<int>> i_scalar_fields; // Remapped data
    };

protected:
    Interface* m_interface;
    ParallelComm* m_pcomm;
    EntityHandle m_mesh_set;
    RemapConfig m_config;
    MeshData m_mesh_data;

public:
    ScalarRemapper(Interface* interface, ParallelComm* pcomm, EntityHandle mesh_set);
    virtual ~ScalarRemapper() = default;

    // Configuration
    ErrorCode configure(const RemapConfig& config);

    // Main remapping interface
    ErrorCode remap_scalars(const ParallelPointCloudReader::PointData& point_data);

    // Access results
    const MeshData& get_mesh_data() const { return m_mesh_data; }
    ErrorCode write_to_tags(const std::string& tag_prefix = "");

protected:
    // Abstract method to be implemented by derived classes
    virtual ErrorCode perform_remapping(const ParallelPointCloudReader::PointData& point_data) = 0;

    // Utility methods
    ErrorCode extract_mesh_centroids();
    ErrorCode compute_element_centroid(EntityHandle element, ParallelPointCloudReader::PointType3D& centroid);

    // Validation and statistics
    ErrorCode validate_remapping_results();
    void print_remapping_statistics();

    std::vector<nanoflann::ResultItem<size_t, ParallelPointCloudReader::CoordinateType>>  find_nearest_point(const ParallelPointCloudReader::PointType3D& target_point,
                          const ParallelPointCloudReader::PointData& point_data, const size_t *max_neighbors = nullptr, const ParallelPointCloudReader::CoordinateType* search_radius = nullptr);

    // KD-tree members for fast nearest neighbor queries
    std::unique_ptr<PointCloudAdapter> m_adapter;
    std::unique_ptr<KDTree> m_kdtree;
    bool m_kdtree_built;

    ErrorCode build_kdtree(const ParallelPointCloudReader::PointData& point_data);
};

/**
 * @brief Nearest neighbor remapping implementation
 *
 * Maps each mesh element centroid to the value of the nearest point cloud point.
 * This is the simplest and fastest remapping method, suitable for dense point clouds.
 */
class NearestNeighborRemapper : public ScalarRemapper {
public:
    NearestNeighborRemapper(Interface* interface, ParallelComm* pcomm, EntityHandle mesh_set);
    ~NearestNeighborRemapper();

protected:
    ErrorCode perform_remapping(const ParallelPointCloudReader::PointData& point_data) override;

private:
    // size_t find_nearest_point(const ParallelPointCloudReader::PointType3D& target_point,
    //                       const ParallelPointCloudReader::PointData& point_data);

    // // KD-tree members for fast nearest neighbor queries
    // std::unique_ptr<PointCloudAdapter> m_adapter;
    // std::unique_ptr<KDTree> m_kdtree;
    // bool m_kdtree_built;

    // ErrorCode build_kdtree(const ParallelPointCloudReader::PointData& point_data);
};

/**
 * @brief Inverse distance weighted remapping implementation
 *
 * Maps each mesh element centroid using inverse distance weighted interpolation
 * from multiple nearby point cloud points. More accurate but computationally expensive.
 */
class InverseDistanceRemapper : public ScalarRemapper {
public:
    InverseDistanceRemapper(Interface* interface, ParallelComm* pcomm, EntityHandle mesh_set);

protected:
    ErrorCode perform_remapping(const ParallelPointCloudReader::PointData& point_data) override;

private:
    struct WeightedPoint {
        int index;
        double weight;
    };

    std::vector<WeightedPoint> find_weighted_neighbors(
        const ParallelPointCloudReader::PointType3D& target_point,
        const ParallelPointCloudReader::PointData& point_data);

    ParallelPointCloudReader::CoordinateType compute_inverse_distance_weight(ParallelPointCloudReader::CoordinateType distance, ParallelPointCloudReader::CoordinateType power = 2.0);
};

/**
 * @brief Point cloud to spectral element projection remapper
 *
 * Implements the LinearRemapFVtoGLL_Averaged algorithm for projecting point cloud data
 * onto spectral element meshes using GLL quadrature points and KD-tree spatial searches.
 */
class PCSpectralProjectionRemapper : public ScalarRemapper {
public:
    PCSpectralProjectionRemapper(Interface* interface, ParallelComm* pcomm, EntityHandle mesh_set);

protected:
    ErrorCode perform_remapping(const ParallelPointCloudReader::PointData& point_data) override;

private:
    // Spectral element mesh validation
    ErrorCode validate_quadrilateral_mesh();

    // Point cloud to spectral element projection
    ErrorCode project_point_cloud_to_spectral_elements(
        const ParallelPointCloudReader::PointData& point_data);
};

/**
 * @brief Factory class for creating remapping instances
 */
class RemapperFactory {
public:
    enum RemapMethod {
        PC_AVERAGED_SPECTRAL_PROJECTION,  // Default: Point cloud to spectral element projection
        NEAREST_NEIGHBOR,
        INVERSE_DISTANCE,
        BILINEAR,
        CUBIC_SPLINE
    };

    static std::unique_ptr<ScalarRemapper> create_remapper(
        RemapMethod method,
        Interface* interface,
        ParallelComm* pcomm,
        EntityHandle mesh_set);
};

} // namespace moab

#endif // SCALAR_REMAPPER_HPP

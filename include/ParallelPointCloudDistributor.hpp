#ifndef PARALLEL_POINT_CLOUD_DISTRIBUTOR_HPP
#define PARALLEL_POINT_CLOUD_DISTRIBUTOR_HPP

#include "ParallelPointCloudReader.hpp"
#include "moab/Interface.hpp"
#include "moab/gs.hpp"
#include <vector>
#include <map>

namespace moab {

/**
 * @brief Crystal router-based point redistribution for efficient spatial communication
 *
 * Uses MOAB's crystal router with TupleList to efficiently redistribute NetCDF point cloud
 * data based on mesh bounding boxes. Transfers coordinates and all scalar variables together
 * in one bulk communication operation.
 */
class ParallelPointCloudDistributor {
public:
    struct DistributionStats {
        size_t points_sent = 0;
        size_t points_received = 0;
        size_t total_transfers = 0;
        double communication_time_ms = 0.0;
        std::vector<size_t> points_per_rank;
    };

    struct CrystalRouterConfig {
        bool allow_multiple_ownership = true;  // Points can belong to multiple ranks if bboxes overlap
        double bbox_expansion_factor = 0.0;    // Expand bboxes by this factor for better coverage
        size_t max_tuple_list_size = 1000000;  // Maximum tuples per crystal router call
        bool enable_statistics = true;
        bool verbose = false;
    };

private:
    // Interface* m_interface;
    ParallelComm* m_pcomm;
    CrystalRouterConfig m_config;
    bool m_is_usgs_format;

    // Store variable names for consistent packing/unpacking
    std::vector<std::string> m_variable_names;

    // Crystal router communication data
    struct PointTuple {
        int target_rank;     // Destination rank
        double lon, lat;     // Geographic coordinates
        double x, y, z;      // Cartesian coordinates
        // Scalar variables will be packed separately
    };

public:
    ParallelPointCloudDistributor(Interface* interface, ParallelComm* pcomm, bool is_usgs_format);

    ErrorCode configure(const CrystalRouterConfig& config);

    /**
     * @brief Redistribute points using crystal router based on bounding box ownership
     *
     * @param input_points Local point cloud data from NetCDF reading
     * @param all_bboxes Bounding boxes from all ranks (gathered via MPI_Allgather)
     * @param output_points Redistributed points owned by this rank
     * @param stats Communication and redistribution statistics
     */
    ErrorCode redistribute_points_crystal_router(
        const ParallelPointCloudReader::PointData& input_points,
        const std::vector<ParallelPointCloudReader::BoundingBox>& all_bboxes,
        ParallelPointCloudReader::PointData& output_points,
        DistributionStats& stats);

    ErrorCode redistribute_points_batched(
        const ParallelPointCloudReader::PointData& input_points,
        const std::vector<ParallelPointCloudReader::BoundingBox>& all_bboxes,
        ParallelPointCloudReader::PointData& output_points,
        DistributionStats& stats);

private:
    /**
     * @brief Determine which ranks should receive each point based on bounding box containment
     */
    ErrorCode assign_points_to_ranks(
        const ParallelPointCloudReader::PointData& input_points,
        const std::vector<ParallelPointCloudReader::BoundingBox>& all_bboxes,
        std::vector<std::vector<int>>& point_to_ranks);

    /**
     * @brief Pack point coordinates into TupleList for crystal router communication
     */
    ErrorCode pack_coordinates_to_tuplelist(
        const ParallelPointCloudReader::PointData& input_points,
        const std::vector<std::vector<int>>& point_to_ranks,
        TupleList& coordinate_tuples);

    /**
     * @brief Pack scalar variables into TupleList for crystal router communication
     */
    template<typename T>
    ErrorCode pack_scalars_to_tuplelist(
        const std::unordered_map<std::string, std::vector<T>>& scalar_variables,
        const std::vector<std::vector<int>>& point_to_ranks,
        TupleList& scalar_tuples);

    /**
     * @brief Unpack received coordinates from TupleList after crystal router communication
     */
    ErrorCode unpack_coordinates_from_tuplelist(
        const TupleList& coordinate_tuples,
        ParallelPointCloudReader::PointData& output_points);

    /**
     * @brief Unpack received scalar variables from TupleList after crystal router communication
     */
    template<typename T>
    ErrorCode unpack_scalars_from_tuplelist(
        const TupleList& scalar_tuples,
        std::unordered_map<std::string, std::vector<T>>& scalar_variables);

    /**
     * @brief Check if a point (lon, lat) is contained within a bounding box
     */
    bool point_in_bbox(double lon, double lat,
                      const ParallelPointCloudReader::BoundingBox& bbox) const;

    /**
     * @brief Expand bounding box by configured factor for better point coverage
     */
    ParallelPointCloudReader::BoundingBox expand_bbox(
        const ParallelPointCloudReader::BoundingBox& bbox) const;
};

} // namespace moab

#endif // PARALLEL_POINT_CLOUD_DISTRIBUTOR_HPP

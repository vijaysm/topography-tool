#include "ParallelPointCloudDistributor.hpp"
#include "moab/TupleList.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace moab {

ParallelPointCloudDistributor::ParallelPointCloudDistributor(Interface* interface, ParallelComm* pcomm, bool usgs_format)
    : m_interface(interface), m_pcomm(pcomm), m_is_usgs_format(usgs_format) {
    // Set default configuration
    m_config.allow_multiple_ownership = true;
    m_config.bbox_expansion_factor = 0.0;
    m_config.max_tuple_list_size = 1000000;
    m_config.enable_statistics = true;
    // m_config.verbose = m_pcomm->rank() ? false : true;
}

ErrorCode ParallelPointCloudDistributor::configure(const CrystalRouterConfig& config) {
    m_config = config;
    return MB_SUCCESS;
}

ErrorCode ParallelPointCloudDistributor::redistribute_points_crystal_router(
    const ParallelPointCloudReader::PointData& input_points,
    const std::vector<ParallelPointCloudReader::BoundingBox>& all_bboxes,
    ParallelPointCloudReader::PointData& output_points,
    DistributionStats& stats) {

    if (m_pcomm->rank() == 0 && m_config.enable_statistics) {
        std::cout << "Starting crystal router redistribution with "
                  << input_points.lonlat_coordinates.size() << " input points" << std::endl;
    }

    // Step 1: Determine which ranks should receive each point
    if (m_config.verbose) std::cout << "Assigning local points to ranks based on bounding boxes of QUAD mesh" << std::endl;
    std::vector<std::vector<int>> point_to_ranks;
    MB_CHK_ERR(assign_points_to_ranks(input_points, all_bboxes, point_to_ranks));

    // Step 2: Pack and transfer coordinates using crystal router
    TupleList coordinate_tuples;
    if (m_config.verbose) std::cout << "Packing coordinates to tuplelist for communication ..." << std::endl;
    MB_CHK_ERR(pack_coordinates_to_tuplelist(input_points, point_to_ranks, coordinate_tuples));

    if (m_config.verbose) std::cout << "Performing crystal router communication for coordinates ..." << std::endl;
    // Perform crystal router communication for coordinates
    m_pcomm->proc_config().crystal_router()->gs_transfer(1, coordinate_tuples, 0);

    if (m_config.verbose) std::cout << "Unpacking received coordinates ..." << std::endl;
    // Unpack received coordinates
    output_points.lonlat_coordinates.clear();
    MB_CHK_ERR(unpack_coordinates_from_tuplelist(coordinate_tuples, output_points));

    // Step 3: Transfer all scalar variables using crystal router
    output_points.d_scalar_variables.clear();
    output_points.i_scalar_variables.clear();
    {
        TupleList scalar_tuples;
        if (m_config.verbose) std::cout << "Packing scalar variables to tuplelist for communication ..." << std::endl;
        if (m_is_usgs_format) {
            MB_CHK_ERR(pack_scalars_to_tuplelist(input_points.i_scalar_variables, point_to_ranks, scalar_tuples));
        } else {
            MB_CHK_ERR(pack_scalars_to_tuplelist(input_points.d_scalar_variables, point_to_ranks, scalar_tuples));
        }

        if (m_config.verbose) std::cout << "Performing crystal router communication for scalar variables ..." << std::endl;
        // Perform crystal router communication for this scalar variable
        m_pcomm->proc_config().crystal_router()->gs_transfer(1, scalar_tuples, 0);

        if (m_config.verbose) std::cout << "Unpacking received scalar variables ..." << std::endl;
        // Unpack received scalar data
        if (m_is_usgs_format) {
            MB_CHK_ERR(unpack_scalars_from_tuplelist(scalar_tuples, output_points.i_scalar_variables));
        } else {
            MB_CHK_ERR(unpack_scalars_from_tuplelist(scalar_tuples, output_points.d_scalar_variables));
        }
    }

    // Compute statistics
    stats.points_received = output_points.lonlat_coordinates.size();
    // stats.communication_time_ms = 0.0; // Timing temporarily disabled due to namespace conflicts

    // Count total transfers (points can be sent to multiple ranks)
    stats.total_transfers = 0;
    for (const auto& ranks : point_to_ranks) {
        stats.total_transfers += ranks.size();
    }
    stats.points_sent = input_points.lonlat_coordinates.size();

    if (m_config.verbose && m_config.enable_statistics) {
        std::cout << "Crystal router redistribution completed" << std::endl;
        std::cout << "Total transfers: " << stats.total_transfers
                  << ", Points received: " << stats.points_received << std::endl;
    }

    return MB_SUCCESS;
}

ErrorCode ParallelPointCloudDistributor::assign_points_to_ranks(
    const ParallelPointCloudReader::PointData& input_points,
    const std::vector<ParallelPointCloudReader::BoundingBox>& all_bboxes,
    std::vector<std::vector<int>>& point_to_ranks) {

    point_to_ranks.clear();
    point_to_ranks.resize(input_points.lonlat_coordinates.size());

    size_t total_assignments = 0;

    // For each point, find all ranks whose bounding boxes contain it
    for (size_t i = 0; i < input_points.lonlat_coordinates.size(); ++i) {
        const auto& coord = input_points.lonlat_coordinates[i];

        auto& pointowners = point_to_ranks[i];

        // Check containment in each rank's bounding box
        for (size_t irank = 0; irank < all_bboxes.size(); ++irank) {
            // all of our bounding boxes are already expanded based on the factor.
            // ParallelPointCloudReader::BoundingBox bbox = expand_bbox(all_bboxes[irank]);
            const ParallelPointCloudReader::BoundingBox& bbox = all_bboxes[irank];

            if (point_in_bbox(coord[0], coord[1], bbox)) {
                pointowners.push_back(irank);
                total_assignments++;
            }
        }

        // If point doesn't belong to any rank, assign to current rank as fallback
        if (pointowners.empty()) {
            pointowners.push_back(m_pcomm->rank());
            total_assignments++;
        }
    }

    // Check for excessive point duplication
    size_t num_points = input_points.lonlat_coordinates.size();
    double duplication_factor = static_cast<double>(total_assignments) / num_points;

    if (m_config.verbose) std::cout << "Rank " << m_pcomm->rank() << " point assignment: " << num_points
              << " points -> " << total_assignments << " assignments (factor: "
              << duplication_factor << ")" << std::endl;

    if (duplication_factor > 5.0) {
        std::cerr << "WARNING: Rank " << m_pcomm->rank() << " has excessive point duplication factor: "
                  << duplication_factor << std::endl;
        std::cerr << "This may cause MPI buffer overflow. Consider using root-based distribution instead." << std::endl;
    }

    return MB_SUCCESS;
}

ErrorCode ParallelPointCloudDistributor::pack_coordinates_to_tuplelist(
    const ParallelPointCloudReader::PointData& input_points,
    const std::vector<std::vector<int>>& point_to_ranks,
    TupleList& coordinate_tuples) {

    // Count total tuples needed
    size_t total_tuples = 0;
    for (const auto& ranks : point_to_ranks) {
        total_tuples += ranks.size();
    }

    // Check for potential overflow - TupleList uses int internally
    const size_t MAX_SAFE_TUPLES = std::numeric_limits<int>::max() / 8; // Conservative limit
    if (total_tuples > MAX_SAFE_TUPLES) {
        std::cerr << "ERROR: Rank " << m_pcomm->rank() << " - total_tuples (" << total_tuples
                  << ") exceeds safe limit (" << MAX_SAFE_TUPLES << ")" << std::endl;
        std::cerr << "This indicates excessive point duplication due to overlapping bounding boxes." << std::endl;
        std::cerr << "Consider reducing buffer_factor or using fewer MPI ranks." << std::endl;
        return MB_FAILURE;
    }

    if (m_config.verbose) std::cout << "Rank " << m_pcomm->rank() << " packing " << total_tuples << " coordinate tuples" << std::endl;

    // Initialize TupleList: 1 integer (target rank) + 2 reals (lon, lat)
    coordinate_tuples.initialize(1, 0, 0, 2, total_tuples);
    coordinate_tuples.enableWriteAccess();

    size_t tuple_idx = 0;
    for (size_t point_idx = 0; point_idx < input_points.lonlat_coordinates.size(); ++point_idx) {
        const auto& coord = input_points.lonlat_coordinates[point_idx];

        for (int target_rank : point_to_ranks[point_idx]) {
            // Pack: target_rank, lon, lat, x, y, z
            coordinate_tuples.vi_wr[tuple_idx] = target_rank;
            coordinate_tuples.vr_wr[tuple_idx * 2 + 0] = coord[0]; // lon or x
            coordinate_tuples.vr_wr[tuple_idx * 2 + 1] = coord[1]; // lat or y
            tuple_idx++;
        }
    }

    coordinate_tuples.set_n(tuple_idx);
    return MB_SUCCESS;
}

template<typename T>
ErrorCode ParallelPointCloudDistributor::pack_scalars_to_tuplelist(
    // const ParallelPointCloudReader::PointData& input_points,
    const std::unordered_map<std::string, std::vector<T>>& scalar_variables,
    const std::vector<std::vector<int>>& point_to_ranks,
    TupleList& scalar_tuples) {

    // auto scalar_variables = (m_is_usgs_format ? input_points.i_scalar_variables : input_points.d_scalar_variables);
    const size_t nscalarbytes = sizeof(T);

    // Store variable names in member variable for consistent packing/unpacking
    m_variable_names.clear();
    m_variable_names.reserve(scalar_variables.size());
    for(auto it = scalar_variables.begin(); it != scalar_variables.end(); ++it) {
        if (m_pcomm->rank() == 0) std::cout << "Adding variable to list: -" << it->first << "-" << std::endl;
        if (it->first.size() > 0) {
            m_variable_names.push_back(it->first);
        }
    }

    // Count total tuples needed
    size_t total_tuples = 0;
    for (const auto& ranks : point_to_ranks) {
        total_tuples += ranks.size();
    }

    const size_t nvars = m_variable_names.size();

    // Check for potential overflow - TupleList uses int internally
    const size_t MAX_SAFE_TUPLES = std::numeric_limits<int>::max() / (nvars * nscalarbytes + sizeof(int)); // Account for nvars reals per tuple
    if (total_tuples > MAX_SAFE_TUPLES) {
        std::cerr << "ERROR: Rank " << m_pcomm->rank() << " - total_tuples (" << total_tuples
                  << ") with " << nvars << " variables exceeds safe limit (" << MAX_SAFE_TUPLES << ")" << std::endl;
        std::cerr << "This indicates excessive point duplication due to overlapping bounding boxes." << std::endl;
        std::cerr << "Consider reducing buffer_factor or using fewer MPI ranks." << std::endl;
        return MB_FAILURE;
    }

    if (m_config.verbose) std::cout << "Rank " << m_pcomm->rank() << " packing " << total_tuples << " scalar tuples with " << nvars << " variables" << std::endl;

    // Initialize TupleList: 1 integer (target rank) + N real (scalar value)
    if (m_is_usgs_format)
        scalar_tuples.initialize(1+nvars, 0, 0, 0, total_tuples);
    else
        scalar_tuples.initialize(1, 0, 0, nvars, total_tuples);
    scalar_tuples.enableWriteAccess();

    if (nvars == 0) {
        scalar_tuples.set_n(0);
        return MB_SUCCESS;
    }

    auto var_it = scalar_variables.find(m_variable_names[0]);
    if (var_it == scalar_variables.end()) {
        return MB_FAILURE;
    }

    const auto& first_scalar_data = var_it->second;

    std::vector<double> minmax(2*nvars, 0.0);
    std::vector<double> point_scalars(nvars, 0.0);

    size_t tuple_idx = 0;
    for (size_t point_idx = 0; point_idx < first_scalar_data.size(); ++point_idx) {
        for (size_t var_idx = 0; var_idx < nvars; ++var_idx) {
            auto var_it = scalar_variables.find(m_variable_names[var_idx]);
            if (var_it == scalar_variables.end()) {
                std::cout << "DID NOT FIND VARIABLE:: " << m_variable_names[var_idx] << std::endl;
                return MB_FAILURE;
                // continue;
            }
            const auto& scalar_data = var_it->second;
            point_scalars[var_idx] = scalar_data[point_idx];
            if (point_idx == 0) {
                minmax[var_idx] = point_scalars[var_idx];
                minmax[var_idx + nvars] = point_scalars[var_idx];
            }
            minmax[var_idx] = std::min(minmax[var_idx], point_scalars[var_idx]);
            minmax[var_idx + nvars] = std::max(minmax[var_idx + nvars], point_scalars[var_idx]);
        }
        for (int target_rank : point_to_ranks[point_idx]) {
            // Pack: target_rank, scalar_value
            const size_t offset = tuple_idx * nvars;
            if (m_is_usgs_format) {
                scalar_tuples.vi_wr[tuple_idx] = target_rank;
                for (size_t var_idx = 0; var_idx < nvars; ++var_idx)
                    scalar_tuples.vi_wr[offset + var_idx + 1] = point_scalars[var_idx];
            } else {
                scalar_tuples.vi_wr[tuple_idx] = target_rank;
                for (size_t var_idx = 0; var_idx < nvars; ++var_idx)
                    scalar_tuples.vr_wr[offset + var_idx] = point_scalars[var_idx];
            }
            tuple_idx++;
        }
    }

    if (m_pcomm->rank() == 0 && m_config.enable_statistics) {
        for (size_t var_idx = 0; var_idx < nvars; ++var_idx) {
            std::cout << "\tScalar min/max: " << m_variable_names[var_idx] << " min: " << minmax[var_idx] << " max: " << minmax[var_idx + nvars] << std::endl;
        }
    }

    scalar_tuples.set_n(tuple_idx);
    return MB_SUCCESS;
}

ErrorCode ParallelPointCloudDistributor::unpack_coordinates_from_tuplelist(
    const TupleList& coordinate_tuples,
    ParallelPointCloudReader::PointData& output_points) {

    const size_t n_tuples = coordinate_tuples.get_n();
    if (n_tuples == 0) {
        // std::cout << m_pcomm->rank() << ": Uh-oh. No coordinate tuples to unpack" << std::endl;
        return MB_SUCCESS;
    }
    // output_points.coordinates.reserve(n_tuples);
    output_points.lonlat_coordinates.reserve(n_tuples);

    for (size_t i = 0; i < n_tuples; ++i) {
        ParallelPointCloudReader::CoordinateType lon = coordinate_tuples.vr_rd[i * 2 + 0]; // lon
        ParallelPointCloudReader::CoordinateType lat = coordinate_tuples.vr_rd[i * 2 + 1]; // lat

        // Store lat/lon coordinates
        output_points.lonlat_coordinates.push_back({lon, lat});

        // Convert to Cartesian coordinates for mesh operations
        // std::array<ParallelPointCloudReader::CoordinateType, 3> cart_coord;
        // RLLtoXYZ_Deg(lon, lat, cart_coord);
        // output_points.coordinates.push_back(cart_coord);
    }

    return MB_SUCCESS;
}

template<typename T>
ErrorCode ParallelPointCloudDistributor::unpack_scalars_from_tuplelist(
    const TupleList& scalar_tuples,
    std::unordered_map<std::string, std::vector<T>>& scalar_variables) {

    const size_t n_tuples = scalar_tuples.get_n();
    if (n_tuples == 0) {
        // std::cout << m_pcomm->rank() << ": Uh-oh. No scalar tuples to unpack" << std::endl;
        return MB_SUCCESS;
    }

    // Use the stored variable names from the packing stage
    size_t nvars = m_variable_names.size();

    if (m_pcomm->rank() == 0) std::cout << "Rank " << m_pcomm->rank() << " unpacking " << n_tuples
              << " tuples with " << nvars << " variables each" << std::endl;

    if (nvars == 0) {
        std::cout << m_pcomm->rank() << ": Uh-oh. No variables to unpack" << std::endl;
        return MB_FAILURE;
    }

    // Unpack each variable using the stored variable names
    for (size_t var_idx = 0; var_idx < nvars; ++var_idx) {
        const std::string& var_name = m_variable_names[var_idx];
        std::vector<T>& scalar_data = scalar_variables[var_name];
        scalar_data.resize(n_tuples);

        T minmax[2] = {std::numeric_limits<T>::max(), std::numeric_limits<T>::min()};
        for (size_t i = 0; i < n_tuples; ++i) {
            if (m_is_usgs_format) {
                scalar_data[i] = scalar_tuples.vi_rd[i * nvars + var_idx];
            } else {
                scalar_data[i] = scalar_tuples.vr_rd[i * nvars + var_idx];
            }
            minmax[0] = std::min(minmax[0], scalar_data[i]);
            minmax[1] = std::max(minmax[1], scalar_data[i]);
        }

        if (m_pcomm->rank() == 0) std::cout << "Rank " << m_pcomm->rank() << " unpacked variable '" << var_name
                  << "' with " << scalar_data.size() << " values and min/max: " << minmax[0] << " / " << minmax[1] << std::endl;
    }

    return MB_SUCCESS;
}

bool ParallelPointCloudDistributor::point_in_bbox(double lon, double lat,
                                                 const ParallelPointCloudReader::BoundingBox& bbox) const {
    // Check longitude first and then latitude (straightforward)
    if (lon < bbox.min_coords[0] || lon > bbox.max_coords[0]) {
        return false;
    }
    if (lat < bbox.min_coords[1] || lat > bbox.max_coords[1]) {
        return false;
    }
    // if none of the above is false, then we are inside the box.
    return true;
}

ParallelPointCloudReader::BoundingBox ParallelPointCloudDistributor::expand_bbox(
    const ParallelPointCloudReader::BoundingBox& bbox) const {

    if (m_config.bbox_expansion_factor <= 0.0) {
        return bbox; // No expansion
    }

    ParallelPointCloudReader::BoundingBox expanded = bbox;

    for (int dim = 0; dim < 2; ++dim) {
        double range = bbox.max_coords[dim] - bbox.min_coords[dim];
        double expansion = range * m_config.bbox_expansion_factor;

        expanded.min_coords[dim] -= expansion;
        expanded.max_coords[dim] += expansion;
    }

    return expanded;
}

// remove_duplicate_points function removed - not needed for crystal router implementation


ErrorCode ParallelPointCloudDistributor::redistribute_points_batched(
    const ParallelPointCloudReader::PointData& input_points,
    const std::vector<ParallelPointCloudReader::BoundingBox>& all_bboxes,
    ParallelPointCloudReader::PointData& output_points,
    DistributionStats& stats) {

    if (m_config.verbose && m_config.enable_statistics) {
        std::cout << "Starting batched crystal router redistribution with "
                  << input_points.lonlat_coordinates.size() << " input points" << std::endl;
    }

    // Step 1: Determine which ranks should receive each point
    std::vector<std::vector<int>> point_to_ranks;
    MB_CHK_ERR(assign_points_to_ranks(input_points, all_bboxes, point_to_ranks));

    // Define a safe batch size to avoid MPI overflow
    // Each tuple: 1 int (4 bytes) + 2 doubles (16 bytes) = 20 bytes
    const size_t MAX_TUPLES_PER_BATCH = (1024 * 1024 * 1024); // 1GB limit / 20 bytes per tuple

    size_t num_points = input_points.lonlat_coordinates.size();
    size_t processed_points = 0;

    while (processed_points < num_points) {
        size_t current_batch_tuples = 0;
        size_t end_point_idx = processed_points;

        // Determine the range of points for the current batch
        while (end_point_idx < num_points && current_batch_tuples < MAX_TUPLES_PER_BATCH) {
            current_batch_tuples += point_to_ranks[end_point_idx].size();
            end_point_idx++;
        }

        if (m_config.verbose) {
            std::cout << "Processing batch: points " << processed_points << " to " << end_point_idx
                      << " (" << current_batch_tuples << " tuples)" << std::endl;
        }

        // Create a sub-vector for the current batch's rank assignments
        std::vector<std::vector<int>> batch_point_to_ranks(
            point_to_ranks.begin() + processed_points,
            point_to_ranks.begin() + end_point_idx
        );

        // Create a temporary PointData for the current batch
        ParallelPointCloudReader::PointData batch_input_points;
        batch_input_points.lonlat_coordinates.assign(
            input_points.lonlat_coordinates.begin() + processed_points,
            input_points.lonlat_coordinates.begin() + end_point_idx
        );
        if (m_config.verbose) std::cout << "Found " << input_points.d_scalar_variables.size() << " scalar variables" << std::endl;
        for (const auto& var_pair : input_points.d_scalar_variables) {
            const std::string& var_name = var_pair.first;
            const std::vector<double>& all_scalar_data = var_pair.second;
            batch_input_points.d_scalar_variables[var_name].assign(
                all_scalar_data.begin() + processed_points,
                all_scalar_data.begin() + end_point_idx
            );
        }

        // Pack and transfer coordinates for the batch
        TupleList coord_tuples;
        MB_CHK_ERR(pack_coordinates_to_tuplelist(batch_input_points, batch_point_to_ranks, coord_tuples));
        m_pcomm->proc_config().crystal_router()->gs_transfer(1, coord_tuples, 0);
        MB_CHK_ERR(unpack_coordinates_from_tuplelist(coord_tuples, output_points));

        TupleList scalar_tuples;
        if (m_is_usgs_format) {
            MB_CHK_ERR(pack_scalars_to_tuplelist(batch_input_points.i_scalar_variables, batch_point_to_ranks, scalar_tuples));
            m_pcomm->proc_config().crystal_router()->gs_transfer(1, scalar_tuples, 0);
            MB_CHK_ERR(unpack_scalars_from_tuplelist(scalar_tuples, output_points.i_scalar_variables));
        } else {
            MB_CHK_ERR(pack_scalars_to_tuplelist(batch_input_points.d_scalar_variables, batch_point_to_ranks, scalar_tuples));
            m_pcomm->proc_config().crystal_router()->gs_transfer(1, scalar_tuples, 0);
            MB_CHK_ERR(unpack_scalars_from_tuplelist(scalar_tuples, output_points.d_scalar_variables));
        }

        processed_points = end_point_idx;
        MPI_Barrier(m_pcomm->comm());
    }

    // Compute statistics
    stats.points_received = output_points.lonlat_coordinates.size();
    stats.communication_time_ms = 0.0; // Timing temporarily disabled due to namespace conflicts

    // Count total transfers (points can be sent to multiple ranks)
    stats.total_transfers = 0;
    for (const auto& ranks : point_to_ranks) {
        stats.total_transfers += ranks.size();
    }
    stats.points_sent = input_points.lonlat_coordinates.size();

    if (m_config.verbose && m_config.enable_statistics) {
        std::cout << "Crystal router redistribution completed" << std::endl;
        std::cout << "Total transfers: " << stats.total_transfers
                  << ", Points received: " << stats.points_received << std::endl;
    }
    return MB_SUCCESS;
}

} // namespace moab

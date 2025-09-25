#include "ScalarRemapper.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <limits>
#include <chrono>
#include <numeric>

namespace moab {

static double compute_distance(const ParallelPointCloudReader::PointType3D& p1, const ParallelPointCloudReader::PointType& p2) {

    ParallelPointCloudReader::PointType3D p2_3d;
    RLLtoXYZ_Deg(p2[0], p2[1], p2_3d);

    double dMag = 0.0;
    for (size_t i = 0; i < p1.size(); ++i) {
        dMag += (p1[i] - p2_3d[i])*(p1[i] - p2_3d[i]);
    }
    return std::sqrt(dMag);
}

ScalarRemapper::ScalarRemapper(Interface* interface, ParallelComm* pcomm, EntityHandle mesh_set)
    : m_interface(interface), m_pcomm(pcomm), m_mesh_set(mesh_set) {
}

ErrorCode ScalarRemapper::configure(const RemapConfig& config) {
    m_config = config;

    // Extract mesh element centroids
    MB_CHK_ERR(extract_mesh_centroids());

    if (m_pcomm->rank() == 0) {
        std::cout << "Configured scalar remapper with " << m_mesh_data.elements.size()
                  << " local mesh elements" << std::endl;
        std::cout << "Variables to remap: ";
        for (const auto& var : m_config.scalar_var_names) {
            std::cout << var << " ";
        }
        std::cout << std::endl;
    }

    return MB_SUCCESS;
}

ErrorCode ScalarRemapper::remap_scalars(const ParallelPointCloudReader::PointData& point_data) {

    if (m_pcomm->rank() == 0) {
        std::cout << "Starting scalar remapping with " << point_data.lonlat_coordinates.size()
                  << " point cloud points" << std::endl;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // Initialize scalar fields for output
    for (const auto& var_name : m_config.scalar_var_names) {
        if (m_config.is_usgs_format)
            m_mesh_data.i_scalar_fields[var_name].resize(m_mesh_data.elements.size(), 0);
        else
            m_mesh_data.d_scalar_fields[var_name].resize(m_mesh_data.elements.size(), 0.0);
    }

    std::cout.flush();

    // Perform the actual remapping (implemented by derived classes)
    MB_CHK_ERR(perform_remapping(point_data));

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    if (m_pcomm->rank() == 0) {
        std::cout << "Remapping completed in " << duration.count() << " ms" << std::endl;
    }

    // Validate results and print statistics
    MB_CHK_ERR(validate_remapping_results());
    print_remapping_statistics();

    // Copy results to output
    // mesh_data = m_mesh_data;

    return MB_SUCCESS;
}

ErrorCode ScalarRemapper::extract_mesh_centroids() {
    // Get all elements in the mesh set
    Range elements;
    ErrorCode rval = m_interface->get_entities_by_dimension(m_mesh_set, 2, elements); // 2D elements
    if (rval != MB_SUCCESS || elements.empty()) {
        rval = m_interface->get_entities_by_dimension(m_mesh_set, 3, elements); // 3D elements
        if (rval != MB_SUCCESS || elements.empty()) {
            std::cerr << "No 2D or 3D elements found in mesh" << std::endl;
            return MB_FAILURE;
        }
    }

    // Reserve space
    m_mesh_data.elements.clear();
    m_mesh_data.centroids.clear();
    m_mesh_data.elements.reserve(elements.size());
    m_mesh_data.centroids.reserve(elements.size());

    // Compute centroids for each element
    for (EntityHandle element : elements) {
        ParallelPointCloudReader::PointType3D centroid;
        MB_CHK_ERR(compute_element_centroid(element, centroid));

        m_mesh_data.elements.push_back(element);
        m_mesh_data.centroids.push_back(centroid);
    }

    if (m_pcomm->rank() == 0) {
        std::cout << "Extracted " << m_mesh_data.elements.size() << " element centroids" << std::endl;
        if (!m_mesh_data.centroids.empty()) {
            std::cout << "First centroid: (" << m_mesh_data.centroids[0][0] << ", "
                      << m_mesh_data.centroids[0][1] << ", " << m_mesh_data.centroids[0][2] << ")" << std::endl;
        }
    }

    return MB_SUCCESS;
}

ErrorCode ScalarRemapper::compute_element_centroid(EntityHandle element, ParallelPointCloudReader::PointType3D& centroid) {

    // Get element centroid directly (this is just average of vertex coordinates)
    double centroid_data[3];
    MB_CHK_ERR(m_interface->get_coords(&element, 1, centroid_data));

    ParallelPointCloudReader::CoordinateType dMag = 0.0;
    for (size_t i = 0; i < 3; ++i) {
        dMag += centroid_data[i]*centroid_data[i];
    }
    // project to unit sphere
    dMag = std::sqrt(dMag);
    for (size_t i = 0; i < 3; ++i) {
        centroid[i] = centroid_data[i]/dMag;
    }

    return MB_SUCCESS;
}

ErrorCode ScalarRemapper::validate_remapping_results() {
    // Check for any NaN or infinite values
    for (const auto& field_pair : m_mesh_data.d_scalar_fields) {
        const auto& field_data = field_pair.second;
        for (double value : field_data) {
            if (std::isnan(value) || std::isinf(value)) {
                std::cerr << "Invalid value detected in remapped field " << field_pair.first << std::endl;
                return MB_FAILURE;
            }
        }
    }
    // next check all integer field tags
    for (const auto& field_pair : m_mesh_data.i_scalar_fields) {
        const auto& field_data = field_pair.second;
        for (int value : field_data) {
            if (std::isnan(value) || std::isinf(value)) {
                std::cerr << "Invalid value detected in remapped field " << field_pair.first << std::endl;
                return MB_FAILURE;
            }
        }
    }

    return MB_SUCCESS;
}

void ScalarRemapper::print_remapping_statistics() {
    if (m_pcomm->rank() != 0) return;

    std::cout << "\n=== Remapping Statistics ===" << std::endl;

    for (const auto& field_pair : m_mesh_data.d_scalar_fields) {
        const std::string& var_name = field_pair.first;
        const auto& field_data = field_pair.second;

        if (field_data.empty()) continue;

        double min_val = *std::min_element(field_data.begin(), field_data.end());
        double max_val = *std::max_element(field_data.begin(), field_data.end());
        double sum = std::accumulate(field_data.begin(), field_data.end(), 0.0);
        double avg = sum / field_data.size();

        std::cout << var_name << " - Min: " << min_val << ", Max: " << max_val
                  << ", Avg: " << avg << " (on " << field_data.size() << " elements)" << std::endl;
    }

    for (const auto& field_pair : m_mesh_data.i_scalar_fields) {
        const std::string& var_name = field_pair.first;
        const auto& field_data = field_pair.second;

        if (field_data.empty()) continue;

        int min_val = *std::min_element(field_data.begin(), field_data.end());
        int max_val = *std::max_element(field_data.begin(), field_data.end());
        int sum = std::accumulate(field_data.begin(), field_data.end(), 0);
        double avg = sum / field_data.size();

        std::cout << var_name << " - Min: " << min_val << ", Max: " << max_val
                  << ", Avg: " << avg << " (on " << field_data.size() << " elements)" << std::endl;
    }
}

template<typename T>
static ErrorCode write_to_tag(moab::Interface* interface, const std::string& tag_name, const std::vector<EntityHandle>& elements, const std::vector<T>& data) {
    // Create or get the tag
    Tag tag;
    if (sizeof(T) == sizeof(double)) {
        MB_CHK_ERR(interface->tag_get_handle(tag_name.c_str(), 1, MB_TYPE_DOUBLE, tag,
                                                MB_TAG_DENSE | MB_TAG_CREAT));
    } else {
        MB_CHK_ERR(interface->tag_get_handle(tag_name.c_str(), 1, MB_TYPE_INTEGER, tag,
                                                MB_TAG_DENSE | MB_TAG_CREAT));
    }

    // Set tag data on elements
    if (!elements.empty() && !data.empty()) {
        MB_CHK_ERR(interface->tag_set_data(tag, elements.data(),
                                        elements.size(), data.data()));
    }

    return MB_SUCCESS;
}

ErrorCode ScalarRemapper::write_to_tags(const std::string& tag_prefix) {
    for (const auto& field_pair : m_mesh_data.d_scalar_fields) {
        const std::string& var_name = field_pair.first;
        const auto& field_data = field_pair.second;

        // Create tag name
        std::string tag_name = tag_prefix + var_name;

        // Create or get the tag
        MB_CHK_ERR(write_to_tag<double>(m_interface, tag_name, m_mesh_data.elements, field_data));

        if (m_pcomm->rank() == 0) {
            std::cout << "Created tag '" << tag_name << "' with " << field_data.size() << " values" << std::endl;
        }
    }

    for (const auto& field_pair : m_mesh_data.i_scalar_fields) {
        const std::string& var_name = field_pair.first;
        const auto& field_data = field_pair.second;

        // Create tag name
        std::string tag_name = tag_prefix + var_name;

        // Create or get the tag
        MB_CHK_ERR(write_to_tag<int>(m_interface, tag_name, m_mesh_data.elements, field_data));

        if (m_pcomm->rank() == 0) {
            std::cout << "Created tag '" << tag_name << "' with " << field_data.size() << " values" << std::endl;
        }
    }

    return MB_SUCCESS;
}

// Nearest Neighbor Remapper Implementation
NearestNeighborRemapper::NearestNeighborRemapper(Interface* interface, ParallelComm* pcomm, EntityHandle mesh_set)
    : ScalarRemapper(interface, pcomm, mesh_set), m_kdtree_built(false) {
}

NearestNeighborRemapper::~NearestNeighborRemapper() {
    // KD-tree cleanup handled automatically by unique_ptr
}

ErrorCode NearestNeighborRemapper::build_kdtree(const ParallelPointCloudReader::PointData& point_data) {
    if (point_data.lonlat_coordinates.empty()) {
        return MB_FAILURE;
    }

    try {
        // Create adapter for point cloud data
        m_adapter = std::unique_ptr<PointCloudAdapter>(new PointCloudAdapter(point_data.lonlat_coordinates));

        // Create KD-tree with 10 max leaf size for good performance
        m_kdtree = std::unique_ptr<KDTree>(new KDTree(3, *m_adapter,
                                                        nanoflann::KDTreeSingleIndexAdaptorParams(20,
                                                            nanoflann::KDTreeSingleIndexAdaptorFlags::SkipInitialBuildIndex,
                                                            12 /* number of threads */)
                                                    ) );

        // Build the index
        if (m_pcomm->rank() == 0) {
            std::cout << "Building KD-tree index now..." << std::endl;
        }
        m_kdtree->buildIndex();
        m_kdtree_built = true;

        if (m_pcomm->rank() == 0) {
            std::cout << "Built KD-tree index for " << point_data.lonlat_coordinates.size() << " points" << std::endl;
        }

        return MB_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "Error building KD-tree: " << e.what() << std::endl;
        m_kdtree_built = false;
        return MB_FAILURE;
    }
}

ErrorCode NearestNeighborRemapper::perform_remapping(const ParallelPointCloudReader::PointData& point_data) {
    if (point_data.lonlat_coordinates.empty()) {
        if (m_pcomm->rank() == 0) {
            std::cout << "No point cloud data available for remapping" << std::endl;
        }
        return MB_SUCCESS;
    }

    // Build KD-tree for fast nearest neighbor queries
    if (!m_kdtree_built) {
        MB_CHK_ERR(build_kdtree(point_data));
    }

    std::cout.flush();

#pragma omp parallel for shared(m_mesh_data, point_data, m_kdtree)
    for (size_t elem_idx = 0; elem_idx < m_mesh_data.centroids.size(); ++elem_idx) {
        const ParallelPointCloudReader::PointType3D& centroid = m_mesh_data.centroids[elem_idx];

        size_t nearest_point_idx = find_nearest_point(centroid, point_data);

        if (elem_idx < 10) {
            std::cout << "Nearest point for element " << elem_idx << " is " << nearest_point_idx << std::endl;
        }
        if (nearest_point_idx != static_cast<size_t>(-1)) {
            // Copy scalar values from nearest point to mesh element
            for (const auto& var_name : m_config.scalar_var_names) {
                auto var_it = point_data.d_scalar_variables.find(var_name);
                if (var_it != point_data.d_scalar_variables.end() &&
                    nearest_point_idx < var_it->second.size()) {
                    m_mesh_data.d_scalar_fields[var_name][elem_idx] = var_it->second[nearest_point_idx];
                    if (elem_idx < 10) {
                        std::cout << "Double Data (" << var_name << ") at nearest neighbor element: " << nearest_point_idx << " is " << var_it->second[nearest_point_idx] << std::endl;
                    }
                }
                else {
                    auto ivar_it = point_data.i_scalar_variables.find(var_name);
                    if (ivar_it != point_data.i_scalar_variables.end() &&
                        nearest_point_idx < ivar_it->second.size()) {
                        m_mesh_data.i_scalar_fields[var_name][elem_idx] = ivar_it->second[nearest_point_idx];
                    }
                    if (elem_idx < 10) {
                        std::cout << "Integer Data (" << var_name << ") at nearest neighbor element: " << nearest_point_idx << " is " << ivar_it->second[nearest_point_idx] << std::endl;
                    }
                }
            }
        }
        else {
            std::cout << m_pcomm->rank() << ": No nearest point found for element " << elem_idx << std::endl;
        }
    }

    return MB_SUCCESS;
}

size_t NearestNeighborRemapper::find_nearest_point(const ParallelPointCloudReader::PointType3D& target_point,
                                               const ParallelPointCloudReader::PointData& point_data) {
    if (!m_kdtree_built || !m_kdtree) {
        // Fallback to linear search if KD-tree is not available
        size_t nearest_idx = static_cast<size_t>(-1);
        ParallelPointCloudReader::CoordinateType min_distance = std::numeric_limits<ParallelPointCloudReader::CoordinateType>::max();

        // ParallelPointCloudReader::CoordinateType lon, lat;
        // const auto coordpoint = target_point.data();
        // XYZtoRLL_Deg(coordpoint, lon, lat);
        for (size_t i = 0; i < point_data.lonlat_coordinates.size(); ++i) {
            ParallelPointCloudReader::CoordinateType distance = compute_distance(target_point, point_data.lonlat_coordinates[i]);

            // Check search radius constraint
            if (m_config.search_radius > 0.0 && distance > m_config.search_radius) {
                continue;
            }

            if (distance < min_distance) {
                min_distance = distance;
                nearest_idx = i;
            }
        }

        return nearest_idx;
    }

    // Use KD-tree for fast nearest neighbor search
    try {
        // const double query_pt[3] = {target_point[0], target_point[1], target_point[2]};

        if (m_config.search_radius > 0.0) {
            // Radius search with distance constraint
            std::vector<nanoflann::ResultItem<size_t, ParallelPointCloudReader::CoordinateType>> ret_matches;
            nanoflann::SearchParameters params;
            const size_t num_matches = m_kdtree->radiusSearch(
                target_point.data(), m_config.search_radius * m_config.search_radius, ret_matches, params);

            if (num_matches == 0) {
                return static_cast<size_t>(-1);
            }

            // Find the closest match within radius
            size_t best_idx = ret_matches[0].first;
            auto best_dist = ret_matches[0].second;
            for (size_t i = 1; i < num_matches; ++i) {
                if (ret_matches[i].second < best_dist) {
                    best_dist = ret_matches[i].second;
                    best_idx = ret_matches[i].first;
                }
            }
            return best_idx;
        } else {
            // Simple nearest neighbor search
            std::vector<size_t> ret_index(m_config.max_neighbors);
            std::vector<ParallelPointCloudReader::CoordinateType> out_dist_sqr(m_config.max_neighbors);
            nanoflann::KNNResultSet<ParallelPointCloudReader::CoordinateType> result_set(m_config.max_neighbors);
            result_set.init(ret_index.data(), out_dist_sqr.data());

            m_kdtree->findNeighbors(result_set, target_point.data(), nanoflann::SearchParameters());

            return ret_index[0]; // Return the nearest point index
        }
    } catch (const std::exception& e) {
        std::cerr << "Error in KD-tree search: " << e.what() << std::endl;
        return static_cast<size_t>(-1);
    }
}

// Inverse Distance Remapper Implementation
InverseDistanceRemapper::InverseDistanceRemapper(Interface* interface, ParallelComm* pcomm, EntityHandle mesh_set)
    : ScalarRemapper(interface, pcomm, mesh_set) {
}

ErrorCode InverseDistanceRemapper::perform_remapping(const ParallelPointCloudReader::PointData& point_data) {
    if (point_data.lonlat_coordinates.empty()) {
        if (m_pcomm->rank() == 0) {
            std::cout << "No point cloud data available for remapping" << std::endl;
        }
        return MB_SUCCESS;
    }

    // For each mesh element centroid, find weighted neighbors
    for (size_t elem_idx = 0; elem_idx < m_mesh_data.centroids.size(); ++elem_idx) {
        const auto& centroid = m_mesh_data.centroids[elem_idx];

        std::vector<WeightedPoint> weighted_neighbors = find_weighted_neighbors(centroid, point_data);

        if (!weighted_neighbors.empty()) {
            // Interpolate scalar values using inverse distance weights
            for (const auto& var_name : m_config.scalar_var_names) {
                auto var_it = point_data.d_scalar_variables.find(var_name);
                if (var_it != point_data.d_scalar_variables.end()) {
                    ParallelPointCloudReader::CoordinateType weighted_sum = 0.0;
                    ParallelPointCloudReader::CoordinateType total_weight = 0.0;

                    for (const auto& wp : weighted_neighbors) {
                        if (wp.index < static_cast<int>(var_it->second.size())) {
                            weighted_sum += var_it->second[wp.index] * wp.weight;
                            total_weight += wp.weight;
                        }
                    }

                    if (total_weight > 0.0) {
                        m_mesh_data.d_scalar_fields[var_name][elem_idx] = weighted_sum / total_weight;
                    }
                }
            }
        }
    }

    return MB_SUCCESS;
}

std::vector<InverseDistanceRemapper::WeightedPoint>
InverseDistanceRemapper::find_weighted_neighbors(const ParallelPointCloudReader::PointType3D& target_point,
                                                const ParallelPointCloudReader::PointData& point_data) {
    std::vector<WeightedPoint> weighted_points;

    // const auto coordpoint = target_point.data();
    // ParallelPointCloudReader::CoordinateType lon, lat;
    // XYZtoRLL_Deg(coordpoint, lon, lat);
    for (size_t i = 0; i < point_data.lonlat_coordinates.size(); ++i) {
        ParallelPointCloudReader::CoordinateType distance = compute_distance(target_point, point_data.lonlat_coordinates[i]);

        // Check search radius constraint
        if (m_config.search_radius > 0.0 && distance > m_config.search_radius) {
            continue;
        }

        ParallelPointCloudReader::CoordinateType weight = compute_inverse_distance_weight(distance);
        weighted_points.push_back({static_cast<int>(i), weight});
    }

    // Sort by weight (descending) and keep only max_neighbors
    std::sort(weighted_points.begin(), weighted_points.end(),
              [](const WeightedPoint& a, const WeightedPoint& b) {
                  return a.weight > b.weight;
              });

    if (weighted_points.size() > static_cast<size_t>(m_config.max_neighbors)) {
        weighted_points.resize(m_config.max_neighbors);
    }

    return weighted_points;
}

ParallelPointCloudReader::CoordinateType InverseDistanceRemapper::compute_inverse_distance_weight(ParallelPointCloudReader::CoordinateType distance, ParallelPointCloudReader::CoordinateType power) {
    if (distance < 1e-12) {
        return 1e12; // Very large weight for coincident points
    }
    return 1.0 / std::pow(distance, power);
}

// Factory Implementation
std::unique_ptr<ScalarRemapper> RemapperFactory::create_remapper(
    RemapMethod method,
    Interface* interface,
    ParallelComm* pcomm,
    EntityHandle mesh_set) {

    switch (method) {
        case NEAREST_NEIGHBOR:
            return std::unique_ptr<ScalarRemapper>(new NearestNeighborRemapper(interface, pcomm, mesh_set));
        case INVERSE_DISTANCE:
            return std::unique_ptr<ScalarRemapper>(new InverseDistanceRemapper(interface, pcomm, mesh_set));
        case BILINEAR:
        case CUBIC_SPLINE:
            // TODO: Implement additional methods
            std::cerr << "Remapping method not yet implemented" << std::endl;
            return std::unique_ptr<ScalarRemapper>();
        default:
            std::cerr << "Unknown remapping method" << std::endl;
            return std::unique_ptr<ScalarRemapper>();
    }
}

} // namespace moab

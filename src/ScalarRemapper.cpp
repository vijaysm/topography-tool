#include "ScalarRemapper.hpp"
#include "moab/CartVect.hpp"
#include "moab/IntxMesh/IntxUtils.hpp"
#include "GaussLobattoQuadrature.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <limits>
#include <chrono>
#include <numeric>
#include <set>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace moab {

template<typename T>
static typename T::value_type compute_distance(const T& p1, const T& p2) {
    typename T::value_type dMag = 0.0;
    for (size_t i = 0; i < p1.size(); ++i) {
        dMag += (p1[i] - p2[i])*(p1[i] - p2[i]);
    }
    return std::sqrt(dMag);
}

static ParallelPointCloudReader::CoordinateType compute_distance(const ParallelPointCloudReader::PointType3D& p1, const ParallelPointCloudReader::PointType& p2) {

    ParallelPointCloudReader::PointType3D p2_3d;
    RLLtoXYZ_Deg(p2[0], p2[1], p2_3d);

    return compute_distance(p1, p2_3d);
}


ScalarRemapper::ScalarRemapper(Interface* interface, ParallelComm* pcomm, EntityHandle mesh_set)
    : m_interface(interface), m_pcomm(pcomm), m_mesh_set(mesh_set), m_kdtree_built(false), m_grid_locator_built(false) {
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
        std::cout << "Remapping completed in " << duration.count() / 1000.0 << " seconds" << std::endl;
    }

    // Validate results and print statistics
    MB_CHK_ERR(validate_remapping_results());
    print_remapping_statistics();

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
    CartVect centroid_data;
    MB_CHK_ERR(m_interface->get_coords(&element, 1, centroid_data.array()));

    // project to unit sphere
    centroid_data.normalize();

    for (size_t i = 0; i < 3; ++i) {
        centroid[i] = centroid_data[i];
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

        double min_val = *std::min_element(field_data.begin(), field_data.end());
        double max_val = *std::max_element(field_data.begin(), field_data.end());
        double sum = std::accumulate(field_data.begin(), field_data.end(), 0);
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
    : ScalarRemapper(interface, pcomm, mesh_set) {
}

NearestNeighborRemapper::~NearestNeighborRemapper() {
    // KD-tree cleanup handled automatically by unique_ptr
}

ErrorCode ScalarRemapper::build_grid_locator(const ParallelPointCloudReader::PointData& point_data) {
    if (point_data.lonlat_coordinates.empty()) {
        return MB_FAILURE;
    }

    try {
        if (m_pcomm->rank() == 0) {
            std::cout << "Building RegularGridLocator for USGS format data..." << std::endl;
        }

        // For USGS format, we need to extract unique lat/lon values from the point cloud
        // Since the data is organized as a regular grid, we can extract unique coordinates
        std::vector<double> lats, lons;
        std::set<double> unique_lats, unique_lons;

        for (const auto& pt : point_data.lonlat_coordinates) {
            unique_lons.insert(pt[0]);
            unique_lats.insert(pt[1]);
        }

        lons.assign(unique_lons.begin(), unique_lons.end());
        lats.assign(unique_lats.begin(), unique_lats.end());

        if (m_pcomm->rank() == 0) {
            std::cout << "Detected grid dimensions: " << lats.size() << " x " << lons.size() << std::endl;
        }

        auto build_start = std::chrono::high_resolution_clock::now();
        m_grid_locator = std::unique_ptr<RegularGridLocator>(
            new RegularGridLocator(lats, lons, m_config.distance_metric));
        m_grid_locator_built = true;

        auto build_end = std::chrono::high_resolution_clock::now();
        auto build_duration = std::chrono::duration_cast<std::chrono::milliseconds>(build_end - build_start);
        if (m_pcomm->rank() == 0) {
            std::cout << "RegularGridLocator build time: " << build_duration.count()/1000.0 << " seconds" << std::endl;
        }

        return MB_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "Error building RegularGridLocator: " << e.what() << std::endl;
        m_grid_locator_built = false;
        return MB_FAILURE;
    }
}

ErrorCode ScalarRemapper::build_kdtree(const ParallelPointCloudReader::PointData& point_data) {
    if (point_data.lonlat_coordinates.empty()) {
        return MB_FAILURE;
    }
    int num_threads = 1;
#ifdef _OPENMP
#pragma omp parallel
    {
        num_threads = omp_get_max_threads();
    }
    // num_threads = omp_get_num_threads();
#endif

    try {
        // Create adapter for point cloud data
        m_adapter = std::unique_ptr<PointCloudAdapter>(new PointCloudAdapter(point_data.lonlat_coordinates));

        // Create KD-tree with 10 max leaf size for good performance
        if (m_pcomm->rank() == 0) {
            std::cout << "Building KD-tree index now with " << num_threads << " threads..." << std::endl;
        }

        auto redist_start = std::chrono::high_resolution_clock::now();
        m_kdtree = std::unique_ptr<KDTree>(new KDTree(3, *m_adapter,
                                                        nanoflann::KDTreeSingleIndexAdaptorParams(16,
                                                            nanoflann::KDTreeSingleIndexAdaptorFlags::SkipInitialBuildIndex,
                                                            num_threads /* number of threads */)
                                                    ) );

        // Build the index
        m_kdtree->buildIndex();
        m_kdtree_built = true;

        auto redist_end = std::chrono::high_resolution_clock::now();
        auto redist_duration = std::chrono::duration_cast<std::chrono::milliseconds>(redist_end - redist_start);
        if (m_pcomm->rank() == 0) {
            std::cout << "KD-tree build time: " << redist_duration.count()/1000.0 << " seconds" << std::endl;
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

    // Build spatial index (KD-tree or RegularGridLocator)
    if (m_config.is_usgs_format && !m_config.use_kd_tree) {
        if (!m_grid_locator_built) {
            MB_CHK_ERR(build_grid_locator(point_data));
        }
    } else {
        if (!m_kdtree_built) {
            MB_CHK_ERR(build_kdtree(point_data));
        }
    }

    std::cout.flush();

    constexpr size_t max_neighbors = 1;
#pragma omp parallel for shared(m_mesh_data, point_data, m_kdtree)
    for (size_t elem_idx = 0; elem_idx < m_mesh_data.centroids.size(); ++elem_idx) {
        const ParallelPointCloudReader::PointType3D& centroid = m_mesh_data.centroids[elem_idx];

        auto nearest_point = find_nearest_point(centroid, point_data, &max_neighbors);
        size_t nearest_point_idx = nearest_point.size() > 0 ? nearest_point[0].first : static_cast<size_t>(-1);
        // ParallelPointCloudReader::CoordinateType nearest_point_dist_sq = nearest_point.second * nearest_point.second;

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

std::vector<nanoflann::ResultItem<size_t, ParallelPointCloudReader::CoordinateType>> ScalarRemapper::find_nearest_point(const ParallelPointCloudReader::PointType3D& target_point,
                                               const ParallelPointCloudReader::PointData& point_data,
                                             const size_t* user_max_neighbors,
                                            const ParallelPointCloudReader::CoordinateType* user_search_radius) {
    size_t max_neighbors = 1;
    ParallelPointCloudReader::CoordinateType search_radius = 0.0;

    if (user_max_neighbors) max_neighbors = *user_max_neighbors;
    if (user_search_radius) search_radius = *user_search_radius;

    std::vector<nanoflann::ResultItem<size_t, ParallelPointCloudReader::CoordinateType>> ret_matches;

    // Use RegularGridLocator for USGS format if enabled
    if (m_config.is_usgs_format && !m_config.use_kd_tree && m_grid_locator_built && m_grid_locator) {
        // Convert 3D Cartesian to lon/lat for grid query
        ParallelPointCloudReader::CoordinateType lon, lat;
        XYZtoRLL_Deg(target_point.data(), lon, lat);
        ParallelPointCloudReader::PointType3D query_pt = {lon, lat, 0.0};

        if (search_radius > 0.0) {
            // Radius search
            size_t num_matches = m_grid_locator->radiusSearch(query_pt, search_radius, ret_matches);
            if (num_matches == 0) {
                ret_matches.push_back(nanoflann::ResultItem<size_t, ParallelPointCloudReader::CoordinateType>(static_cast<size_t>(-1), 0.0));
            }
        } else {
            // kNN search
            std::vector<size_t> indices(max_neighbors);
            std::vector<ParallelPointCloudReader::CoordinateType> distances_sq(max_neighbors);
            m_grid_locator->knnSearch(query_pt, max_neighbors, indices.data(), distances_sq.data());

            for (size_t i = 0; i < max_neighbors; ++i) {
                ret_matches.push_back({indices[i], distances_sq[i]});
            }
        }
        return ret_matches;
    }

    // Fallback to KD-tree or linear search
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
            if (search_radius > 0.0 && distance > search_radius) {
                continue;
            }

            if (distance < min_distance) {
                min_distance = distance;
                nearest_idx = i;
            }
        }

        ret_matches.push_back(nanoflann::ResultItem<size_t, ParallelPointCloudReader::CoordinateType>(nearest_idx, min_distance));

        return ret_matches;
    }

    // Use KD-tree for fast nearest neighbor search
    try {
        // const double query_pt[3] = {target_point[0], target_point[1], target_point[2]};
        nanoflann::SearchParameters params;
        params.sorted = true;
        params.eps = std::numeric_limits<ParallelPointCloudReader::CoordinateType>::epsilon() * 10; // 10x machine epsilon

        if (search_radius > 0.0) {
            // Radius search with distance constraint
            const size_t num_matches = m_kdtree->radiusSearch(
                target_point.data(), search_radius, ret_matches, params);

            if (num_matches == 0) {
                ret_matches.push_back(nanoflann::ResultItem<size_t, ParallelPointCloudReader::CoordinateType>(static_cast<size_t>(-1), 0.0));
            }
            return ret_matches;

            // Find the closest match within radius
            // size_t best_idx = ret_matches[0].first;
            // auto best_dist = ret_matches[0].second;
            // for (size_t i = 1; i < num_matches; ++i) {
            //     if (ret_matches[i].second < best_dist) {
            //         best_dist = ret_matches[i].second;
            //         best_idx = ret_matches[i].first;
            //     }
            // }
            // return std::pair<size_t, ParallelPointCloudReader::CoordinateType>(best_idx, best_dist);
        } else {
            // Simple nearest neighbor search
            std::vector<size_t> ret_index(max_neighbors);
            std::vector<ParallelPointCloudReader::CoordinateType> out_dist_sqr(max_neighbors);
            nanoflann::KNNResultSet<ParallelPointCloudReader::CoordinateType> result_set(max_neighbors);
            result_set.init(ret_index.data(), out_dist_sqr.data());

            // m_kdtree->findNeighbors(result_set, target_point.data(), params);
            m_kdtree->knnSearch(target_point.data(), max_neighbors, ret_index.data(), out_dist_sqr.data());

            // return std::pair<size_t, ParallelPointCloudReader::CoordinateType>(ret_index[0], out_dist_sqr[0]);
            // Return the nearest point indices and distances
            for (size_t i = 0; i < max_neighbors; ++i) {
                ret_matches.push_back(nanoflann::ResultItem<size_t, ParallelPointCloudReader::CoordinateType>(ret_index[i], out_dist_sqr[i]));
            }
            return ret_matches;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error in KD-tree search: " << e.what() << std::endl;
        ret_matches.push_back(nanoflann::ResultItem<size_t, ParallelPointCloudReader::CoordinateType>(static_cast<size_t>(-1), 0.0));
        return ret_matches;
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
    constexpr double search_radius = 0.0;

    // const auto coordpoint = target_point.data();
    // ParallelPointCloudReader::CoordinateType lon, lat;
    // XYZtoRLL_Deg(coordpoint, lon, lat);
    for (size_t i = 0; i < point_data.lonlat_coordinates.size(); ++i) {
        ParallelPointCloudReader::CoordinateType distance = compute_distance(target_point, point_data.lonlat_coordinates[i]);

        // Check search radius constraint
        if (search_radius > 0.0 && distance > search_radius) {
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
        case PC_AVERAGED_SPECTRAL_PROJECTION:
            return std::unique_ptr<ScalarRemapper>(new PCSpectralProjectionRemapper(interface, pcomm, mesh_set));
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



static moab::ErrorCode Remapper_ApplyLocalMap(
    Interface* mb,
    const double nodes[12],
	double dAlpha,
	double dBeta,
	CartVect& node,
	CartVect* dDx1G = nullptr,
	CartVect* dDx2G = nullptr
) {
	// Calculate nodal locations on the plane
	double dXc =
		  nodes[0*3+0] * (1.0 - dAlpha) * (1.0 - dBeta)
		+ nodes[1*3+0] *        dAlpha  * (1.0 - dBeta)
		+ nodes[2*3+0] *        dAlpha  *        dBeta
		+ nodes[3*3+0] * (1.0 - dAlpha) *        dBeta;

	double dYc =
		  nodes[0*3+1] * (1.0 - dAlpha) * (1.0 - dBeta)
		+ nodes[1*3+1] *        dAlpha  * (1.0 - dBeta)
		+ nodes[2*3+1] *        dAlpha  *        dBeta
		+ nodes[3*3+1] * (1.0 - dAlpha) *        dBeta;

	double dZc =
		  nodes[0*3+2] * (1.0 - dAlpha) * (1.0 - dBeta)
		+ nodes[1*3+2] *        dAlpha  * (1.0 - dBeta)
		+ nodes[2*3+2] *        dAlpha  *        dBeta
		+ nodes[3*3+2] * (1.0 - dAlpha) *        dBeta;

	double dR = sqrt(dXc * dXc + dYc * dYc + dZc * dZc);

	// Mapped node location
	node[0] = dXc / dR;
	node[1] = dYc / dR;
	node[2] = dZc / dR;


	// Pointwise basis vectors in Cartesian geometry
	CartVect dDx1F(
		(1.0 - dBeta) * (nodes[1*3+0] - nodes[0*3+0])
		+      dBeta  * (nodes[2*3+0] - nodes[3*3+0]),
		(1.0 - dBeta) * (nodes[1*3+1] - nodes[0*3+1])
		+      dBeta  * (nodes[2*3+1] - nodes[3*3+1]),
		(1.0 - dBeta) * (nodes[1*3+2] - nodes[0*3+2])
		+      dBeta  * (nodes[2*3+2] - nodes[3*3+2]));

	CartVect dDx2F(
		(1.0 - dAlpha) * (nodes[3*3+0] - nodes[0*3+0])
		+      dAlpha  * (nodes[2*3+0] - nodes[1*3+0]),
		(1.0 - dAlpha) * (nodes[3*3+1] - nodes[0*3+1])
		+      dAlpha  * (nodes[2*3+1] - nodes[1*3+1]),
		(1.0 - dAlpha) * (nodes[3*3+2] - nodes[0*3+2])
		+      dAlpha  * (nodes[2*3+2] - nodes[1*3+2]));

	// Pointwise basis vectors in spherical geometry
	double dDenomTerm = 1.0 / (dR * dR * dR);

	if (dDx1G) {
		*dDx1G = dDenomTerm * CartVect(
			- dXc * (dYc * dDx1F[1] + dZc * dDx1F[2])
				+ (dYc * dYc + dZc * dZc) * dDx1F[0],
			- dYc * (dXc * dDx1F[0] + dZc * dDx1F[2])
				+ (dXc * dXc + dZc * dZc) * dDx1F[1],
			- dZc * (dXc * dDx1F[0] + dYc * dDx1F[1])
			+ (dXc * dXc + dYc * dYc) * dDx1F[2]);
	}

	if (dDx2G) {
		*dDx2G = dDenomTerm * CartVect(
			- dXc * (dYc * dDx2F[1] + dZc * dDx2F[2])
				+ (dYc * dYc + dZc * dZc) * dDx2F[0],
			- dYc * (dXc * dDx2F[0] + dZc * dDx2F[2])
				+ (dXc * dXc + dZc * dZc) * dDx2F[1],
			- dZc * (dXc * dDx2F[0] + dYc * dDx2F[1])
			+ (dXc * dXc + dYc * dYc) * dDx2F[2]);
    }

    return moab::MB_SUCCESS;
}


double Remapper_GenerateFEMetaData(
    Interface* mb,
    EntityHandle face,
    const std::vector<double>& dG,
    const std::vector<double>& dW,
	std::vector<double> & dataGLLJacobian,
    const bool apply_bubble_correction = false
) {
    const int nP = dG.size();

	// Initialize data structures
	dataGLLJacobian.resize(nP * nP);

	// Write metadata
    const EntityHandle* connectivity;
    int nConnectivity;
    MB_CHK_SET_ERR(mb->get_connectivity(face, connectivity, nConnectivity, true), "Failed to get connectivity");

    if (nP != 4 || nConnectivity != 4) {
        MB_CHK_SET_ERR(MB_FAILURE, "Mesh must only contain quadrilateral elements");
    }

    // Default area_method = lHuiller; Options: Girard, lHuiller, GaussQuadrature (if TR is available)
    moab::IntxAreaUtils areaAdaptor( moab::IntxAreaUtils::lHuiller );

    double nConnCoords[12]; // 4 nodes, 3D coordinates
    MB_CHK_SET_ERR(mb->get_coords(connectivity, nConnectivity, nConnCoords), "Failed to get coordinates");

    double dFaceNumericalArea = 0.0;

    for (int j = 0; j < nP; j++) {
        for (int i = 0; i < nP; i++) {

            // Get local map vectors
            CartVect nodeGLL;
            CartVect dDx1G;
            CartVect dDx2G;

            Remapper_ApplyLocalMap(
                mb,
                nConnCoords,
                dG[i],
                dG[j],
                nodeGLL,
                &dDx1G,
                &dDx2G);

            // For continuous GLL determine the mapping from local node index to global index
            // if (fContinuousGLL) {

            //     // Determine if this is a unique Node
            //     std::map<Node, int>::const_iterator iter = mapNodes.find(nodeGLL);
            //     if (iter == mapNodes.end()) {

            //         // Insert new unique node into map
            //         int ixNode = static_cast<int>(mapNodes.size());
            //         mapNodes.insert(std::pair<Node, int>(nodeGLL, ixNode));
            //         dataGLLnodes[j][i][k] = ixNode + 1;

            //     } else {
            //         dataGLLnodes[j][i][k] = iter->second + 1;
            //     }

            // } else {
            //     dataGLLnodes[j][i][k] = ixNode;
            //     ixNode++;
            // }

            // Cross product gives local Jacobian
            CartVect nodeCross = dDx1G * dDx2G;

            // Element area weighted by local GLL weights
            const double dJacobian = nodeCross.length() * dW[i] * dW[j];
            if (dJacobian <= 0.0) {
                MB_CHK_SET_ERR(MB_FAILURE, "Nonpositive Jacobian detected");
            }

            dFaceNumericalArea += dJacobian;

            dataGLLJacobian[j * nP + i] = dJacobian;
        }
    }


    // Apply bubble adjustment to area
    if (apply_bubble_correction) {

        // const double vecFaceArea = dFaceNumericalArea;
        // call IntxUtils calculate face areas using quadrature
        const double vecFaceArea = areaAdaptor.area_spherical_polygon( nConnCoords, 4, 1.0 );

        if (dFaceNumericalArea != vecFaceArea) {
            // Use uniform bubble for linear elements
            if (nP < 3) {
                double dMassDifference = vecFaceArea - dFaceNumericalArea;
                for (int j = 0; j < nP; j++) {
                    for (int i = 0; i < nP; i++) {
                        dataGLLJacobian[j * nP + i] +=
                            dMassDifference * dW[i] * dW[j];
                    }
                }

                dFaceNumericalArea += dMassDifference;

            // Use HOMME bubble for higher order elements
            } else {
                double dMassDifference = vecFaceArea - dFaceNumericalArea;

                double dInteriorMassSum = 0;
                for (int i = 1; i < nP-1; i++) {
                    for (int j = 1; j < nP-1; j++) {
                            dInteriorMassSum += dataGLLJacobian[j * nP + i];
                    }
                }

                // Check that dInteriorMassSum is not too small
                if (std::abs(dInteriorMassSum) < 1e-15) {
                    MB_CHK_SET_ERR(MB_FAILURE, "--bubble correction cannot be performed, "
                        "sum of inner weights is too small");
                }

                dInteriorMassSum = dMassDifference / dInteriorMassSum;
                for (int j = 1; j < nP-1; j++) {
                    for (int i = 1; i < nP-1; i++) {
                        dataGLLJacobian[j * nP + i] *= (1.0 + dInteriorMassSum);
                    }
                }

                dFaceNumericalArea += dMassDifference;
            }
        }
    }

    // Return the compute area from element
	return dFaceNumericalArea;
}


///////////////////////////////////////////////////////////////////////////////
// ----------------------
// KD-tree point cloud
// ----------------------
struct MOABCentroidCloud
{
    constexpr static int DIM = 3;
    std::vector< std::array< double, DIM > > points;
    std::vector< moab::EntityHandle > elements;

    inline void init( size_t length )
    {
        points.reserve( length );
        elements.reserve( length );
    }

    inline size_t kdtree_get_point_count() const
    {
        return points.size();
    }

    inline double kdtree_get_pt( const size_t idx, const size_t dim ) const
    {
        return points[idx][dim];
    }

    template < class BBOX >
    bool kdtree_get_bbox( BBOX& ) const
    {
        return false;
    }
};

using MOABKDTree = nanoflann::KDTreeSingleIndexAdaptor< nanoflann::L2_Simple_Adaptor< ParallelPointCloudReader::CoordinateType, MOABCentroidCloud, ParallelPointCloudReader::CoordinateType >,
                                                    MOABCentroidCloud,  // DatasetAdaptor
                                                    3,                  // 3D
                                                    size_t              // IndexType
                                                    >;

// ----------------------
// Radius search wrapper
// ----------------------
std::vector< size_t > radius_search_kdtree( const MOABKDTree& tree,
                                            const MOABCentroidCloud& cloud,
                                            const std::array< ParallelPointCloudReader::CoordinateType, 3 >& query_pt,
                                            ParallelPointCloudReader::CoordinateType radius )
{
    ParallelPointCloudReader::CoordinateType radius_sq = radius * radius;
    // ParallelPointCloudReader::CoordinateType radius_sq = radius;
    std::vector< nanoflann::ResultItem< size_t, ParallelPointCloudReader::CoordinateType > > matches;
    nanoflann::SearchParameters params;
    params.sorted = true;

    const ParallelPointCloudReader::CoordinateType query_pt_sq = query_pt[0] * query_pt[0] + query_pt[1] * query_pt[1] + query_pt[2] * query_pt[2];
    assert( query_pt_sq > 1.0 - 1e-10 && query_pt_sq < 1.0 + 1.0e-10 ); // Check if the point is on the unit sphere

    tree.radiusSearch( query_pt.data(), radius_sq, matches, params );

    std::vector< size_t > found_elements;
    if( !matches.empty() )
    {
        // Return all matches within the radius
        for( const auto& match : matches )
            found_elements.emplace_back( cloud.elements[match.first] );
    }
    else
    {
        // Fallback to nearest neighbor
        size_t nearest_index;
        ParallelPointCloudReader::CoordinateType nearest_dist_sq;
        tree.knnSearch( query_pt.data(), 1, &nearest_index, &nearest_dist_sq );
        found_elements.emplace_back( cloud.elements[nearest_index] );
    }

    return found_elements;
}

///////////////////////////////////////////////////////////////////////////////

// PCSpectralProjectionRemapper Implementation
PCSpectralProjectionRemapper::PCSpectralProjectionRemapper(Interface* interface, ParallelComm* pcomm, EntityHandle mesh_set)
    : ScalarRemapper(interface, pcomm, mesh_set) {
}

ErrorCode PCSpectralProjectionRemapper::perform_remapping(const ParallelPointCloudReader::PointData& point_data) {
    if (point_data.lonlat_coordinates.empty()) {
        if (m_pcomm->rank() == 0) {
            std::cout << "No point cloud data available for remapping" << std::endl;
        }
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

    std::cout.flush();

    if (m_pcomm->rank() == 0) {
        std::cout << "Starting PC averaged spectral projection with " << point_data.lonlat_coordinates.size()
                  << " point cloud points and spectral order " << m_config.spectral_order << std::endl;
    }

    // Project point cloud data to all spectral elements
    MB_CHK_ERR(project_point_cloud_to_spectral_elements(point_data));

    return MB_SUCCESS;
}

ErrorCode PCSpectralProjectionRemapper::validate_quadrilateral_mesh() {
    // Check that all elements are quadrilaterals
    for (EntityHandle element : m_mesh_data.elements) {
        const EntityHandle* connectivity;
        int nConnectivity;
        MB_CHK_ERR(m_interface->get_connectivity(element, connectivity, nConnectivity, true));

        if (nConnectivity != 4) {
            if (m_pcomm->rank() == 0) {
                std::cerr << "Error: Spectral element projection requires quadrilateral mesh elements. "
                          << "Found element with " << nConnectivity << " vertices." << std::endl;
            }
            return MB_FAILURE;
        }
    }

    if (m_pcomm->rank() == 0) {
        std::cout << "Validated " << m_mesh_data.elements.size() << " quadrilateral elements for spectral projection" << std::endl;
    }

    return MB_SUCCESS;
}

ErrorCode PCSpectralProjectionRemapper::project_point_cloud_to_spectral_elements(
    const ParallelPointCloudReader::PointData& point_data) {

    if (m_pcomm->rank() == 0) {
        std::cout << "Performing spectral element projection with " << point_data.lonlat_coordinates.size()
                  << " point cloud points" << std::endl;
    }

    const int nP = m_config.spectral_order;

    if (m_pcomm->rank() == 0) {
        std::cout << "Processing " << m_mesh_data.elements.size() << " quadrilateral elements in parallel" << std::endl;
    }

    // Get GLL quadrature points and weights
    std::vector<double> dG, dW;
    MB_CHK_SET_ERR(GaussLobattoQuadrature::GetPoints(nP, dG, dW, 0.0, 1.0), "Failed to get GLL quadrature points and weights");

    // Parallel processing of mesh elements
    std::vector<ErrorCode> element_errors(m_mesh_data.elements.size(), MB_SUCCESS);

    std::cout.precision(10);

    const size_t nvars = m_config.scalar_var_names.size();

#pragma omp parallel for schedule(dynamic, 1) firstprivate(dG, dW) shared(m_kdtree, element_errors, point_data)
    for (size_t elem_idx = 0; elem_idx < m_mesh_data.elements.size(); ++elem_idx) {
        if ((elem_idx * 20) % m_mesh_data.elements.size() == 0) {
            std::cout << "Processing element " << elem_idx << " of " << m_mesh_data.elements.size() << std::endl;
        }

        ErrorCode rval;
        EntityHandle face = m_mesh_data.elements[elem_idx];
        const EntityHandle* connectivity;
        int nConnectivity;
        rval = m_interface->get_connectivity(face, connectivity, nConnectivity, true);
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
        double dNumericalArea = Remapper_GenerateFEMetaData(m_interface, face, dG, dW, dataGLLJacobian, m_config.apply_bubble_correction);

        // Thread-safe output for first element only (use atomic check)
        // static bool first_output_done = false;
        // if (!first_output_done && m_pcomm->rank() == 0) {
        //     #pragma omp critical
        //     {
        //         if (!first_output_done) {
        //             std::cout << "GLL Jacobian size: " << dataGLLJacobian.size() << std::endl;
        //             first_output_done = true;
        //         }
        //     }
        // }

        // Initialize element-averaged values (thread-local)
        std::unordered_map<std::string, double> element_averages_d;
        // std::unordered_map<std::string, int> element_averages_i;
        std::unordered_map<std::string, double> total_weights;  // Per-variable weight tracking

        // Pre-compute all GLL node coordinates for this element (vectorizable)
        std::vector<bool> gll_valid(nP * nP, false);

        // Process all GLL nodes for this element
        for (int j = 0; j < nP; j++) {
            for (int i = 0; i < nP; i++) {
                int gll_idx = j * nP + i;
                CartVect nodeGLL;
                rval = Remapper_ApplyLocalMap(m_interface, nConnCoords,
                                                        dG[i], dG[j], nodeGLL);
                if (MB_SUCCESS != rval) {
                    gll_valid[gll_idx] = false;
                    continue;
                }

                // GLL point in 3D Cartesian coordinates (matches KD-tree coordinate system)
                ParallelPointCloudReader::PointType3D gll_point = {
                    static_cast<ParallelPointCloudReader::CoordinateType>(nodeGLL[0]),
                    static_cast<ParallelPointCloudReader::CoordinateType>(nodeGLL[1]),
                    static_cast<ParallelPointCloudReader::CoordinateType>(nodeGLL[2])
                };

                // Optimized KD-tree search with thread-local storage:
                thread_local std::vector<size_t> neighbor_indices;
                thread_local std::vector<ParallelPointCloudReader::CoordinateType> distances_sq;
                thread_local std::vector<nanoflann::ResultItem<size_t, ParallelPointCloudReader::CoordinateType>> matches;

                neighbor_indices.clear();
                distances_sq.clear();
                matches.clear();

                // Compute weighted average from neighboring points (vectorized)
                const ParallelPointCloudReader::CoordinateType search_radius = dataGLLJacobian[gll_idx];
                const size_t max_neighbors = 5000; // if it exceeds, perhaps SE mesh resolution is too coarse?

                auto nearest_points = find_nearest_point(gll_point, point_data, nullptr, &search_radius);

                // Check if radius search found valid results (not just dummy -1 index)
                bool has_valid_results = !nearest_points.empty() &&
                                         nearest_points[0].first != static_cast<size_t>(-1);

                if (!has_valid_results) {
                    // Fallback to nearest neighbor WITHOUT radius constraint
                    const size_t single_neighbor = 1;
                    const ParallelPointCloudReader::CoordinateType no_radius = 0.0;  // 0.0 means no radius constraint
                    nearest_points = find_nearest_point(gll_point, point_data, &single_neighbor, &no_radius);
                }

                // Check if we still have zero points after fallback
                if (nearest_points.empty()) {
                    if (elem_idx < 10) {
                        #pragma omp critical
                        std::cout << "WARNING: Element " << elem_idx << ", GLL node (" << i << "," << j
                                  << ") has ZERO neighbors even after fallback!" << std::endl;
                    }
                    continue; // Skip this GLL node
                }

                // Pre-compute inverse distance weights (vectorizable)
                thread_local std::vector<ParallelPointCloudReader::CoordinateType> inv_weights;
                inv_weights.resize(nearest_points.size());
                std::fill(inv_weights.begin(), inv_weights.end(), 1.0);

                // Vectorized distance and weight computation
                // #pragma omp simd
                // for (size_t k = 0; k < nearest_points.size(); ++k) {
                //     ParallelPointCloudReader::CoordinateType distance = std::sqrt(nearest_points[k].second);
                //     inv_weights[k] = 1.0 / (distance + 1e-10);
                // }

                // const double gll_weight_extensive = dW[i]*dW[j]*dataGLLJacobian[gll_idx]; // For extensive properties (area, mass)
                // const double gll_weight_intensive = dW[i]*dW[j]; // For intensive properties (fractions, ratios) - NO Jacobian!
                const double gll_weight = dW[i]*dW[j];

                // Process each scalar variable
                for (const auto& var_name : m_config.scalar_var_names) {
                    double weighted_sum = 0.0;
                    double weight_sum = 0.0;

                    // Determine if this is an extensive property (needs Jacobian)
                    // bool is_extensive = (var_name == "area");
                    // const double gll_weight = is_extensive ? gll_weight_extensive : gll_weight_intensive;

                    // Check if it's a double variable
                    auto var_it = point_data.d_scalar_variables.find(var_name);
                    if (!m_config.is_usgs_format && var_it != point_data.d_scalar_variables.end()) {
                        const auto& values = var_it->second;

                        // Vectorized weighted sum computation
                        #pragma omp simd reduction(+:weighted_sum,weight_sum)
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
                            const auto& values = ivar_it->second;

                            // Vectorized weighted sum computation for integers
                            #pragma omp simd reduction(+:weighted_sum,weight_sum)
                            for (size_t k = 0; k < nearest_points.size(); ++k) {
                                size_t pt_idx = nearest_points[k].first;
                                if (pt_idx < values.size()) {
                                    weighted_sum += inv_weights[k] * static_cast<double>(values[pt_idx]);
                                    weight_sum += inv_weights[k];
                                }
                            }
                        }
                    }

                    // Add GLL-weighted contribution to element average
                    if (fabs(weight_sum) > 0.0)
                    {
                        double gll_value = weighted_sum / weight_sum;
                        element_averages_d[var_name] += gll_weight * gll_value;
                        total_weights[var_name] += gll_weight;  // Track weight per variable
                    }
                }
            }
        }

        // Normalize element averages and store in mesh data (thread-safe writes)
        // Use per-variable weights for proper normalization
        for (const auto& var_name : m_config.scalar_var_names) {
            auto it = element_averages_d.find(var_name);
            auto wt_it = total_weights.find(var_name);

            if (it != element_averages_d.end() && wt_it != total_weights.end() && fabs(wt_it->second) > 0.0) {
                const double inv_total_weight = 1.0 / wt_it->second;

                auto var_it = point_data.d_scalar_variables.find(var_name);
                if (var_it != point_data.d_scalar_variables.end()) {
                    m_mesh_data.d_scalar_fields[var_name][elem_idx] = it->second * inv_total_weight;
                }
                else {
                    auto ivar_it = point_data.i_scalar_variables.find(var_name);
                    if (ivar_it != point_data.i_scalar_variables.end()) {
                        m_mesh_data.i_scalar_fields[var_name][elem_idx] = static_cast<int>(it->second * inv_total_weight);
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

    if (error_count > 0 && m_pcomm->rank() == 0) {
        std::cout << "Warning: " << error_count << " elements failed processing out of "
                  << m_mesh_data.elements.size() << " total elements" << std::endl;
    }

    // Thread-safe debugging output after parallel region
    if (m_pcomm->rank() == 0) {
        for (const auto& var_name : m_config.scalar_var_names) {
            std::cout << "Sample values for " << var_name << ": ";
            for (size_t i = 0; i < std::min(size_t(5), m_mesh_data.elements.size()); ++i) {
                auto d_it = m_mesh_data.d_scalar_fields.find(var_name);
                auto i_it = m_mesh_data.i_scalar_fields.find(var_name);
                if (d_it != m_mesh_data.d_scalar_fields.end()) {
                    std::cout << d_it->second[i] << " ";
                } else if (i_it != m_mesh_data.i_scalar_fields.end()) {
                    std::cout << i_it->second[i] << " ";
                }
            }
            std::cout << std::endl;
        }
    }

    return MB_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////

// NOTE: This function is from TempestRemap and needs significant adaptation
// The functionality is now implemented in PCSpectralProjectionRemapper class
/*
moab::ErrorCode LinearRemapFVtoGLL_Averaged( const std::vector< double >& dataGLLNodalArea )
{
    // Order of the finite element method
    constexpr int nP = 4;

    // GLL quadrature nodes
    std::vector< double > dG;
    std::vector< double > dW;

    GaussLobattoQuadrature::GetPoints( nP, 0.0, 1.0, dG, dW );

	// Number of Faces
    std::vector<EntityHandle> faces;
    m_interface->get_entities_by_dimension(m_mesh_set, 2, faces);


    // Initialize coordinates for map
    // mapRemap.InitializeTargetCoordinatesFromMeshFE(
    //     meshTarget, optsAlg.nPout, dataGLLNodes);

    // Generate the continuous Jacobian
    constexpr bool fContinuous = true;

    if (fContinuous) {
        GenerateUniqueJacobian(
            dataGLLNodes,
            dataGLLJacobian,
            mapRemap.GetTargetAreas());

    } else {
        GenerateDiscontinuousJacobian(
            dataGLLJacobian,
            mapRemap.GetTargetAreas());
    }

    // Announcements
    moab::DebugOutput dbgprint( std::cout, this->rank, 0 );
    dbgprint.set_prefix( "[LinearRemapFVtoSE_Averaged]: " );
    if( is_root )
    {
        dbgprint.printf( 0, "Finite Volume to Spectral Element Projection\n" );
    }

    DataArray3D<int> dataGLLNodes;
    DataArray3D<double> dataGLLJacobian;
    {
        AnnounceStartBlock("Generating output mesh meta data");
        double dNumericalArea =
            Remapper_GenerateFEMetaData(
                mb,
                face,
                dG, dW,
                dataGLLJacobian);

        Announce("Output Mesh Numerical Area: %1.15e", dNumericalArea);
        AnnounceEndBlock(NULL);
    }


    // Loop through all faces on meshInput
    const unsigned outputFrequency = ( m_meshOutput->faces.size() / 10 ) + 1;

    MOABCentroidCloud cloud;
    {
        // Initialize the kd-tree
        cloud.init( source_vertices.size() );

        std::vector<double> srccoords(source_vertices.size() * 3);
        MB_CHK_ERR( m_interface->get_coords( source_vertices, srccoords.data() ) );

        // Loop through all elements and add to the tree
        for( size_t ielem = 0; ielem < source_vertices.size(); ielem++ )
        {
            const size_t offset = ielem * 3;
            const double query_pt_sq =
                std::sqrt( srccoords[offset] * srccoords[offset] + srccoords[offset+1] * srccoords[offset+1] + srccoords[offset+2] * srccoords[offset+2] );

            // Rescale the coordinates to the unit sphere and add to the point cloud
            cloud.points.emplace_back( std::array< double, 3 >( { srccoords[offset]/query_pt_sq, srccoords[offset+1]/query_pt_sq, srccoords[offset+2]/query_pt_sq } ) );

            // Get the vertex index
            cloud.elements.emplace_back( ielem );
        }
    }

    // if( is_root ) dbgprint.printf( 0, "Building Kd-tree now..." );
    KDTree tree( 3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams( 10 ) );
    tree.buildIndex();
    // if( is_root ) dbgprint.printf( 0, "Finished building Kd-tree index..." );
    for( size_t ixOutput = 0; ixOutput < m_meshOutput->faces.size(); ixOutput++ )
    {
        // Output every 1000 elements
// #ifdef VERBOSE
        if( ixOutput % outputFrequency == 0 && is_root )
        {
            dbgprint.printf( 0, "Element %zu/%lu\n", ixOutput, m_meshOutput->faces.size() );
        }
// #endif
        // This Face
        const Face& faceSecond = m_meshOutput->faces[ixOutput];

        // Area of the First Face
        // double dSecondArea = m_meshOutput->vecFaceArea[ixOutput];
        // Node nodecenter = GetFaceCentroid( faceSecond, m_meshOutput->nodes );

        for( int p = 0; p < nP; p++ )
        {
            for( int q = 0; q < nP; q++ )
            {
                int ixOutputGlobal;
                if( fContinuous )
                    ixOutputGlobal = dataGLLNodes[p][q][ixOutput] - 1;
                else
                    ixOutputGlobal = ixOutput * nP * nP + p * nP + q;

                // Coordinate of the GLL point
                // Node nodeRef = m_meshOutput->nodes[ixOutputGlobal];

                Node nodeRef;
                ApplyLocalMap( faceSecond, m_meshOutput->nodes, dG[p], dG[q], nodeRef );

                const std::array< double, 3 > query = { nodeRef.x, nodeRef.y, nodeRef.z };

                // The radius for the search is the sqrt of the Jacobian
                const double radius = std::sqrt( dataGLLJacobian[p][q][ixOutput] );

                std::cout << "Searching elements within radius " << radius
                          << " for query point: " << query[0] << ", " << query[1] << ", " << query[2] << "\n";
                // Now let us search for the nearest elements within search radius
                auto results = radius_search_kdtree( tree, cloud, query, radius );

                // Find how many elements were found
                size_t nResults = results.size();

                // Newton-Cotes equal-weight method
                // const double dWeight  = dataGLLNodalArea[ixOutput] / nResults;
                const double dWeight = 1.0 / nResults;

                if (nResults == 0)
                {
                    printf( "No elements found within radius %f for query point: %f, %f, %f",
                                 radius, query[0], query[1], query[2] );
                    MB_CHK_SET_ERR( moab::MB_FAILURE, "No elements found within radius" );
                }

                for( auto ixFirstElement : results )
                {
                    std::cout << ixFirstElement << "\tFound " << nResults << " elements within radius " << radius
                                << " for query point: " << query[0] << ", " << query[1] << ", " << query[2] << "\n";
                    // std::cout << "\t[ " << ixOutputGlobal << "] Found association of point " << query[0] << ", "
                    //           << query[1] << ", " << query[2] << " to " << ixFirstElement << "\n";
                    if ( ixFirstElement < 0 || ixFirstElement >= source_vertices.size() )
                    {
                        printf( "Logic error: source element has to be between 0 and %d, but received %d for row %d\n",
                                     source_vertices.size(), ixFirstElement, ixOutputGlobal );
                        MB_CHK_SET_ERR( moab::MB_FAILURE, "Logic error: source element has to be between 0 and %d, but received %d for row %d\n",
                                     source_vertices.size(), ixFirstElement, ixOutputGlobal );
                    }
                    smatMap( ixOutputGlobal, ixFirstElement ) += dWeight;
                }
            }
        }
    }

    return moab::MB_SUCCESS;
}
*/

} // namespace moab

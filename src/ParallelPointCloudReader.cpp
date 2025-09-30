#include "ParallelPointCloudReader.hpp"
#include "ParallelPointCloudDistributor.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <unistd.h>
#include <limits>
// #include <pnetcdf>

namespace moab {

ParallelPointCloudReader::ParallelPointCloudReader(Interface* interface, ParallelComm* pcomm, EntityHandle mesh_set)
    : m_interface(interface), m_pcomm(pcomm), m_mesh_set(mesh_set), m_ncfile(nullptr), m_total_points(0), m_is_usgs_format(false) {
}

ParallelPointCloudReader::~ParallelPointCloudReader() {
    cleanup_netcdf();
}

ErrorCode ParallelPointCloudReader::configure(const ReadConfig& config) {
    // initialize the configuration parameters
    m_config = config;

    // Compute bounding boxes for mesh decomposition
    // ErrorCode rval = compute_mesh_bounding_boxes();
    // if (rval != MB_SUCCESS) {
    //     if (m_pcomm->rank() == 0) {
    //         std::cerr << "Error: Failed to compute mesh bounding boxes" << std::endl;
    //     }
    //     return rval;
    // }
    // For USGS format, compute bounding box directly in lat/lon space from mesh vertices
    BoundingBox lonlat_bbox;
    MB_CHK_SET_ERR(compute_lonlat_bounding_box_from_mesh(this->m_local_bbox), "Failed to compute local lon/lat bounding box");

    MB_CHK_SET_ERR(gather_all_bounding_boxes(m_all_bboxes), "Failed to gather all the bounding boxes");

    return MB_SUCCESS;
}

ErrorCode ParallelPointCloudReader::initialize_netcdf() {
    try {
        if (m_pcomm->rank() == 0) {
            std::cout << "Initializing NetCDF file: " << m_config.netcdf_filename << std::endl;
        }

        m_ncfile = nullptr;

#ifdef MOAB_HAVE_PNETCDF
        // if (m_config.use_collective_io)
        {
            try {
                m_ncfile = new PnetCDF::NcmpiFile(m_pcomm->comm(), m_config.netcdf_filename.c_str(), PnetCDF::NcmpiFile::read, PnetCDF::NcmpiFile::classic2);
                if (m_pcomm->rank() == 0) {
                    std::cout << "Successfully opened file with PnetCDF." << std::endl;
                }
            } catch (const PnetCDF::exceptions::NcmpiException& e) {
                if (m_ncfile) delete m_ncfile;
                m_ncfile = nullptr;
                if (m_pcomm->rank() == 0) {
                    std::cerr << "Error: PnetCDF opening failed: " << e.what() << ". This may indicate the file is not a PnetCDF file or a file system issue." << std::endl;
                }
                return MB_FAILURE;
            }
        }
#else
        if (m_pcomm->rank() == 0) {
            std::cerr << "Error: MOAB_HAVE_PNETCDF is not defined. This reader requires PnetCDF for parallel I/O." << std::endl;
        }
        return MB_FAILURE;
#endif

        if (!m_ncfile) {
            std::cerr << "Error: Could not open NetCDF file: " << m_config.netcdf_filename << std::endl;
            return MB_FAILURE;
        }

        // Choose reading strategy based on file format and configuration
        // Detect file format
        MB_CHK_ERR(detect_netcdf_format());

    } catch (const std::exception& e) {
        std::cerr << "An exception occurred during NetCDF initialization: " << e.what() << std::endl;
        return MB_FAILURE;
    }

    return MB_SUCCESS;
}


ErrorCode ParallelPointCloudReader::detect_netcdf_format() {
    try {
#ifdef MOAB_HAVE_PNETCDF
        if (m_pcomm->rank() == 0) {
            std::cout << "Detecting NetCDF format..." << std::endl;
        }

        // Get all dimensions and variables using the PnetCDF API
        const std::multimap<std::string, PnetCDF::NcmpiDim> dims_map = m_ncfile->getDims();
        const std::multimap<std::string, PnetCDF::NcmpiVar> vars_map = m_ncfile->getVars();

        if (m_pcomm->rank() == 0) {
            std::cout << "NetCDF file has " << dims_map.size() << " dimensions and " << vars_map.size() << " variables" << std::endl;
        }

        // Store all variables for later access
        for (const auto& var_pair : vars_map) {
            m_vars[var_pair.first] = var_pair.second;
        }

        // Check for USGS format: lat/lon dimensions and htopo variable
        bool has_lat = false, has_lon = false, has_htopo = false, has_fract = false;
        for (const auto& dim_pair : dims_map) {
            if (!dim_pair.first.compare("lat") || !dim_pair.first.compare("latitude"))
            {
              has_lat = true;
              lat_var_name = !dim_pair.first.compare("lat")  ? "lat" : "latitude";
            }
            else if (!dim_pair.first.compare("lon") || !dim_pair.first.compare("longitude"))
            {
              has_lon = true;
              lon_var_name = !dim_pair.first.compare("lon")  ? "lon" : "longitude";
            }
        }
        if (m_vars.count("htopo") || m_vars.count("Elevation")) {
          has_htopo = true;
          topo_var_name = m_vars.count("htopo") ? "htopo" : "Elevation";
        }
        if (m_vars.count("landfract") || m_vars.count("LandWater")) {
          has_fract = true;
          fract_var_name = m_vars.count("landfract") ? "landfract" : "LandWater";
        }

        if (has_lat && has_lon && has_htopo) {
            m_is_usgs_format = true;
            
            // Get dimension sizes
            PnetCDF::NcmpiDim lat_dim = m_ncfile->getDim(lat_var_name);
            PnetCDF::NcmpiDim lon_dim = m_ncfile->getDim(lon_var_name);
            nlats = lat_dim.getSize();
            nlons = lon_dim.getSize();
            m_total_points = nlats * nlons;
            
            // Set up configuration
            m_config.coord_var_names = {lon_var_name, lat_var_name};
            if (has_fract) {
                m_config.scalar_var_names = {topo_var_name, fract_var_name};
            } else {
                m_config.scalar_var_names = {topo_var_name};
            }
            
            if (m_pcomm->rank() == 0) {
                std::cout << "Detected USGS format NetCDF file" << std::endl;
                std::cout << "  Coordinate dimensions: " << lat_var_name << " (" << nlats << "), " 
                          << lon_var_name << " (" << nlons << ")" << std::endl;
                std::cout << "  Topography variable: " << topo_var_name << std::endl;
                if (has_fract) {
                    std::cout << "  Land fraction variable: " << fract_var_name << std::endl;
                }
            }
        } else {
            if (m_pcomm->rank() == 0) std::cout << "Not USGS format, detecting coordinate variables..." << std::endl;

            std::vector<std::string> lon_names = {"xc", "lon", "longitude", "x"};
            std::vector<std::string> lat_names = {"yc", "lat", "latitude", "y"};

            for (const auto& name : lon_names) {
                if (m_vars.count(name)) {
                    lon_var_name = name;
                    break;
                }
            }
            for (const auto& name : lat_names) {
                if (m_vars.count(name)) {
                    lat_var_name = name;
                    break;
                }
            }

            if (!lon_var_name.empty() && !lat_var_name.empty()) {
                m_config.coord_var_names = {lon_var_name, lat_var_name};

                PnetCDF::NcmpiVar coord_var = m_vars.at(lon_var_name);
                m_total_points = 1;
                // points.clear();
                m_unique_points.clear();
                for (const auto& dim : coord_var.getDims()) {
                    m_total_points *= dim.getSize();
                }
                nlats = coord_var.getDims()[0].getSize();
                nlons = coord_var.getDims()[1].getSize();

                // Automatically detect scalar variables
                std::vector<std::string> scalar_vars;
                for (const auto& var_pair : m_vars) {
                    const std::string& var_name = var_pair.first;
                    if (var_name != lon_var_name && var_name != lat_var_name && var_name.compare("xv") && var_name.compare("yv")) {
                        scalar_vars.push_back(var_name);
                    }
                }
                m_config.scalar_var_names = scalar_vars;

            } else {
                std::cerr << "Error: Could not find coordinate variables in NetCDF file" << std::endl;
                return MB_FAILURE;
            }
        }

        if (m_pcomm->rank() == 0) {
            std::cout << "Detected coordinate variables: " << lon_var_name << "(" << nlons << "), " << lat_var_name << "(" << nlats << ")" << std::endl;
        }

        if (m_pcomm->rank() == 0) {
            std::cout << "Total points in dataset: " << m_total_points << std::endl;
            for(const auto& var : m_config.scalar_var_names)
            {
                std::cout << "Found variable: " << var << std::endl;
            }
        }
        return MB_SUCCESS;
#else
        std::cerr << "Error: MOAB was not built with PnetCDF support" << std::endl;
        return MB_FAILURE;
#endif
    } catch (const std::exception& e) {
        std::cerr << "Error in detect_netcdf_format(): " << e.what() << std::endl;
        return MB_FAILURE;
    }
}

// ErrorCode ParallelPointCloudReader::read_usgs_format(PointData& points) {
//     try {
// #ifdef MOAB_HAVE_PNETCDF
//         // Get dimensions using PnetCDF API
//         PnetCDF::NcmpiDim lat_dim = m_ncfile->getDim("lat");
//         PnetCDF::NcmpiDim lon_dim = m_ncfile->getDim("lon");

//         const size_t nlat = lat_dim.getSize();
//         const size_t nlon = lon_dim.getSize();
//         m_total_points = nlat * nlon;

//         if (m_pcomm->rank() == 0) {
//             std::cout << "USGS format: " << nlat << " x " << nlon << " = " << m_total_points << " points" << std::endl;
//         }

//         // Get coordinate variables using PnetCDF API
//         PnetCDF::NcmpiVar lat_var = m_vars.at("lat");
//         PnetCDF::NcmpiVar lon_var = m_vars.at("lon");

//         // First, read coordinate arrays to determine spatial bounds
//         // This is a one-time read of 1D arrays, much smaller than the full 2D data
//         std::vector<double> lats(nlat);
//         std::vector<double> lons(nlon);

//         // Read coordinate arrays using PnetCDF collective I/O
//         std::vector<MPI_Offset> start = {0};
//         std::vector<MPI_Offset> lat_count = {static_cast<MPI_Offset>(nlat)};
//         std::vector<MPI_Offset> lon_count = {static_cast<MPI_Offset>(nlon)};

//         lat_var.getVar_all(start, lat_count, lats.data());
//         lon_var.getVar_all(start, lon_count, lons.data());

//         // For USGS format, compute bounding box directly in lat/lon space from mesh vertices
//         const BoundingBox& lonlat_bbox = this->m_local_bbox;

//         // Step 1: Each rank reads ntotalpoints/nranks points
//         size_t nlon_per_rank = nlon / m_pcomm->size();
//         size_t remainder = nlon % m_pcomm->size();

//         // Last rank gets the remainder nlon
//         size_t my_start_idx = m_pcomm->rank() * nlon_per_rank;
//         size_t my_count = nlon_per_rank;
//         if (m_pcomm->rank() == m_pcomm->size() - 1) {
//             my_count += remainder;
//         }

//         // Find the index ranges that intersect with our bounding box
//         size_t lat_start = 0, lat_chunk_count = nlat;
//         size_t lon_start = my_start_idx, lon_chunk_count = my_count;

//         // MB_CHK_ERR(find_spatial_index_range(lats, lonlat_bbox.min_coords[1], lonlat_bbox.max_coords[1],
//         //                                    lat_start, lat_chunk_count));
//         // MB_CHK_ERR(find_spatial_index_range(lons, lonlat_bbox.min_coords[0], lonlat_bbox.max_coords[0],
//         //                                    lon_start, lon_chunk_count));

//         // if (m_pcomm->rank() == 0)
//         {
//             std::cout << "Process " << m_pcomm->rank() << " reading lat[" << lat_start << ":"
//                       << lat_start + lat_chunk_count << "] lon[" << lon_start << ":"
//                       << lon_start + lon_chunk_count << "] = " << lat_chunk_count * lon_chunk_count << " points" << std::endl;
//         }

//         // Only read the subset of coordinates we need
//         std::vector<double> local_lats(lat_chunk_count);
//         std::vector<double> local_lons(lon_chunk_count);

//         // Read coordinate subsets using PnetCDF
//         std::vector<MPI_Offset> lat_subset_start = {static_cast<MPI_Offset>(lat_start)};
//         std::vector<MPI_Offset> lat_subset_count = {static_cast<MPI_Offset>(lat_chunk_count)};
//         std::vector<MPI_Offset> lon_subset_start = {static_cast<MPI_Offset>(lon_start)};
//         std::vector<MPI_Offset> lon_subset_count = {static_cast<MPI_Offset>(lon_chunk_count)};

//         lat_var.getVar_all(lat_subset_start, lat_subset_count, local_lats.data());
//         lon_var.getVar_all(lon_subset_start, lon_subset_count, local_lons.data());

//         // Create point cloud from the subset and track valid indices
//         // points.coordinates.reserve(lat_chunk_count * lon_chunk_count);
//         points.lonlat_coordinates.reserve(lat_chunk_count * lon_chunk_count);
//         std::vector<std::pair<size_t, size_t>> valid_indices; // Store (i,j) pairs for valid coordinates

//         for (size_t i = 0; i < lat_chunk_count; ++i) {
//             for (size_t j = 0; j < lon_chunk_count; ++j) {
//                 double lon = local_lons[j];
//                 double lat = local_lats[i];

//                 // Check if this lat/lon point is within our bounding box
//                 if (lon >= lonlat_bbox.min_coords[0] && lon <= lonlat_bbox.max_coords[0] &&
//                     lat >= lonlat_bbox.min_coords[1] && lat <= lonlat_bbox.max_coords[1]) {
//                     PointType current_point = {lon, lat};
//                     auto pit = m_unique_points.find(current_point);
//                     if (pit == m_unique_points.end()) {
//                         points.lonlat_coordinates.push_back(current_point);
//                         valid_indices.push_back({i, j});
//                         m_unique_points.insert(current_point);
//                     }
//                 }
//             }
//         }

//         // if (m_pcomm->rank() == 0)
//         {
//             std::cout << "Process " << m_pcomm->rank() << " found " << points.lonlat_coordinates.size()
//                       << " points within bounding box from " << lat_chunk_count * lon_chunk_count << " candidates" << std::endl;
//         }

//         // Read scalar variables (htopo, landfract) using PnetCDF
//         for (const std::string& var_name : m_config.scalar_var_names) {
//             auto var_it = m_vars.find(var_name);
//             if (var_it != m_vars.end()) {
//                 PnetCDF::NcmpiVar var = var_it->second;

//                 try {
//                     if (m_pcomm->rank() == 0) {
//                         std::cout << "Attempting to read variable '" << var_name
//                                   << "' subset: lat_start=" << lat_start << ", lon_start=" << lon_start
//                                   << ", lat_count=" << lat_chunk_count << ", lon_count=" << lon_chunk_count << std::endl;
//                     }

//                     // Check if the request size exceeds INT_MAX and implement chunked reading
//                     size_t total_elements = lat_chunk_count * lon_chunk_count;
//                     const size_t MAX_CHUNK_SIZE = 100000000; // 100M elements to stay well below INT_MAX

//                     std::vector<int> local_var_data;
//                     local_var_data.reserve(total_elements);

//                     if (total_elements > MAX_CHUNK_SIZE) {
//                         if (m_pcomm->rank() == 0) {
//                             std::cout << "Large dataset detected (" << total_elements
//                                       << " elements). Using chunked reading..." << std::endl;
//                         }

//                         // Read in chunks along the latitude dimension
//                         size_t lat_chunk_size = MAX_CHUNK_SIZE / lon_chunk_count;
//                         if (lat_chunk_size == 0) lat_chunk_size = 1;

//                         for (size_t lat_offset = 0; lat_offset < lat_chunk_count; lat_offset += lat_chunk_size) {
//                             size_t current_lat_count = std::min(lat_chunk_size, lat_chunk_count - lat_offset);
//                             size_t chunk_elements = current_lat_count * lon_chunk_count;

//                             std::vector<int> chunk_data(chunk_elements);
//                             std::vector<MPI_Offset> chunk_start = {
//                                 static_cast<MPI_Offset>(lat_start + lat_offset),
//                                 static_cast<MPI_Offset>(lon_start)
//                             };
//                             std::vector<MPI_Offset> chunk_count = {
//                                 static_cast<MPI_Offset>(current_lat_count),
//                                 static_cast<MPI_Offset>(lon_chunk_count)
//                             };

//                             var.getVar_all(chunk_start, chunk_count, chunk_data.data());

//                             // Append to main data vector
//                             local_var_data.insert(local_var_data.end(), chunk_data.begin(), chunk_data.end());

//                             if (m_pcomm->rank() == 0) {
//                                 std::cout << "Read chunk " << (lat_offset / lat_chunk_size + 1)
//                                           << " (" << chunk_elements << " elements)" << std::endl;
//                             }
//                         }
//                     } else {
//                         // Small enough to read in one operation
//                         local_var_data.resize(total_elements);
//                         std::vector<MPI_Offset> var_start = {static_cast<MPI_Offset>(lat_start), static_cast<MPI_Offset>(lon_start)};
//                         std::vector<MPI_Offset> var_count = {static_cast<MPI_Offset>(lat_chunk_count), static_cast<MPI_Offset>(lon_chunk_count)};
//                         var.getVar_all(var_start, var_count, local_var_data.data());
//                     }

//                     if (m_pcomm->rank() == 0) {
//                         std::cout << "Successfully read " << local_var_data.size() << " values for '" << var_name << "'" << std::endl;
//                     }

//                     // Extract data using the same valid indices from coordinate generation
//                     std::vector<double> filtered_data;
//                     filtered_data.reserve(valid_indices.size());

//                     for (const auto& idx_pair : valid_indices) {
//                         size_t i = idx_pair.first;
//                         size_t j = idx_pair.second;
//                         size_t data_idx = i * lon_chunk_count + j;
//                         filtered_data.push_back(local_var_data[data_idx]);
//                     }

//                     points.scalar_variables[var_name] = std::move(filtered_data);

//                     if (m_pcomm->rank() == 0) {
//                         std::cout << "Read " << filtered_data.size() << " values for variable '"
//                                   << var_name << "' (coordinates: " << points.lonlat_coordinates.size()
//                                   << ", lat_count: " << lat_chunk_count << ", lon_count: " << lon_chunk_count << ")" << std::endl;
//                     }

//                 } catch (const std::exception& e) {
//                     if (m_pcomm->rank() == 0) {
//                         std::cerr << "Warning: Failed to read variable '" << var_name
//                                   << "': " << e.what() << std::endl;
//                     }
//                 }
//             }
//         }

//         return MB_SUCCESS;
// #endif
//     } catch (const std::exception& e) {
//         std::cerr << "Error in read_usgs_format(): " << e.what() << std::endl;
//         return MB_FAILURE;
//     }
// }


ErrorCode ParallelPointCloudReader::compute_lonlat_bounding_box_from_mesh(BoundingBox& lonlat_bbox) {
    // Get all vertices from the mesh set
    Range vertices, elements;
    MB_CHK_ERR(m_interface->get_entities_by_type(m_mesh_set, MBVERTEX, vertices));
    MB_CHK_ERR(m_interface->get_entities_by_type(m_mesh_set, MBQUAD, elements));
    elements.merge(vertices);

    if (elements.empty()) {
        return MB_FAILURE;
    }

    // Get vertex coordinates
    std::vector<double> coords(elements.size() * 3);
    MB_CHK_ERR(m_interface->get_coords(elements, coords.data()));

    // Initialize bounding box with invalid values to detect if we find any valid coordinates
    lonlat_bbox.min_coords.fill(std::numeric_limits<double>::max());
    lonlat_bbox.max_coords.fill(std::numeric_limits<double>::lowest());

    // Convert each vertex from Cartesian to lat/lon and update bounding box
    for (size_t i = 0; i < elements.size(); ++i) {
        const double *coordinates = coords.data() + i * 3;

        CoordinateType lon, lat;
        MB_CHK_ERR(XYZtoRLL_Deg(coordinates, lon, lat));

        // Initialize bounding box on first valid coordinate
            // Update bounding box
        lonlat_bbox.min_coords[0] = std::min(lonlat_bbox.min_coords[0], lon);
        lonlat_bbox.max_coords[0] = std::max(lonlat_bbox.max_coords[0], lon);
        lonlat_bbox.min_coords[1] = std::min(lonlat_bbox.min_coords[1], lat);
        lonlat_bbox.max_coords[1] = std::max(lonlat_bbox.max_coords[1], lat);
    }

    // If no valid coordinates found, set a default global bounding box
    if (lonlat_bbox.min_coords[0] == std::numeric_limits<double>::max() || lonlat_bbox.min_coords[1] == std::numeric_limits<double>::max() || lonlat_bbox.max_coords[0] == std::numeric_limits<double>::lowest() || lonlat_bbox.max_coords[1] == std::numeric_limits<double>::lowest()) {
        MB_CHK_SET_ERR(MB_FAILURE, "No valid coordinates found in mesh");
    }

    // Add buffer factor: only needed for parallel computation
    if (m_pcomm->size() > 1)
        lonlat_bbox.expand(m_config.buffer_factor);

    // Ensure longitude stays within [0, 180] bounds and latitude within [-90, 90]
    // assert(lonlat_bbox.min_coords[0] >= 0.0);
    // assert(lonlat_bbox.min_coords[1] >= -90.0);
    // if(lonlat_bbox.max_coords[0] > 360.0)
    // std::cout << "Found longitude " << lonlat_bbox.max_coords[0] << " which is greater than 360" << std::endl;
    // assert(lonlat_bbox.max_coords[0] <= 360.0);
    // assert(lonlat_bbox.max_coords[1] <= 90.0);

    {
        std::cout << m_pcomm->rank() << ": Computed lat/lon bounding box: lon[" << lonlat_bbox.min_coords[0]
                  << ", " << lonlat_bbox.max_coords[0] << "] lat[" << lonlat_bbox.min_coords[1]
                  << ", " << lonlat_bbox.max_coords[1] << "]" << std::endl;
    }

    return MB_SUCCESS;
}

ErrorCode ParallelPointCloudReader::gather_all_bounding_boxes(std::vector<BoundingBox>& all_bboxes) {
    // int rank = m_pcomm->rank(); // Unused variable
    size_t size = m_pcomm->size();

    // Prepare local bounding box data for communication
    std::vector<double> local_bbox_data(DIM * 2);
    std::copy(m_local_bbox.min_coords.begin(), m_local_bbox.min_coords.end(), local_bbox_data.begin());
    std::copy(m_local_bbox.max_coords.begin(), m_local_bbox.max_coords.end(), local_bbox_data.begin() + DIM);

    // Gather all bounding boxes
    std::vector<double> all_bbox_data(size * DIM * 2);
    MPI_Allgather(local_bbox_data.data(), DIM * 2, MPI_DOUBLE, all_bbox_data.data(), DIM * 2, MPI_DOUBLE, m_pcomm->comm());

    // Store all bounding boxes
    all_bboxes.resize(size);
    for (size_t i = 0; i < size; ++i) {
        // Each rank's data: [min_coords, max_coords] = [DIM values, DIM values]
        size_t base_idx = i * DIM * 2;
        std::copy(all_bbox_data.begin() + base_idx,
                  all_bbox_data.begin() + base_idx + DIM,
                  all_bboxes[i].min_coords.begin());
        std::copy(all_bbox_data.begin() + base_idx + DIM,
                  all_bbox_data.begin() + base_idx + DIM * 2,
                  all_bboxes[i].max_coords.begin());
    }

    // Also store in member variable for backward compatibility
    m_all_bboxes = all_bboxes;

    return MB_SUCCESS;
}

ErrorCode ParallelPointCloudReader::read_coordinates_chunk(size_t start_idx, size_t count,
                                   std::vector<ParallelPointCloudReader::PointType>& coords) {
#ifdef MOAB_HAVE_PNETCDF
    if (m_config.coord_var_names.size() < 2) return MB_FAILURE;

    try {
        PnetCDF::NcmpiVar x_var = m_vars.at(m_config.coord_var_names[0]);
        PnetCDF::NcmpiVar y_var = m_vars.at(m_config.coord_var_names[1]);

        // Check variable dimensions to determine proper reading strategy
        int x_ndims = x_var.getDimCount();
        int y_ndims = y_var.getDimCount();

        if (m_pcomm->rank() == 0) {
            std::cout << "Coordinate variables: " << m_config.coord_var_names[0]
                      << " (dims=" << x_ndims << "), " << m_config.coord_var_names[1]
                      << " (dims=" << y_ndims << ")" << std::endl;
        }

        std::vector<MPI_Offset> start, read_count;

        if (x_ndims == 1 && y_ndims == 1) {
            // 1D coordinate arrays - could be USGS format or standard point arrays
            // Check if this is USGS format by examining coordinate variable names
            if (m_is_usgs_format) {
                // USGS format: lon and lat are separate 1D arrays defining a grid
                // Need to convert linear index to 2D grid indices

                // Get dimension sizes
                PnetCDF::NcmpiDim lat_dim = m_ncfile->getDim(lat_var_name);
                PnetCDF::NcmpiDim lon_dim = m_ncfile->getDim(lon_var_name);
                assert( nlats - lat_dim.getSize() == 0);
                assert( nlons - lon_dim.getSize() == 0);

                if (m_pcomm->rank() == 0) {
                    std::cout << "USGS coordinate grid: " << nlats << " lat x " << nlons << " lon" << std::endl;
                }

                // Read entire coordinate arrays (they are 1D and relatively small)
                std::vector<CoordinateType> lats(nlats), lons(nlons);

                std::vector<MPI_Offset> lat_start = {nlats_start};
                std::vector<MPI_Offset> lat_count = {nlats_count};
                std::vector<MPI_Offset> lon_start = {nlons_start};
                std::vector<MPI_Offset> lon_count = {nlons_count};

                y_var.getVar_all(lat_start, lat_count, lats.data());  // lat
                x_var.getVar_all(lon_start, lon_count, lons.data());  // lon

                // Generate coordinate pairs for the requested chunk
                coords.resize(count);
                for (size_t i = 0; i < count; ++i) {
                    size_t global_idx = start_idx + i;
                    size_t lat_idx = global_idx / nlons;  // Row index
                    size_t lon_idx = global_idx % nlons;  // Column index

                    if (lat_idx < nlats && lon_idx < nlons) {
                        coords[i] = {lons[lon_idx], lats[lat_idx]};
                    } else {
                        std::cout << "Error: Point " << global_idx << " is out of bounds" << std::endl;
                        coords[i] = {0.0, 0.0};  // Out of bounds
                    }
                }
                // // For USGS format, store the 1D lat/lon arrays instead of generating all pairs
                // // This saves memory for large grids
                // coords.resize(nlats_count * nlons_count);

                // // Store just the 1D arrays - they will be populated in read_local_chunk_distributed
                // // Here we just create placeholder coordinates for size tracking
                // size_t coord_idx = 0;
                // for (MPI_Offset ilat = 0; ilat < nlats_count; ++ilat) {
                //     for (MPI_Offset ilon = 0; ilon < nlons_count; ++ilon) {
                //         coords[coord_idx++] = {0.0, 0.0};  // Placeholder
                //     }
                // }

                return MB_SUCCESS;
            } else {
                // Standard 1D coordinate arrays - point-by-point coordinates
                start = {static_cast<MPI_Offset>(start_idx)};
                read_count = {static_cast<MPI_Offset>(count)};
            }
        } else if (x_ndims == 2 && y_ndims == 2) {
            // 2D coordinate arrays - need to handle ni/nj indexing
            // For domain files, coordinates are typically stored as (nj, ni) or (ni, nj)
            // We need to convert linear index to 2D indices

            // Get dimension sizes
            std::vector<PnetCDF::NcmpiDim> x_dims = x_var.getDims();
            std::vector<PnetCDF::NcmpiDim> y_dims = y_var.getDims();

            size_t dim0_size = x_dims[0].getSize();
            size_t dim1_size = x_dims[1].getSize();

            if (m_pcomm->rank() == 0) {
                std::cout << "2D coordinate arrays: " << dim0_size << " x " << dim1_size << std::endl;
            }

            // Read entire coordinate arrays for now (simpler approach)
            start = {0, 0};
            read_count = {static_cast<MPI_Offset>(dim0_size), static_cast<MPI_Offset>(dim1_size)};

            size_t total_coords = dim0_size * dim1_size;
            std::vector<CoordinateType> x_all(total_coords), y_all(total_coords);

            x_var.getVar_all(start, read_count, x_all.data());
            y_var.getVar_all(start, read_count, y_all.data());

            // Extract the requested subset
            coords.resize(count);
            for (size_t i = 0; i < count && (start_idx + i) < total_coords; ++i) {
                size_t idx = start_idx + i;
                coords[i] = {x_all[idx], y_all[idx]};
            }

            return MB_SUCCESS;
        } else {
            if (m_pcomm->rank() == 0) {
                std::cerr << "Unsupported coordinate variable dimensions: "
                          << x_ndims << ", " << y_ndims << std::endl;
            }
            return MB_FAILURE;
        }

        std::vector<CoordinateType> x_coords(count), y_coords(count);
        x_var.getVar_all(start, read_count, x_coords.data());
        y_var.getVar_all(start, read_count, y_coords.data());

        coords.resize(count);
        for (size_t i = 0; i < count; ++i) {
            coords[i] = {x_coords[i], y_coords[i]};
        }
        return MB_SUCCESS;

    } catch (const std::exception& e) {
        if (m_pcomm->rank() == 0) {
            std::cerr << "Error reading coordinates: " << e.what() << std::endl;
        }
        return MB_FAILURE;
    }
#else
    return MB_FAILURE;
#endif
}

template<typename T>
ErrorCode ParallelPointCloudReader::read_scalar_variable_chunk(const std::string& var_name, size_t start_idx,
                                       size_t count, std::vector<T>& data) {
#ifdef MOAB_HAVE_PNETCDF
    auto var_it = m_vars.find(var_name);
    if (var_it == m_vars.end()) return MB_FAILURE;

    try {
        PnetCDF::NcmpiVar var = var_it->second;
        int ndims = var.getDimCount();

        if (m_pcomm->rank() == 0) {
            std::cout << "Reading scalar variable '" << var_name << "' with " << ndims << " dimensions" << std::endl;
        }

        std::vector<MPI_Offset> start, read_count;

        if (ndims == 1) {
            // 1D variable - standard case
            start = {static_cast<MPI_Offset>(start_idx)};
            read_count = {static_cast<MPI_Offset>(count)};

            data.resize(count);
            var.getVar_all(start, read_count, data.data());

        } else if (ndims == 2) {
            // 2D variable - handle USGS format with spatial chunking
            std::vector<PnetCDF::NcmpiDim> dims = var.getDims();
            size_t dim0_size = dims[0].getSize();
            size_t dim1_size = dims[1].getSize();
            size_t total_size = dim0_size * dim1_size;

            // Check if this is USGS format (lat x lon grid)
            if (m_is_usgs_format || (dims[0].getName() == "lat" && dims[1].getName() == "lon")) {
                // USGS format: use spatial filtering to read only needed subset
                // This should have been set up in read_usgs_format()

                // For now, return failure to avoid reading the entire massive dataset
                // if (m_pcomm->rank() == 0) {
                //     std::cout << "Skipping large USGS 2D variable '" << var_name
                //               << "' (" << dim0_size << " x " << dim1_size << " = " << total_size << " points)" << std::endl;
                // }
                // return MB_FAILURE;
                size_t total_elements_to_read = nlats_count * nlons_count;
                const size_t MAX_ELEMENTS_PER_CHUNK = 500*1000*1000; // 100M elements, well below INT_MAX

                std::vector<T> temp_buffer;
                temp_buffer.reserve(total_elements_to_read);

                if (total_elements_to_read > MAX_ELEMENTS_PER_CHUNK) {
                    if (m_pcomm->rank() == 0) {
                        std::cout << "Reading variable '" << var_name << "' in chunks to avoid overflow." << std::endl;
                    }

                    size_t lat_chunk_size = MAX_ELEMENTS_PER_CHUNK / nlons_count;
                    if (lat_chunk_size == 0) lat_chunk_size = 1; // Ensure progress

                    for (MPI_Offset lat_offset = 0; lat_offset < nlats_count; lat_offset += lat_chunk_size) {
                        MPI_Offset current_lat_count = std::min(static_cast<MPI_Offset>(lat_chunk_size), nlats_count - lat_offset);
                        size_t chunk_elements = current_lat_count * nlons_count;
                        if (m_pcomm->rank() == 0) {
                            std::cout << "\tReading latitude chunk from " << nlats_start + lat_offset << " to " << nlats_start + lat_offset + current_lat_count << "." << std::endl;
                        }

                        std::vector<T> chunk_buffer(chunk_elements);
                        start = {nlats_start + lat_offset, nlons_start};
                        read_count = {current_lat_count, nlons_count};

                        var.getVar_all(start, read_count, chunk_buffer.data());
                        temp_buffer.insert(temp_buffer.end(), chunk_buffer.begin(), chunk_buffer.end());
                    }
                } else {
                    // Read all at once
                    temp_buffer.resize(total_elements_to_read);
                    start = {nlats_start, nlons_start};
                    read_count = {nlats_count, nlons_count};
                    var.getVar_all(start, read_count, temp_buffer.data());
                }

                // The temp_buffer now contains the raw 2D data block for this rank's slice.
                // The caller (`read_points_distributed`) will be responsible for filtering this data
                // using the valid_indices it generates during coordinate processing.
                data = std::move(temp_buffer);
            } else {
                // Standard 2D variable - read entire array and extract subset
                start = {0, 0};
                read_count = {static_cast<MPI_Offset>(dim0_size), static_cast<MPI_Offset>(dim1_size)};

                std::vector<T> all_data(total_size);
                var.getVar_all(start, read_count, all_data.data());

                // Extract the requested subset
                data.resize(count);
                for (size_t i = 0; i < count && (start_idx + i) < total_size; ++i) {
                    data[i] = all_data[start_idx + i];
                }
            }

        } else if (ndims == 3) {
            // 3D variable - skip for now (likely vertex coordinates or time-dependent data)
            if (m_pcomm->rank() == 0) {
                std::cout << "WARNING:Skipping 3D variable '" << var_name << "' (not supported for scalar data)" << std::endl;
            }
            return MB_SUCCESS;
        } else {
            if (m_pcomm->rank() == 0) {
                std::cerr << "Unsupported variable dimensions: " << ndims << " for variable " << var_name << std::endl;
            }
            return MB_FAILURE;
        }

        return MB_SUCCESS;
    } catch (const std::exception& e) {
        if (m_pcomm->rank() == 0) {
            std::cerr << "Error reading scalar variable '" << var_name << "': " << e.what() << std::endl;
        }
        return MB_FAILURE;
    }
#else
    return MB_FAILURE;
#endif
}


// moab::ErrorCode moab::ParallelPointCloudReader::read_and_distribute_root_based(PointData& local_points) {
//     if (m_pcomm->rank() == 0) {
//         std::cout << "Using root-based buffered distribution system" << std::endl;
//     }

//     // Step 1: Gather all bounding boxes on root
//     MB_CHK_ERR(gather_all_bounding_boxes_on_root());

//     // Step 2: Root reads and distributes data, also accumulates its own data
//     if (m_pcomm->rank() == 0) {
//         MB_CHK_ERR(root_read_and_distribute_points(local_points));
//     } else {
//         // Step 3: Non-root ranks receive their distributed data
//         MB_CHK_ERR(receive_distributed_data(local_points));
//     }

//     return MB_SUCCESS;
// }

// moab::ErrorCode moab::ParallelPointCloudReader::gather_all_bounding_boxes_on_root() {
//     int num_ranks = m_pcomm->size();

//     // Prepare local bounding box data for sending
//     std::vector<double> local_bbox_data(6);
//     for (int i = 0; i < 3; i++) {
//         local_bbox_data[i] = m_local_bbox.min_coords[i];
//         local_bbox_data[i + 3] = m_local_bbox.max_coords[i];
//     }

//     if (m_pcomm->rank() == 0) {
//         m_all_bboxes.resize(num_ranks);
//         std::vector<double> all_bbox_data(num_ranks * 6);

//         // Gather all bounding boxes
//         MPI_Gather(local_bbox_data.data(), 6, MPI_DOUBLE,
//                    all_bbox_data.data(), 6, MPI_DOUBLE, 0, m_pcomm->comm());

//         // Convert to BoundingBox structures
//         for (int rank = 0; rank < num_ranks; rank++) {
//             for (int i = 0; i < 3; i++) {
//                 m_all_bboxes[rank].min_coords[i] = all_bbox_data[rank * 6 + i];
//                 m_all_bboxes[rank].max_coords[i] = all_bbox_data[rank * 6 + i + 3];
//             }
//         }

//         std::cout << "Gathered bounding boxes from " << num_ranks << " ranks" << std::endl;
//     } else {
//         // Non-root ranks just send their bounding box
//         MPI_Gather(local_bbox_data.data(), 6, MPI_DOUBLE,
//                    nullptr, 0, MPI_DOUBLE, 0, m_pcomm->comm());
//     }

//     return MB_SUCCESS;
// }

// moab::ErrorCode moab::ParallelPointCloudReader::root_read_and_distribute_points(PointData& root_local_points) {
//     if (m_pcomm->rank() != 0) return MB_SUCCESS;

//     std::cout << "Root process reading and distributing " << m_total_points << " points" << std::endl;

//     int num_ranks = m_pcomm->size();
//     std::vector<PointBuffer> rank_buffers(num_ranks);

//     // Initialize root's local data
//     root_local_points.clear();
//     for (const auto& var_name : m_config.scalar_var_names) {
//         root_local_points.scalar_variables[var_name] = std::vector<double>();
//     }

//     // Initialize scalar and vector variable buffers for all ranks
//     for (int rank = 0; rank < num_ranks; rank++) {
//         for (const auto& var_name : m_config.scalar_var_names) {
//             rank_buffers[rank].scalar_variables[var_name] = std::vector<double>();
//         }
//     }

//     // Read data in chunks and distribute
//     std::vector<std::array<double, 3>> chunk_coords;
//     std::unordered_map<std::string, std::vector<double>> chunk_scalars;
//     std::unordered_map<std::string, std::vector<double>> chunk_vectors;

//     size_t points_processed = 0;

//     while (points_processed < m_total_points) {
//         size_t chunk_size = std::min(m_config.chunk_size, m_total_points - points_processed);

//         // Read coordinates
//         chunk_coords.clear();
//         if (read_coordinates_chunk(points_processed, chunk_size, chunk_coords) != MB_SUCCESS) {
//             std::cerr << "Error reading coordinates chunk at " << points_processed << std::endl;
//             return MB_FAILURE;
//         }

//         // Read scalar variables
//         for (const auto& var_name : m_config.scalar_var_names) {
//             chunk_scalars[var_name].clear();
//             if (read_scalar_variable_chunk(var_name, points_processed, chunk_size, chunk_scalars[var_name]) != MB_SUCCESS) {
//                 std::cerr << "Error reading scalar variable " << var_name << " at " << points_processed << std::endl;
//                 return MB_FAILURE;
//             }
//         }

//         // Distribute points to appropriate ranks
//         for (size_t i = 0; i < chunk_coords.size(); i++) {
//             std::vector<int> owner_ranks;
//             if (determine_point_owners(chunk_coords[i], owner_ranks) != MB_SUCCESS) {
//                 continue; // Skip points that don't belong to any rank
//             }

//             // Add point to all owner ranks
//             for (int rank : owner_ranks) {
//                 if (rank == 0) {
//                     // Add directly to root's local data
//                     root_local_points.coordinates.push_back(chunk_coords[i]);

//                     // Add scalar data
//                     for (const auto& var_name : m_config.scalar_var_names) {
//                         root_local_points.scalar_variables[var_name].push_back(chunk_scalars[var_name][i]);
//                     }

//                 } else {
//                     // Add to buffer for other ranks
//                     rank_buffers[rank].coordinates.push_back(chunk_coords[i]);

//                     // Add scalar data
//                     for (const auto& var_name : m_config.scalar_var_names) {
//                         rank_buffers[rank].scalar_variables[var_name].push_back(chunk_scalars[var_name][i]);
//                     }

//                 }
//             }
//         }

//         points_processed += chunk_size;

//         // Send buffers if they exceed the size limit
//         for (int rank = 0; rank < num_ranks; rank++) {
//             if (rank_buffers[rank].memory_size() >= m_buffer_size_per_rank) {
//                 if (send_buffered_data_to_ranks(rank_buffers) != MB_SUCCESS) {
//                     std::cerr << "Error sending buffered data" << std::endl;
//                     return MB_FAILURE;
//                 }
//                 // Clear buffers after sending
//                 for (int r = 0; r < num_ranks; r++) {
//                     rank_buffers[r].clear();
//                     // Reinitialize variable maps
//                     for (const auto& var_name : m_config.scalar_var_names) {
//                         rank_buffers[r].scalar_variables[var_name] = std::vector<double>();
//                     }
//                 }
//                 break; // Send all buffers at once, then continue
//             }
//         }

//         if (points_processed % (m_config.chunk_size * 10) == 0) {
//             std::cout << "Processed " << points_processed << " / " << m_total_points
//                       << " points (" << (100.0 * points_processed / m_total_points) << "%)" << std::endl;
//         }
//     }

//     // Send any remaining buffered data
//     if (send_buffered_data_to_ranks(rank_buffers) != MB_SUCCESS) {
//         std::cerr << "Error sending final buffered data" << std::endl;
//         return MB_FAILURE;
//     }

//     // Send termination signal to all ranks
//     for (int rank = 1; rank < num_ranks; rank++) {
//         int termination_signal = -1;
//         MPI_Send(&termination_signal, 1, MPI_INT, rank, 0, m_pcomm->comm());
//     }

//     std::cout << "Root process completed data distribution. Root has "
//               << root_local_points.coordinates.size() << " local points" << std::endl;
//     return MB_SUCCESS;
// }

moab::ErrorCode moab::ParallelPointCloudReader::determine_point_owners(const PointType& point, std::vector<int>& owner_ranks) {
    owner_ranks.clear();

    for (int rank = 0; rank < static_cast<int>(m_all_bboxes.size()); rank++) {
        if (m_all_bboxes[rank].contains(point)) {
            owner_ranks.push_back(rank);
        }
    }

    return MB_SUCCESS;
}

moab::ErrorCode moab::ParallelPointCloudReader::read_and_redistribute_distributed(PointData& local_points) {
    if (m_pcomm->rank() == 0) {
        std::cout << "=== Starting distributed reading and redistribution ===" << std::endl;
    }

    // Step 1: Each rank reads ntotalpoints/nranks points
    size_t points_per_rank = (nlons / m_pcomm->size());
    size_t remainder = (nlons % m_pcomm->size());

    // Last rank gets the remainder points
    size_t my_count = points_per_rank;
    std::cout << "Remainder: " << remainder << std::endl;
    if (m_pcomm->rank() < remainder) {
        my_count++;
    }

    size_t my_start_idx = points_per_rank * (m_pcomm->rank());
    for (size_t irank = 0; irank < m_pcomm->rank(); irank++) {
        if (irank < remainder) {
            my_start_idx++;
        }
    }

    assert(my_start_idx >=0 && my_count > 0);

    nlats_start = 0;
    nlats_count = nlats;
    nlons_start = my_start_idx;
    nlons_count = my_count;

    // now let us scale by number of latitudes
    points_per_rank *= nlats;
    my_count *= nlats;
    my_start_idx *= nlats;

    if (m_pcomm->rank() == 0) {
        std::cout << "Each rank reading ~" << points_per_rank << " points" << std::endl;
        std::cout << "Total points: " << m_total_points << ", ranks: " << m_pcomm->size() << std::endl;
    }

    std::cout << "Rank " << m_pcomm->rank() << " reading points [" << my_start_idx
              << ", " << (my_start_idx + my_count - 1) << "] = " << my_count << " points" << std::endl;

    // Step 2: Read my chunk of data
    MB_CHK_ERR(read_local_chunk_distributed(my_start_idx, my_count, local_points));

    // Note: Bounding box is already computed from H5M mesh in configure()
    // We use the mesh bounding box (in lat/lon) to filter relevant NetCDF points

    // Step 3: Bounding boxes are now populated in m_all_bboxes

    // Step 4: Redistribute points based on bounding box ownership
    // MB_CHK_ERR(redistribute_points_by_ownership(initial_points, local_points));

    // Gather statistics
    size_t local_count = local_points.size();
    std::vector<size_t> all_counts(m_pcomm->size());
    MPI_Gather(&local_count, 1, MPI_UNSIGNED_LONG, all_counts.data(), 1, MPI_UNSIGNED_LONG, 0, m_pcomm->comm());

    size_t total_points = 0;
    if (m_pcomm->rank() == 0) {
        for (size_t irank = 0; irank < m_pcomm->size(); ++irank) {
            total_points += all_counts[irank];
            std::cout << "Rank " << irank << " final points: " << all_counts[irank] << std::endl;
        }
        std::cout << "Total points before redistribution: " << total_points << std::endl;
    }

    // let us delete the file handle so that we can do
    // everything else in memory
    delete m_ncfile;
    m_ncfile = nullptr;

    if (m_pcomm->size() > 1) // if we read the data in parallel, redistribute based on ownership
    {
        // Crystal router-based redistribution using mesh bounding boxes
        ParallelPointCloudDistributor distributor(m_interface, m_pcomm, m_is_usgs_format);
        // Create configuration options to control the behavior
        ParallelPointCloudDistributor::CrystalRouterConfig dist_config;
        dist_config.allow_multiple_ownership = true;

        // Limit buffer factor for large datasets to prevent MPI overflow
        // double safe_buffer_factor = m_config.buffer_factor;
        // if (m_pcomm->size() > 2 && m_total_points > 1000000) {
        //     // For large datasets with many ranks, reduce buffer factor to prevent overflow
        //     safe_buffer_factor = std::min(static_cast<double>(m_config.buffer_factor), 0.01); // Max 1% buffer
        //     if (m_pcomm->rank() == 0) {
        //         std::cout << "Large dataset detected (" << m_total_points << " points, "
        //                   << m_pcomm->size() << " ranks). Reducing buffer factor from "
        //                   << m_config.buffer_factor << " to " << safe_buffer_factor
        //                   << " to prevent MPI overflow." << std::endl;
        //     }
        // }

        // dist_config.bbox_expansion_factor = safe_buffer_factor; // expansion for better coverage
        dist_config.enable_statistics = true;

        distributor.configure(dist_config);

        // Gather all mesh bounding boxes from all ranks
        // std::vector<ParallelPointCloudReader::BoundingBox> all_bboxes;
        // MB_CHK_ERR(reader.gather_all_bounding_boxes(all_bboxes));
        const auto& all_bboxes = this->m_all_bboxes;

        ParallelPointCloudReader::PointData redistributed_points;
        ParallelPointCloudDistributor::DistributionStats stats;

        auto redist_start = std::chrono::high_resolution_clock::now();

        // ErrorCode rval = distributor.redistribute_points_crystal_router(local_points, all_bboxes, redistributed_points, stats);
        ErrorCode rval = distributor.redistribute_points_batched(local_points, all_bboxes, redistributed_points, stats);
        if (rval != MB_SUCCESS) {
            if (m_pcomm->rank() == 0) {
                std::cerr << "Crystal router redistribution failed, likely due to MPI buffer overflow." << std::endl;
                std::cerr << "Consider using root-based distribution (use_root_based_distribution=true)" << std::endl;
                std::cerr << "or reducing buffer_factor for large datasets." << std::endl;
            }
            return rval;
        }

        auto redist_end = std::chrono::high_resolution_clock::now();
        auto redist_duration = std::chrono::duration_cast<std::chrono::milliseconds>(redist_end - redist_start);

        if (m_pcomm->rank() == 0) {
            std::cout << "\n=== Crystal Router Redistribution Results ===" << std::endl;
            std::cout << "Communication time: " << redist_duration.count() /*stats.communication_time_ms*/ << " ms" << std::endl;
            std::cout << "Points sent: " << stats.points_sent << ", Points received: " << stats.points_received << std::endl;
            std::cout << "Total transfers: " << stats.total_transfers << std::endl;
        }

        // Process points (compute some statistics)
        if (local_points.size() > 0 && m_config.print_statistics) {
            if (m_pcomm->rank() == 0) {
                std::cout << "\n=== Scalar Variable Statistics ===" << std::endl;
            }

            // Compute statistics for each scalar variable
            for (const auto& var_pair : redistributed_points.d_scalar_variables) {
                const std::string& var_name = var_pair.first;
                const std::vector<double>& values = var_pair.second;

                double sum_val = 0.0, min_val = 1e10, max_val = -1e10;
                for (double val : values) {
                    sum_val += val;
                    min_val = std::min(min_val, val);
                    max_val = std::max(max_val, val);
                }

                // Global reduction for statistics
                double global_stats[3] = {sum_val, min_val, -max_val}; // Negative for max reduction
                double reduced_stats[3];
                MPI_Allreduce(global_stats, reduced_stats, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                MPI_Allreduce(global_stats + 1, reduced_stats + 1, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
                MPI_Allreduce(global_stats + 2, reduced_stats + 2, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
                reduced_stats[2] = -reduced_stats[2]; // Convert back to max

                if (m_pcomm->rank() == 0) {
                    std::cout << var_name << " - Average: " << reduced_stats[0] / redistributed_points.size()
                              << ", Min: " << reduced_stats[1]
                              << ", Max: " << reduced_stats[2] << std::endl;
                }
            }
        }

        // Use redistributed points for further processing
        local_points.clear();
        local_points = redistributed_points;
    }

    return MB_SUCCESS;
}

moab::ErrorCode moab::ParallelPointCloudReader::read_local_chunk_distributed(size_t start_idx, size_t count, PointData& chunk_data) {
    try {
        chunk_data.clear();
        chunk_data.reserve(count);

        // For USGS format, read and store 1D lat/lon arrays efficiently
        if (m_is_usgs_format) {
            // Read the actual 1D coordinate arrays for this rank's chunk
            PnetCDF::NcmpiVar lon_var = m_vars.at(lon_var_name);
            PnetCDF::NcmpiVar lat_var = m_vars.at(lat_var_name);

            chunk_data.latitudes.resize(nlats_count);
            chunk_data.longitudes.resize(nlons_count);

            std::vector<MPI_Offset> lat_start = {nlats_start};
            std::vector<MPI_Offset> lat_count = {nlats_count};
            std::vector<MPI_Offset> lon_start = {nlons_start};
            std::vector<MPI_Offset> lon_count = {nlons_count};

            lat_var.getVar_all(lat_start, lat_count, chunk_data.latitudes.data());
            lon_var.getVar_all(lon_start, lon_count, chunk_data.longitudes.data());

            // Mark as structured grid - coordinates will be computed on-the-fly
            chunk_data.is_structured_grid = true;

            if (m_pcomm->rank() == 0) {
                std::cout << "USGS format: Stored " << chunk_data.latitudes.size() << " latitudes and "
                          << chunk_data.longitudes.size() << " longitudes (total grid points: "
                          << chunk_data.size() << ", computed on-the-fly)" << std::endl;
            }
        } else {
            // Standard format: read coordinates directly into explicit storage
            MB_CHK_ERR(read_coordinates_chunk(start_idx, count, chunk_data.lonlat_coordinates));
            chunk_data.is_structured_grid = false;
        }

        // Read scalar variables with buffered reading for large datasets
        for (const auto& var_name : m_config.scalar_var_names) {

            if (m_is_usgs_format) {
                std::vector<int> scalar_data;
                MB_CHK_ERR(read_scalar_variable_chunk(var_name, start_idx, count, scalar_data));

                // The scalar data is read in the same order as coordinates, so no filtering needed
                if (scalar_data.size() == 0) continue; // nothing to do.

                chunk_data.i_scalar_variables[var_name] = std::move(scalar_data);

                if (chunk_data.i_scalar_variables[var_name].size() != chunk_data.size()) {
                    std::cerr << "WARNING: Scalar variable '" << var_name << "' has a different size ("
                            << chunk_data.i_scalar_variables[var_name].size() << ") than the total points ("
                            << chunk_data.size() << "). Data may be misaligned." << std::endl;
                }
            } else {
                std::vector<double> scalar_data;
                MB_CHK_ERR(read_scalar_variable_chunk(var_name, start_idx, count, scalar_data));

                // The scalar data is read in the same order as coordinates, so no filtering needed
                if (scalar_data.size() == 0) continue; // nothing to do.

                chunk_data.d_scalar_variables[var_name] = std::move(scalar_data);

                if (chunk_data.d_scalar_variables[var_name].size() != chunk_data.size()) {
                    std::cerr << "WARNING: Scalar variable '" << var_name << "' has a different size ("
                            << chunk_data.d_scalar_variables[var_name].size() << ") than the total points ("
                            << chunk_data.size() << "). Data may be misaligned." << std::endl;
                }
            }
        }

        std::cout << "Rank " << m_pcomm->rank() << " read " << chunk_data.size()
                  << " points with " << chunk_data.d_scalar_variables.size() << " scalar variables" << std::endl;

        return MB_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "Error in read_local_chunk_distributed: " << e.what() << std::endl;
        return MB_FAILURE;
    }
}

// moab::ErrorCode moab::ParallelPointCloudReader::redistribute_points_by_ownership(PointData& initial_points, PointData& final_points) {
//     if (m_pcomm->rank() == 0) {
//         std::cout << "Starting point redistribution based on bounding box ownership" << std::endl;
//     }

//     // Step 1: Determine which points this rank needs to send to which other ranks
//     // Use closest bounding box centroid to distribute points more evenly
//     std::vector<std::vector<size_t>> points_to_send(m_pcomm->size());

//     // Debug: Print first few points and bounding boxes to understand coordinate system
//     if (m_pcomm->rank() == 0 && initial_points.coordinates.size() > 0) {
//         std::cout << "DEBUG: First 3 points (lat/lon):" << std::endl;
//         for (size_t i = 0; i < std::min(size_t(3), initial_points.lonlat_coordinates.size()); ++i) {
//             const auto& pt = initial_points.lonlat_coordinates[i];
//             std::cout << "  Point " << i << ": lon=" << pt[0] << ", lat=" << pt[1] << std::endl;
//         }

//         std::cout << "DEBUG: Bounding boxes (lat/lon):" << std::endl;
//         for (size_t irank = 0; irank < m_pcomm->size(); ++irank) {
//             const auto& bbox = m_all_bboxes[irank];
//             std::cout << "  Rank " << irank << " bbox: lon[" << bbox.min_coords[0] << ", " << bbox.max_coords[0]
//                       << "] lat[" << bbox.min_coords[1] << ", " << bbox.max_coords[1] << "]" << std::endl;
//         }
//     }

//     for (size_t i = 0; i < initial_points.lonlat_coordinates.size(); ++i) {
//         const auto& lonlat_point = initial_points.lonlat_coordinates[i];

//         for (size_t irank = 0; irank < m_pcomm->size(); ++irank) {
//             // Only consider ranks whose bounding boxes actually contain this point (in lat/lon space)
//             if (m_all_bboxes[irank].contains(lonlat_point)) {
//                 points_to_send[irank].push_back(i);
//             }
//         }
//     }

//     // Step 2: Exchange send/receive counts
//     std::vector<int> send_counts(m_pcomm->size(), 0);
//     std::vector<int> recv_counts(m_pcomm->size(), 0);

//     // Count points to send to each rank
//     for (size_t irank = 0; irank < m_pcomm->size(); ++irank) {
//         send_counts[irank] = static_cast<int>(points_to_send[irank].size());
//     }

//     // All-to-all exchange: send_counts[i] from this rank becomes recv_counts[this_rank] on rank i
//     MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

//     // Calculate total points we'll receive
//     size_t total_recv_points = 0;
//     for (size_t irank = 0; irank < m_pcomm->size(); ++irank) {
//         total_recv_points += recv_counts[irank];
//     }

//     // Verify that we're sending exactly the number of points we read
//     size_t total_points_to_send = 0;
//     for (size_t irank = 0; irank < m_pcomm->size(); ++irank) {
//         total_points_to_send += send_counts[irank];
//     }

//     std::cout << "Rank " << m_pcomm->rank() << " read " << initial_points.coordinates.size()
//               << " points, will send " << total_points_to_send << " points distributed as: ";
//     for (size_t irank = 0; irank < m_pcomm->size(); ++irank) {
//         std::cout << send_counts[irank] << " ";
//     }
//     std::cout << std::endl;

//     // if (total_points_to_send != initial_points.coordinates.size()) {
//     //     std::cerr << "ERROR: Rank " << m_pcomm->rank() << " point count mismatch! Read "
//     //               << initial_points.coordinates.size() << " but sending " << total_points_to_send << std::endl;
//     //     return MB_FAILURE;
//     // }

//     std::cout << "Rank " << m_pcomm->rank() << " will receive " << total_recv_points
//               << " points total from ranks: ";
//     for (size_t irank = 0; irank < m_pcomm->size(); ++irank) {
//         if (recv_counts[irank] > 0) std::cout << irank << "(" << recv_counts[irank] << ") ";
//     }
//     std::cout << std::endl;

//     // Step 3: Handle local data first (points staying on same rank)
//     final_points.clear();
//     final_points.reserve(total_recv_points);

//     if (send_counts[m_pcomm->rank()] > 0) {
//         for (size_t pt_idx : points_to_send[m_pcomm->rank()]) {
//             final_points.coordinates.push_back(initial_points.coordinates[pt_idx]);
//             final_points.lonlat_coordinates.push_back(initial_points.lonlat_coordinates[pt_idx]);

//             for (const auto& var_name : m_config.scalar_var_names) {
//                 auto var_it = initial_points.scalar_variables.find(var_name);
//                 if (var_it != initial_points.scalar_variables.end()) {
//                     final_points.scalar_variables[var_name].push_back(var_it->second[pt_idx]);
//                 }
//             }
//         }
//     }

//     // Step 4: Exchange data with other ranks using non-blocking point-to-point communication
//     size_t num_scalars = m_config.scalar_var_names.size();
//     size_t point_data_size = 2 + num_scalars; // 2 for lon/lat + scalars

//     std::vector<MPI_Request> requests;
//     std::vector<std::vector<double>> send_buffers(m_pcomm->size());
//     std::vector<std::vector<double>> recv_buffers(m_pcomm->size());

//     // Post all non-blocking receives
//     for (size_t irank = 0; irank < m_pcomm->size(); ++irank) {
//         if (irank == m_pcomm->rank()) continue;
//         if (recv_counts[irank] > 0) {
//             recv_buffers[irank].resize(recv_counts[irank] * point_data_size);
//             MPI_Request req;
//             MPI_Irecv(recv_buffers[irank].data(), recv_counts[irank] * point_data_size, MPI_DOUBLE,
//                       irank, 0, MPI_COMM_WORLD, &req);
//             requests.push_back(req);
//         }
//     }

//     // Post all non-blocking sends
//     for (size_t irank = 0; irank < m_pcomm->size(); ++irank) {
//         if (irank == m_pcomm->rank()) continue;
//         if (send_counts[irank] > 0) {
//             send_buffers[irank].resize(send_counts[irank] * point_data_size);
//             for (size_t i = 0; i < points_to_send[irank].size(); ++i) {
//                 size_t pt_idx = points_to_send[irank][i];
//                 send_buffers[irank][i * point_data_size + 0] = initial_points.lonlat_coordinates[pt_idx][0];
//                 send_buffers[irank][i * point_data_size + 1] = initial_points.lonlat_coordinates[pt_idx][1];

//                 for (size_t j = 0; j < num_scalars; ++j) {
//                     const auto& var_name = m_config.scalar_var_names[j];
//                     auto var_it = initial_points.scalar_variables.find(var_name);
//                     if (var_it != initial_points.scalar_variables.end() && pt_idx < var_it->second.size()) {
//                         send_buffers[irank][i * point_data_size + 2 + j] = var_it->second[pt_idx];
//                     } else {
//                         send_buffers[irank][i * point_data_size + 2 + j] = 0.0; // Default value for missing variables
//                     }
//                 }
//             }
//             MPI_Request req;
//             MPI_Isend(send_buffers[irank].data(), send_counts[irank] * point_data_size, MPI_DOUBLE,
//                       irank, 0, MPI_COMM_WORLD, &req);
//             requests.push_back(req);
//         }
//     }

//     // Wait for all communication to complete
//     if (!requests.empty()) {
//         MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
//     }

//     // Process received data
//     for (size_t irank = 0; irank < m_pcomm->size(); ++irank) {
//         if (irank == m_pcomm->rank()) continue;
//         if (recv_counts[irank] > 0) {
//             for (int i = 0; i < recv_counts[irank]; ++i) {
//                 std::array<double, 2> lonlat_coord = {
//                     recv_buffers[irank][i * point_data_size + 0],
//                     recv_buffers[irank][i * point_data_size + 1]
//                 };
//                 final_points.lonlat_coordinates.push_back(lonlat_coord);

//                 // Recompute Cartesian coordinates
//                 std::array<double, 3> cart_coord = {lonlat_coord[0], lonlat_coord[1], 0.0};
//                 RLLtoXYZ_Deg(cart_coord);
//                 final_points.coordinates.push_back(cart_coord);

//                 for (size_t j = 0; j < num_scalars; ++j) {
//                     const auto& var_name = m_config.scalar_var_names[j];
//                     final_points.scalar_variables[var_name].push_back(recv_buffers[irank][i * point_data_size + 2 + j]);
//                 }
//             }
//         }
//     }

//     std::cout << "Rank " << m_pcomm->rank() << " redistribution complete: "
//               << final_points.size() << " final points" << std::endl;

//     return MB_SUCCESS;
// }

// Missing method implementations

moab::ErrorCode moab::ParallelPointCloudReader::read_points(PointData& points) {
    if (m_pcomm->rank() == 0) {
        std::cout << "Starting point cloud reading..." << std::endl;
    }

    // Initialize NetCDF file
    MB_CHK_ERR(initialize_netcdf());

    return read_and_redistribute_distributed(points);

    // if (m_is_usgs_format) {
    //     // Use specialized USGS reader with spatial filtering
    //     return read_usgs_format(points);
    // } else
    // if (m_config.use_root_based_distribution) {
    //     return read_and_distribute_root_based(points);
    // } else {
    //     return read_and_redistribute_distributed(points);
    // }
}

void moab::ParallelPointCloudReader::cleanup_netcdf() {
    if (m_ncfile) {
        delete m_ncfile;
        m_ncfile = nullptr;
    }
}



} // namespace moab

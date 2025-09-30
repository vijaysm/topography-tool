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

        // if (m_pcomm->rank() == 0) {
        //     std::cout << "Total points in dataset: " << m_total_points << std::endl;
        //     for(const auto& var : m_config.scalar_var_names)
        //     {
        //         std::cout << "Found variable: " << var << std::endl;
        //     }
        // }
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
    assert(lonlat_bbox.min_coords[0] >= 0.0);
    assert(lonlat_bbox.min_coords[1] >= -90.0);
    if(lonlat_bbox.max_coords[0] > 360.0)
        std::cout << "Found longitude " << lonlat_bbox.max_coords[0] << " which is greater than 360" << std::endl;
    assert(lonlat_bbox.max_coords[0] <= 360.0);
    assert(lonlat_bbox.max_coords[1] <= 90.0);

    {
        std::cout << m_pcomm->rank() << ": Computed lat/lon bounding box: lon[" << lonlat_bbox.min_coords[0]
                  << ", " << lonlat_bbox.max_coords[0] << "] lat[" << lonlat_bbox.min_coords[1]
                  << ", " << lonlat_bbox.max_coords[1] << "]" << std::endl;
    }

    return MB_SUCCESS;
}

ErrorCode ParallelPointCloudReader::gather_all_bounding_boxes(std::vector<BoundingBox>& all_bboxes) {
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
                // USGS format: use spatial filtering to read only needed subset?

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
                // The caller will be responsible for filtering this data
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


moab::ErrorCode moab::ParallelPointCloudReader::read_and_redistribute_distributed(PointData& local_points) {
    if (m_pcomm->rank() == 0) {
        std::cout << "=== Starting distributed reading and redistribution ===" << std::endl;
    }

    nlats_start = 0;
    nlats_count = nlats;
    nlons_start = 0;
    nlons_count = nlons;

    // now let us scale by number of latitudes
    const size_t points_per_rank = nlons * nlats;
    const size_t my_count = nlons_count * nlats_count;
    const size_t my_start_idx = 0;

    if (m_pcomm->rank() == 0) {
        std::cout << "Each rank reading ~" << points_per_rank << " points" << std::endl;
        std::cout << "Total points: " << m_total_points << ", ranks: " << m_pcomm->size() << std::endl;
    }

    // Step 2: Read my chunk of data
    MB_CHK_ERR(read_local_chunk_distributed(my_start_idx, my_count, local_points));

    // Note: Bounding box is already computed from H5M mesh in configure()
    // We use the mesh bounding box (in lat/lon) to filter relevant NetCDF points

    // Step 3: Bounding boxes are now populated in m_all_bboxes

    // Step 4: Redistribute points based on bounding box ownership
    // MB_CHK_ERR(redistribute_points_by_ownership(initial_points, local_points));

    // let us delete the file handle so that we can do
    // everything else in memory
    delete m_ncfile;
    m_ncfile = nullptr;

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

moab::ErrorCode moab::ParallelPointCloudReader::read_points(PointData& points) {
    if (m_pcomm->rank() == 0) {
        std::cout << "Starting point cloud reading..." << std::endl;
    }

    // Initialize NetCDF file
    MB_CHK_ERR(initialize_netcdf());

    return read_and_redistribute_distributed(points);
}

void moab::ParallelPointCloudReader::cleanup_netcdf() {
    if (m_ncfile) {
        delete m_ncfile;
        m_ncfile = nullptr;
    }
}



} // namespace moab

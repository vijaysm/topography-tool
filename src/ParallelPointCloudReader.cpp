
#include <mpi.h>
#include "ParallelPointCloudReader.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <unistd.h>
#include <limits>
#include <cassert>

namespace moab {

ParallelPointCloudReader::ParallelPointCloudReader(Interface* interface, EntityHandle mesh_set)
    : m_interface(interface), m_mesh_set(mesh_set), m_ncfile(nullptr), m_total_points(0), m_is_usgs_format(false) {
}

ParallelPointCloudReader::~ParallelPointCloudReader() {
    cleanup_netcdf();
}

ErrorCode ParallelPointCloudReader::configure(const ReadConfig& config) {
    // initialize the configuration parameters
    m_config = config;

    // Compute bounding boxes for mesh decomposition
    // Since we have lat/lon coordinates, compute bounding box directly in lat/lon space from mesh vertices
    MB_CHK_SET_ERR(compute_lonlat_bounding_box_from_mesh(this->m_local_bbox), "Failed to compute local lon/lat bounding box");

    return MB_SUCCESS;
}

ErrorCode ParallelPointCloudReader::initialize_netcdf() {
    try {
        m_ncfile = nullptr;

        try {
            m_ncfile = new PnetCDF::NcmpiFile(MPI_COMM_WORLD, m_config.netcdf_filename, PnetCDF::NcmpiFile::read);
        }
        catch (PnetCDF::exceptions::NcInvalidArg& e) {
            std::cerr << "Error: Could not open NetCDF file with PnetCDF (invalid argument): "
                      << m_config.netcdf_filename << std::endl;
            return MB_FAILURE;
        }

        if (!m_ncfile) {
            std::cerr << "Error: Could not open NetCDF file: " << m_config.netcdf_filename << std::endl;
            return MB_FAILURE;
        }

        // Choose reading strategy based on file format and configuration
        // Detect file format
        MB_CHK_ERR(detect_netcdf_format());

        return moab::MB_SUCCESS;

    } catch (const std::exception& e) {
        std::cerr << "An exception occurred during NetCDF initialization: " << e.what() << std::endl;
        return MB_FAILURE;
    }
}

ErrorCode ParallelPointCloudReader::detect_netcdf_format() {
    try {
        std::cout << "Detecting NetCDF format..." << std::endl;

        // Store all variables in a map
        const std::multimap<std::string, PnetCDF::NcmpiDim> dims_map = m_ncfile->getDims();
        const std::multimap<std::string, PnetCDF::NcmpiVar> vars_map = m_ncfile->getVars();

        std::cout << "NetCDF file has " << dims_map.size() << " dimensions and " << vars_map.size() << " variables" << std::endl;

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

            std::cout << "Detected USGS format NetCDF file" << std::endl;
            std::cout << "  Coordinate dimensions: " << lat_var_name << " (" << nlats << "), "
                      << lon_var_name << " (" << nlons << ")" << std::endl;
            std::cout << "  Topography variable: " << topo_var_name << std::endl;
            if (has_fract) {
                std::cout << "  Land fraction variable: " << fract_var_name << std::endl;
            }
        } else {
            std::cout << "Not USGS format, detecting coordinate variables..." << std::endl;

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

        std::cout << "Detected coordinate variables: " << lon_var_name << "(" << nlons << "), " << lat_var_name << "(" << nlats << ")" << std::endl;

        return MB_SUCCESS;
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

    // Ensure longitude stays within [0, 180] bounds and latitude within [-90, 90]
    assert(lonlat_bbox.min_coords[0] >= 0.0);
    assert(lonlat_bbox.min_coords[1] >= -90.0);
    if(lonlat_bbox.max_coords[0] > 360.0)
        std::cout << "Found longitude " << lonlat_bbox.max_coords[0] << " which is greater than 360" << std::endl;
    assert(lonlat_bbox.max_coords[0] <= 360.0);
    assert(lonlat_bbox.max_coords[1] <= 90.0);

    {
        std::cout << "Computed lat/lon bounding box: lon[" << lonlat_bbox.min_coords[0]
                  << ", " << lonlat_bbox.max_coords[0] << "] lat[" << lonlat_bbox.min_coords[1]
                  << ", " << lonlat_bbox.max_coords[1] << "]" << std::endl;
    }

    return MB_SUCCESS;
}


ErrorCode ParallelPointCloudReader::read_coordinates_chunk(size_t start_idx, size_t count,
                                   std::vector<ParallelPointCloudReader::PointType>& coords) {

    if (m_config.coord_var_names.size() < 2) return MB_FAILURE;

    try {
        PnetCDF::NcmpiVar x_var = m_vars.at(m_config.coord_var_names[0]);
        PnetCDF::NcmpiVar y_var = m_vars.at(m_config.coord_var_names[1]);

        // Check variable dimensions to determine proper reading strategy
        int x_ndims = x_var.getDimCount();
        int y_ndims = y_var.getDimCount();

        std::cout << "Coordinate variables: " << m_config.coord_var_names[0]
                  << " (dims=" << x_ndims << "), " << m_config.coord_var_names[1]
                  << " (dims=" << y_ndims << ")" << std::endl;

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

                std::cout << "USGS coordinate grid: " << nlats << " lat x " << nlons << " lon" << std::endl;

                // Read entire coordinate arrays (they are 1D and relatively small)
                std::vector<CoordinateType> lats(nlats), lons(nlons);

                std::vector<MPI_Offset> lat_start = {static_cast<MPI_Offset>(nlats_start)};
                std::vector<MPI_Offset> lat_count = {static_cast<MPI_Offset>(nlats_count)};
                std::vector<MPI_Offset> lon_start = {static_cast<MPI_Offset>(nlons_start)};
                std::vector<MPI_Offset> lon_count = {static_cast<MPI_Offset>(nlons_count)};

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

            std::cout << "2D coordinate arrays: " << dim0_size << " x " << dim1_size << std::endl;

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
            std::cerr << "Unsupported coordinate variable dimensions: "
                      << x_ndims << ", " << y_ndims << std::endl;
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
        std::cerr << "Error reading coordinates: " << e.what() << std::endl;
        return MB_FAILURE;
    }
}

template<typename T>
ErrorCode ParallelPointCloudReader::read_scalar_variable_chunk(const std::string& var_name, size_t start_idx,
                                       size_t count, std::vector<T>& data) {

    auto var_it = m_vars.find(var_name);
    if (var_it == m_vars.end()) return MB_FAILURE;

    try {
        PnetCDF::NcmpiVar var = var_it->second;
        int ndims = var.getDimCount();

        std::cout << "Reading scalar variable '" << var_name << "' with " << ndims << " dimensions" << std::endl;

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
                    std::cout << "Reading variable '" << var_name << "' in chunks to avoid overflow." << std::endl;

                    size_t lat_chunk_size = MAX_ELEMENTS_PER_CHUNK / nlons_count;
                    if (lat_chunk_size == 0) lat_chunk_size = 1; // Ensure progress

                    for (size_t lat_offset = 0; lat_offset < nlats_count; lat_offset += lat_chunk_size) {
                        size_t current_lat_count = std::min(lat_chunk_size, nlats_count - lat_offset);
                        size_t chunk_elements = current_lat_count * nlons_count;
                        std::cout << "\tReading latitude chunk from " << nlats_start + lat_offset << " to " << nlats_start + lat_offset + current_lat_count << "." << std::endl;

                        std::vector<T> chunk_buffer(chunk_elements);
                        start = {static_cast<MPI_Offset>(nlats_start + lat_offset), static_cast<MPI_Offset>(nlons_start)};
                        read_count = {static_cast<MPI_Offset>(current_lat_count), static_cast<MPI_Offset>(nlons_count)};

                        var.getVar_all(start, read_count, chunk_buffer.data());
                        temp_buffer.insert(temp_buffer.end(), chunk_buffer.begin(), chunk_buffer.end());
                    }
                } else {
                    // Read all at once
                    temp_buffer.resize(total_elements_to_read);
                    start = {static_cast<MPI_Offset>(nlats_start), static_cast<MPI_Offset>(nlons_start)};
                    read_count = {static_cast<MPI_Offset>(nlats_count), static_cast<MPI_Offset>(nlons_count)};
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
            std::cout << "WARNING:Skipping 3D variable '" << var_name << "' (not supported for scalar data)" << std::endl;
            return MB_SUCCESS;
        } else {
            std::cerr << "Unsupported variable dimensions: " << ndims << " for variable " << var_name << std::endl;
            return MB_FAILURE;
        }

        return MB_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "Error reading scalar variable '" << var_name << "': " << e.what() << std::endl;
        return MB_FAILURE;
    }
}


moab::ErrorCode moab::ParallelPointCloudReader::read_all_data(PointData& local_points) {
    std::cout << "=== Starting data reading ===" << std::endl;

    nlats_start = 0;
    nlats_count = nlats;
    nlons_start = 0;
    nlons_count = nlons;

    // now let us scale by number of latitudes
    const size_t points_per_rank = nlons * nlats;
    const size_t my_count = nlons_count * nlats_count;
    const size_t my_start_idx = 0;

    std::cout << "Reading ~" << points_per_rank << " points" << std::endl;
    std::cout << "Total points: " << m_total_points << std::endl;

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

            std::vector<MPI_Offset> lat_start = {static_cast<MPI_Offset>(nlats_start)};
            std::vector<MPI_Offset> lat_count = {static_cast<MPI_Offset>(nlats_count)};
            std::vector<MPI_Offset> lon_start = {static_cast<MPI_Offset>(nlons_start)};
            std::vector<MPI_Offset> lon_count = {static_cast<MPI_Offset>(nlons_count)};

            lat_var.getVar_all(lat_start, lat_count, chunk_data.latitudes.data());
            lon_var.getVar_all(lon_start, lon_count, chunk_data.longitudes.data());

            // Mark as structured grid - coordinates will be computed on-the-fly
            chunk_data.is_structured_grid = true;

            std::cout << "USGS format: Stored " << chunk_data.latitudes.size() << " latitudes and "
                      << chunk_data.longitudes.size() << " longitudes (total grid points: "
                      << chunk_data.size() << ", computed on-the-fly)" << std::endl;
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

        std::cout << "Read " << chunk_data.size()
                  << " points with " << chunk_data.d_scalar_variables.size() << " scalar variables" << std::endl;

        return MB_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "Error in read_local_chunk_distributed: " << e.what() << std::endl;
        return MB_FAILURE;
    }
}

moab::ErrorCode moab::ParallelPointCloudReader::read_points(PointData& points) {
    std::cout << "Starting point cloud reading..." << std::endl;

    // Initialize NetCDF file
    MB_CHK_ERR(initialize_netcdf());

    // Call the data reading
    MB_CHK_ERR(read_all_data(points));
    
    return MB_SUCCESS;
}

void moab::ParallelPointCloudReader::cleanup_netcdf() {
    if (m_ncfile) {
        delete m_ncfile;
        m_ncfile = nullptr;
    }
}



} // namespace moab

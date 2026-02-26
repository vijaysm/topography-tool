/**
 * @file ParallelPointCloudReader.cpp
 * @brief Implementation of parallel NetCDF point cloud reader for MOAB
 *
 * This file implements high-performance parallel reading of point cloud
 * datasets from NetCDF files, supporting both structured grids (USGS format)
 * and unstructured point clouds. The implementation uses Parallel-NetCDF for
 * scalable I/O and includes sophisticated spatial decomposition for efficient
 * data distribution.
 *
 * Key Implementation Features:
 * - Parallel NetCDF I/O with PnetCDF for scalable reading
 * - Automatic format detection and appropriate reading strategy selection
 * - Memory-efficient chunked reading for large datasets
 * - Thread-safe operations supporting OpenMP parallelism
 * - Robust error handling and resource management
 * - Spatial decomposition for MPI-based data distribution
 *
 * Author: Vijay Mahadevan
 * Date: 2025-2026
 */

#include "ParallelPointCloudReader.hpp"
#include "easylogging.hpp"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <mpi.h>
#include <unistd.h>

namespace moab {

//===========================================================================
// Constructor and Destructor
//===========================================================================

/**
 * @brief Constructor
 *
 * Initializes the ParallelPointCloudReader with MOAB interface and mesh set.
 * Sets up default values for member variables.
 *
 * @param interface MOAB interface instance
 * @param mesh_set Mesh set containing target elements
 */
ParallelPointCloudReader::ParallelPointCloudReader(Interface *interface,
                                                   EntityHandle mesh_set)
    : m_interface(interface), m_mesh_set(mesh_set), m_ncfile(nullptr),
      m_total_points(0), m_is_usgs_format(false) {}

/**
 * @brief Destructor
 *
 * Cleans up NetCDF resources and releases file handles.
 */
ParallelPointCloudReader::~ParallelPointCloudReader() { cleanup_netcdf(); }

//===========================================================================
// Configuration and Initialization
//===========================================================================

/**
 * @brief Configure the reader with user parameters
 *
 * Sets up the configuration parameters for reading operations.
 * This method should be called before any reading operations.
 *
 * @param config Configuration parameters
 * @return MB_SUCCESS on success
 */
ErrorCode ParallelPointCloudReader::configure(const ReadConfig &config) {
  // Store configuration parameters
  m_config = config;

  return MB_SUCCESS;
}

ErrorCode ParallelPointCloudReader::initialize_netcdf() {
  try {
    m_ncfile = nullptr;

    try {
      m_ncfile = new PnetCDF::NcmpiFile(
          MPI_COMM_WORLD, m_config.netcdf_filename, PnetCDF::NcmpiFile::read);
    } catch (PnetCDF::exceptions::NcInvalidArg &e) {
      LOG(ERROR) << "Error: Could not open NetCDF file with PnetCDF "
                    "(invalid argument): "
                 << m_config.netcdf_filename;
      return MB_FAILURE;
    }

    if (!m_ncfile) {
      LOG(ERROR) << "Error: Could not open NetCDF file: "
                 << m_config.netcdf_filename;
      return MB_FAILURE;
    }

    // Detect file format and set up reading strategy
    MB_CHK_ERR(detect_netcdf_format());

    return moab::MB_SUCCESS;
  } catch (const std::exception &e) {
    LOG(ERROR) << "An exception occurred during NetCDF initialization: "
               << e.what();
    return MB_FAILURE;
  }
}

ErrorCode ParallelPointCloudReader::detect_netcdf_format() {
  try {
    // Load all variables and dimensions from the NetCDF file
    const std::multimap<std::string, PnetCDF::NcmpiDim> dims_map =
        m_ncfile->getDims();
    const std::multimap<std::string, PnetCDF::NcmpiVar> vars_map =
        m_ncfile->getVars();

    // Store all variables for efficient access
    for (const auto &var_pair : vars_map) {
      m_vars[var_pair.first] = var_pair.second;
    }

    LOG(INFO) << "Detecting NetCDF format...";
    LOG(INFO) << "NetCDF file has " << dims_map.size() << " dimensions and "
              << vars_map.size() << " variables";

    // Check for USGS format: lat/lon dimensions and htopo variable
    bool has_lat = false, has_lon = false, has_htopo = false, has_fract = false;
    for (const auto &dim_pair : dims_map) {
      if (!dim_pair.first.compare("lat") ||
          !dim_pair.first.compare("latitude")) {
        has_lat = true;
        lat_var_name = !dim_pair.first.compare("lat") ? "lat" : "latitude";
      } else if (!dim_pair.first.compare("lon") ||
                 !dim_pair.first.compare("longitude")) {
        has_lon = true;
        lon_var_name = !dim_pair.first.compare("lon") ? "lon" : "longitude";
      }
    }
    if (m_vars.count("htopo") || m_vars.count("Elevation") ||
        m_vars.count("terr")) {
      has_htopo = true;
      topo_var_name = m_vars.count("htopo")       ? "htopo"
                      : m_vars.count("Elevation") ? "Elevation"
                                                  : "terr";
    }
    if (m_vars.count("landfract") || m_vars.count("LandWater") ||
        m_vars.count("LANDFRAC")) {
      has_fract = true;
      fract_var_name = m_vars.count("landfract")   ? "landfract"
                       : m_vars.count("LandWater") ? "LandWater"
                                                   : "LANDFRAC";
    }

    // Check for SCRIP format before USGS format
    bool has_grid_size = false;
    bool has_grid_center_lat = false;
    bool has_grid_center_lon = false;
    size_t grid_size_val = 0;

    for (const auto &dim_pair : dims_map) {
      if (dim_pair.first == "grid_size") {
        has_grid_size = true;
        grid_size_val = dim_pair.second.getSize();
      }
    }

    if (m_vars.count("grid_center_lat")) {
      has_grid_center_lat = true;
    }
    if (m_vars.count("grid_center_lon")) {
      has_grid_center_lon = true;
    }

    // Check for unstructured grid with lat/lon variables on grid_size
    bool has_lat_var_on_grid_size = false;
    bool has_lon_var_on_grid_size = false;

    if (has_grid_size) {
      // Check if lat/lon are variables (not dimensions) with grid_size
      // dimension
      if (m_vars.count("lat")) {
        PnetCDF::NcmpiVar lat_var = m_vars.at("lat");
        if (lat_var.getDimCount() == 1 &&
            static_cast<size_t>(lat_var.getDims()[0].getSize()) ==
                grid_size_val) {
          has_lat_var_on_grid_size = true;
        }
      }
      if (m_vars.count("lon")) {
        PnetCDF::NcmpiVar lon_var = m_vars.at("lon");
        if (lon_var.getDimCount() == 1 &&
            static_cast<size_t>(lon_var.getDims()[0].getSize()) ==
                grid_size_val) {
          has_lon_var_on_grid_size = true;
        }
      }
    }

    // SCRIP format detection (grid_center_lat/lon)
    if (has_grid_size && has_grid_center_lat && has_grid_center_lon) {
      m_is_usgs_format = false; // SCRIP is unstructured

      // Set coordinate variable names
      lon_var_name = "grid_center_lon";
      lat_var_name = "grid_center_lat";
      m_config.coord_var_names = {lon_var_name, lat_var_name};

      // Grid size determines total points
      m_total_points = grid_size_val;
      nlats = 1; // No tensor product structure
      nlons = grid_size_val;

      // Automatically detect scalar variables (only if not already
      // specified by user)
      if (m_config.scalar_var_names.empty()) {
        std::vector<std::string> scalar_vars;
        for (const auto &var_pair : m_vars) {
          const std::string &var_name = var_pair.first;
          // Exclude coordinate variables and corner variables
          if (var_name != lon_var_name && var_name != lat_var_name &&
              var_name != "grid_corner_lat" && var_name != "grid_corner_lon" &&
              var_name != "grid_dims") {

            // Check if it's a 1D variable with grid_size dimension
            PnetCDF::NcmpiVar var = var_pair.second;
            if (var.getDimCount() == 1 &&
                static_cast<size_t>(var.getDims()[0].getSize()) ==
                    grid_size_val) {
              scalar_vars.push_back(var_name);
            }
          }
        }
        m_config.scalar_var_names = scalar_vars;
      }

      LOG(INFO) << "Detected SCRIP format NetCDF file";
      LOG(INFO) << "  Grid size: " << grid_size_val << " (unstructured)";
      LOG(INFO) << "  Coordinate variables: " << lon_var_name << ", "
                << lat_var_name;
      LOG(INFO) << "  Scalar variables: ";
      for (const auto &svar : m_config.scalar_var_names) {
        LOG(INFO) << svar << " ";
      }
    }
    // Unstructured grid with lat/lon variables on grid_size (e.g.,
    // cubed-sphere)
    else if (has_grid_size && has_lat_var_on_grid_size &&
             has_lon_var_on_grid_size) {
      m_is_usgs_format = false; // Unstructured

      // Set coordinate variable names
      lon_var_name = "lon";
      lat_var_name = "lat";
      m_config.coord_var_names = {lon_var_name, lat_var_name};

      // Grid size determines total points
      m_total_points = grid_size_val;
      nlats = 1; // No tensor product structure
      nlons = grid_size_val;

      // Automatically detect scalar variables (only if not already
      // specified by user)
      if (m_config.scalar_var_names.empty()) {
        std::vector<std::string> scalar_vars;
        for (const auto &var_pair : m_vars) {
          const std::string &var_name = var_pair.first;
          // Exclude coordinate variables and grid_dims
          if (var_name != lon_var_name && var_name != lat_var_name &&
              var_name != "grid_dims") {

            // Check if it's a 1D variable with grid_size dimension
            PnetCDF::NcmpiVar var = var_pair.second;
            if (var.getDimCount() == 1 &&
                static_cast<size_t>(var.getDims()[0].getSize()) ==
                    grid_size_val) {
              scalar_vars.push_back(var_name);
            }
          }
        }
        m_config.scalar_var_names = scalar_vars;
      }

      LOG(INFO) << "Detected unstructured grid format "
                   "(cubed-sphere/USGS-topo) NetCDF file";
      LOG(INFO) << "  Grid size: " << grid_size_val << " (unstructured)";
      LOG(INFO) << "  Coordinate variables: " << lon_var_name << ", "
                << lat_var_name;
      LOG(INFO) << "  Scalar variables: ";
      for (const auto &svar : m_config.scalar_var_names) {
        LOG(INFO) << svar << " ";
      }
    } else if (has_lat && has_lon && has_htopo) {
      m_is_usgs_format = true;

      // Get dimension sizes
      PnetCDF::NcmpiDim lat_dim = m_ncfile->getDim(lat_var_name);
      PnetCDF::NcmpiDim lon_dim = m_ncfile->getDim(lon_var_name);
      nlats = lat_dim.getSize();
      nlons = lon_dim.getSize();
      m_total_points = nlats * nlons;

      // Set up configuration
      m_config.coord_var_names = {lon_var_name, lat_var_name};

      // Only set scalar vars if not already specified by user
      if (m_config.scalar_var_names.empty()) {
        if (has_fract) {
          m_config.scalar_var_names = {topo_var_name, fract_var_name};
        } else {
          m_config.scalar_var_names = {topo_var_name};
        }
      }

      LOG(INFO) << "Detected USGS format NetCDF file";
      LOG(INFO) << "  Coordinate dimensions: " << lat_var_name << " (" << nlats
                << "), " << lon_var_name << " (" << nlons << ")";
      LOG(INFO) << "  Topography variable: " << topo_var_name;
      if (has_fract) {
        LOG(INFO) << "  Land fraction variable: " << fract_var_name;
      }
    } else {
      LOG(INFO) << "Not USGS format, detecting coordinate variables...";

      std::vector<std::string> lon_names = {"xc", "lon", "longitude", "x"};
      std::vector<std::string> lat_names = {"yc", "lat", "latitude", "y"};

      for (const auto &name : lon_names) {
        if (m_vars.count(name)) {
          lon_var_name = name;
          break;
        }
      }
      for (const auto &name : lat_names) {
        if (m_vars.count(name)) {
          lat_var_name = name;
          break;
        }
      }

      if (!lon_var_name.empty() && !lat_var_name.empty()) {
        m_config.coord_var_names = {lon_var_name, lat_var_name};

        PnetCDF::NcmpiVar coord_var = m_vars.at(lon_var_name);
        m_total_points = 1;
        for (const auto &dim : coord_var.getDims()) {
          m_total_points *= dim.getSize();
        }
        nlats = coord_var.getDims()[0].getSize();
        nlons = coord_var.getDims()[1].getSize();

        // Automatically detect scalar variables (only if not already
        // specified by user)
        if (m_config.scalar_var_names.empty()) {
          std::vector<std::string> scalar_vars;
          for (const auto &var_pair : m_vars) {
            const std::string &var_name = var_pair.first;
            if (var_name != lon_var_name && var_name != lat_var_name &&
                var_name.compare("xv") && var_name.compare("yv")) {
              scalar_vars.push_back(var_name);
            }
          }
          m_config.scalar_var_names = scalar_vars;
        }
      } else {
        LOG(ERROR) << "Error: Could not find coordinate variables in "
                      "NetCDF file";
        return MB_FAILURE;
      }
    }

    LOG(INFO) << "Detected coordinate variables: " << lon_var_name << "("
              << nlons << "), " << lat_var_name << "(" << nlats << ")";

    // Validate that all requested scalar variables exist
    if (!m_config.scalar_var_names.empty()) {
      std::vector<std::string> missing_vars;
      for (const auto &var_name : m_config.scalar_var_names) {
        if (m_vars.find(var_name) == m_vars.end()) {
          missing_vars.push_back(var_name);
        }
      }

      if (!missing_vars.empty()) {
        LOG(ERROR) << "\nERROR: Requested variables not found in NetCDF file:";
        for (const auto &var : missing_vars) {
          LOG(ERROR) << "  - '" << var << "'";
        }
        LOG(ERROR) << "\nAvailable data variables in file: ";
        for (const auto &v : m_vars) {
          // Skip coordinate variables
          if (v.first != lon_var_name && v.first != lat_var_name) {
            LOG(ERROR) << v.first << " ";
          }
        }
        LOG(ERROR) << "\n";
        return MB_FAILURE;
      }
    }

    return MB_SUCCESS;
  } catch (const std::exception &e) {
    LOG(ERROR) << "Error in detect_netcdf_format(): " << e.what();
    return MB_FAILURE;
  }
}

//===========================================================================
// Coordinate Reading Methods
//===========================================================================

/**
 * @brief Read coordinate data chunk from NetCDF file
 *
 * Reads a chunk of coordinate data (longitude and latitude) from the NetCDF
 * file using parallel I/O. Handles both structured grids (USGS format) and
 * unstructured point clouds with appropriate dimension handling.
 *
 * Algorithm:
 * 1. Get coordinate variable handles
 * 2. Determine variable dimensions and reading strategy
 * 3. Set up parallel I/O parameters (start, count)
 * 4. Read data using PnetCDF parallel operations
 * 5. Convert to PointType format
 *
 * Reading Strategies:
 * - Structured Grid: Read from 1D arrays, compute 2D coordinates
 * - Unstructured: Read directly from 1D coordinate arrays
 *
 * @param start_idx Starting index for chunk
 * @param count Number of points to read
 * @param coords Output coordinate array [lon, lat] pairs
 * @return MB_SUCCESS on success, MB_FAILURE on error
 */
ErrorCode ParallelPointCloudReader::read_coordinates_chunk(
    size_t start_idx, size_t count, std::vector<PointType> &coords) {

  // Validate coordinate variable configuration
  if (m_config.coord_var_names.size() < 2)
    return MB_FAILURE;

  try {
    // Get coordinate variable handles
    PnetCDF::NcmpiVar x_var = m_vars.at(m_config.coord_var_names[0]);
    PnetCDF::NcmpiVar y_var = m_vars.at(m_config.coord_var_names[1]);

    // Analyze variable dimensions to determine reading strategy
    int x_ndims = x_var.getDimCount();
    int y_ndims = y_var.getDimCount();

    LOG(INFO) << "Coordinate variables: " << m_config.coord_var_names[0]
              << " (dims=" << x_ndims << "), " << m_config.coord_var_names[1]
              << " (dims=" << y_ndims << ")";

    // Set up parallel I/O parameters
    std::vector<MPI_Offset> start, read_count;

    if (x_ndims == 1 && y_ndims == 1) {
      // 1D coordinate arrays - could be USGS format or standard point
      // arrays Check if this is USGS format by examining coordinate
      // variable names
      if (m_is_usgs_format) {
        // USGS format: lon and lat are separate 1D arrays defining a
        // grid Need to convert linear index to 2D grid indices

        // Get dimension sizes and validate
        PnetCDF::NcmpiDim lat_dim = m_ncfile->getDim(lat_var_name);
        PnetCDF::NcmpiDim lon_dim = m_ncfile->getDim(lon_var_name);
        assert(nlats - lat_dim.getSize() == 0);
        assert(nlons - lon_dim.getSize() == 0);

        LOG(INFO) << "USGS coordinate grid: " << nlats << " lat x " << nlons
                  << " lon";

        // Read entire coordinate arrays (they are 1D and relatively
        // small)
        std::vector<CoordinateType> lats(nlats), lons(nlons);

        // Set up parallel I/O parameters for coordinate arrays
        std::vector<MPI_Offset> lat_start = {
            static_cast<MPI_Offset>(nlats_start)};
        std::vector<MPI_Offset> lat_count = {
            static_cast<MPI_Offset>(nlats_count)};
        std::vector<MPI_Offset> lon_start = {
            static_cast<MPI_Offset>(nlons_start)};
        std::vector<MPI_Offset> lon_count = {
            static_cast<MPI_Offset>(nlons_count)};

        // Read coordinate data using parallel NetCDF
        y_var.getVar_all(lat_start, lat_count, lats.data()); // latitude
        x_var.getVar_all(lon_start, lon_count,
                         lons.data()); // longitude

        // Generate coordinate pairs for the requested chunk
        // Convert linear index to 2D grid indices: Index(ilat, ilon) =
        // ilat * nlon + ilon
        coords.resize(count);
        for (size_t i = 0; i < count; ++i) {
          size_t global_idx = start_idx + i;
          size_t lat_idx = global_idx / nlons; // Row index (latitude)
          size_t lon_idx = global_idx % nlons; // Column index (longitude)

          if (lat_idx < nlats && lon_idx < nlons) {
            coords[i] = {lons[lon_idx], lats[lat_idx]};
          } else {
            LOG(WARNING) << "Point " << global_idx << " is out of bounds";
            coords[i] = {0.0, 0.0}; // Default for out of bounds
          }
        }

        return MB_SUCCESS;
      } else {
        // Standard format: 1D coordinate arrays for unstructured point
        // cloud Each coordinate array has the same length as the number
        // of points

        LOG(INFO) << "Reading unstructured point cloud coordinates";

        // Set up parallel I/O parameters for 1D arrays
        start = {static_cast<MPI_Offset>(start_idx)};
        read_count = {static_cast<MPI_Offset>(count)};

        std::vector<CoordinateType> x_coords(count), y_coords(count);
        x_var.getVar_all(start, read_count, x_coords.data());
        y_var.getVar_all(start, read_count, y_coords.data());

        coords.resize(count);
        for (size_t i = 0; i < count; ++i) {
          coords[i] = {x_coords[i], y_coords[i]};
        }
        return MB_SUCCESS;
      }
    } else if (x_ndims == 2 && y_ndims == 2) {
      // 2D coordinate arrays - need to handle ni/nj indexing
      // For domain files, coordinates are typically stored as (nj, ni) or
      // (ni, nj) We need to convert linear index to 2D indices

      // Get dimension sizes
      std::vector<PnetCDF::NcmpiDim> x_dims = x_var.getDims();
      std::vector<PnetCDF::NcmpiDim> y_dims = y_var.getDims();

      size_t dim0_size = x_dims[0].getSize();
      size_t dim1_size = x_dims[1].getSize();

      LOG(INFO) << "2D coordinate arrays: " << dim0_size << " x " << dim1_size;

      // Read entire coordinate arrays for now (simpler approach)
      start = {0, 0};
      read_count = {static_cast<MPI_Offset>(dim0_size),
                    static_cast<MPI_Offset>(dim1_size)};

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
      LOG(ERROR) << "Unsupported coordinate variable dimensions: " << x_ndims
                 << ", " << y_ndims;
      return MB_FAILURE;
    }
  } catch (const std::exception &e) {
    LOG(ERROR) << "Error reading coordinates: " << e.what();
    return MB_FAILURE;
  }
}
//===========================================================================
// Scalar Variable Reading Methods
//===========================================================================

/**
 * @brief Template method to read scalar variable chunk
 *
 * Reads a chunk of scalar data from a NetCDF variable using parallel I/O.
 * This template method supports both double and integer data types.
 *
 * @tparam T Data type (double or int)
 * @param var_name Variable name in NetCDF file
 * @param start_idx Starting index for chunk
 * @param count Number of values to read
 * @param data Output data array
 * @return MB_SUCCESS on success, MB_FAILURE on error
 */
template <typename T>
ErrorCode ParallelPointCloudReader::read_scalar_variable_chunk(
    const std::string &var_name, size_t start_idx, size_t count,
    std::vector<T> &data) {

  auto var_it = m_vars.find(var_name);
  if (var_it == m_vars.end()) {
    LOG(ERROR) << "ERROR: Variable '" << var_name
               << "' not found in NetCDF file!";
    LOG(ERROR) << "Available variables: ";
    for (const auto &v : m_vars) {
      LOG(ERROR) << v.first << " ";
    }
    return MB_FAILURE;
  }

  try {
    PnetCDF::NcmpiVar var = var_it->second;
    int ndims = var.getDimCount();

    LOG(INFO) << "Reading scalar variable '" << var_name << "' with " << ndims
              << " dimensions";

    std::vector<MPI_Offset> start, read_count;

    if (ndims == 1) {
      // 1D variable - standard case
      start = {static_cast<MPI_Offset>(start_idx)};
      read_count = {static_cast<MPI_Offset>(count)};

      data.resize(count);

      // Read data using parallel NetCDF
      var.getVar_all(start, read_count, data.data());
    } else if (ndims == 2) {
      // 2D variable - handle USGS format with spatial chunking
      std::vector<PnetCDF::NcmpiDim> dims = var.getDims();
      size_t dim0_size = dims[0].getSize();
      size_t dim1_size = dims[1].getSize();
      size_t total_size = dim0_size * dim1_size;

      // Check if this is USGS format (lat x lon grid)
      if (m_is_usgs_format ||
          (dims[0].getName() == "lat" && dims[1].getName() == "lon")) {
        // USGS format: use spatial filtering to read only needed
        // subset?

        size_t total_elements_to_read = nlats_count * nlons_count;
        const size_t MAX_ELEMENTS_PER_CHUNK =
            500 * 1000 * 1000; // 100M elements, well below INT_MAX

        std::vector<T> temp_buffer;
        temp_buffer.reserve(total_elements_to_read);

        if (total_elements_to_read > MAX_ELEMENTS_PER_CHUNK) {
          LOG(INFO) << "Reading variable '" << var_name
                    << "' in chunks to avoid overflow.";

          size_t lat_chunk_size = MAX_ELEMENTS_PER_CHUNK / nlons_count;
          if (lat_chunk_size == 0)
            lat_chunk_size = 1; // Ensure progress

          for (size_t lat_offset = 0; lat_offset < nlats_count;
               lat_offset += lat_chunk_size) {
            size_t current_lat_count =
                std::min(lat_chunk_size, nlats_count - lat_offset);
            size_t chunk_elements = current_lat_count * nlons_count;
            LOG(INFO) << "\tReading latitude chunk from "
                      << nlats_start + lat_offset << " to "
                      << nlats_start + lat_offset + current_lat_count << ".";

            std::vector<T> chunk_buffer(chunk_elements);
            start = {static_cast<MPI_Offset>(nlats_start + lat_offset),
                     static_cast<MPI_Offset>(nlons_start)};
            read_count = {static_cast<MPI_Offset>(current_lat_count),
                          static_cast<MPI_Offset>(nlons_count)};

            var.getVar_all(start, read_count, chunk_buffer.data());
            temp_buffer.insert(temp_buffer.end(), chunk_buffer.begin(),
                               chunk_buffer.end());
          }
        } else {
          // Read all at once
          temp_buffer.resize(total_elements_to_read);
          start = {static_cast<MPI_Offset>(nlats_start),
                   static_cast<MPI_Offset>(nlons_start)};
          read_count = {static_cast<MPI_Offset>(nlats_count),
                        static_cast<MPI_Offset>(nlons_count)};
          var.getVar_all(start, read_count, temp_buffer.data());
        }

        // The temp_buffer now contains the raw 2D data block for this
        // rank's slice. The caller will be responsible for filtering
        // this data using the valid_indices it generates during
        // coordinate processing.
        data = std::move(temp_buffer);
      } else {
        // Standard 2D variable - read entire array and extract subset
        start = {0, 0};
        read_count = {static_cast<MPI_Offset>(dim0_size),
                      static_cast<MPI_Offset>(dim1_size)};

        std::vector<T> all_data(total_size);
        var.getVar_all(start, read_count, all_data.data());

        // Extract the requested subset
        data.resize(count);
        for (size_t i = 0; i < count && (start_idx + i) < total_size; ++i) {
          data[i] = all_data[start_idx + i];
        }
      }
    } else if (ndims == 3) {
      // 3D variable - skip for now (likely vertex coordinates or
      // time-dependent data)
      LOG(INFO) << "WARNING:Skipping 3D variable '" << var_name
                << "' (not supported for scalar data)";
      return MB_SUCCESS;
    } else {
      LOG(ERROR) << "Unsupported variable dimensions: " << ndims
                 << " for variable " << var_name;
      return MB_FAILURE;
    }

    return MB_SUCCESS;
  } catch (const std::exception &e) {
    LOG(ERROR) << "Error reading scalar variable '" << var_name
               << "': " << e.what();
    return MB_FAILURE;
  }
}

moab::ErrorCode
moab::ParallelPointCloudReader::read_all_data(PointData &local_points) {
  LOG(INFO) << "";
  LOG(INFO) << "=== Starting data reading ===";

  nlats_start = 0;
  nlats_count = nlats;
  nlons_start = 0;
  nlons_count = nlons;

  // now let us scale by number of latitudes
  const size_t points_per_rank = nlons * nlats;
  const size_t my_count = nlons_count * nlats_count;
  const size_t my_start_idx = 0;

  LOG(INFO) << "Reading ~" << points_per_rank << " points";
  LOG(INFO) << "Total points: " << m_total_points;

  // Read my chunk of data
  MB_CHK_ERR(
      read_local_chunk_distributed(my_start_idx, my_count, local_points));

  // let us delete the file handle so that we can do
  // everything else in memory
  delete m_ncfile;
  m_ncfile = nullptr;

  return MB_SUCCESS;
}

moab::ErrorCode moab::ParallelPointCloudReader::read_local_chunk_distributed(
    size_t start_idx, size_t count, PointData &chunk_data) {
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

      std::vector<MPI_Offset> lat_start = {
          static_cast<MPI_Offset>(nlats_start)};
      std::vector<MPI_Offset> lat_count = {
          static_cast<MPI_Offset>(nlats_count)};
      std::vector<MPI_Offset> lon_start = {
          static_cast<MPI_Offset>(nlons_start)};
      std::vector<MPI_Offset> lon_count = {
          static_cast<MPI_Offset>(nlons_count)};

      lat_var.getVar_all(lat_start, lat_count, chunk_data.latitudes.data());
      lon_var.getVar_all(lon_start, lon_count, chunk_data.longitudes.data());

      // Mark as structured grid - coordinates will be computed on-the-fly
      chunk_data.is_structured_grid = true;

      LOG(INFO) << "USGS format: Stored " << chunk_data.latitudes.size()
                << " latitudes and " << chunk_data.longitudes.size()
                << " longitudes (total grid points: " << chunk_data.size()
                << ", computed on-the-fly)";
    } else {
      // Standard format: read coordinates directly into explicit storage
      MB_CHK_ERR(read_coordinates_chunk(start_idx, count,
                                        chunk_data.lonlat_coordinates));
      chunk_data.is_structured_grid = false;
    }

    // Read scalar variables with buffered reading for large datasets
    for (const auto &var_name : m_config.scalar_var_names) {

      if (m_is_usgs_format) {
        std::vector<int> scalar_data;
        MB_CHK_ERR(read_scalar_variable_chunk(var_name, start_idx, count,
                                              scalar_data));

        // The scalar data is read in the same order as coordinates, so
        // no filtering needed
        if (scalar_data.size() == 0)
          continue; // nothing to do.

        chunk_data.i_scalar_variables[var_name] = std::move(scalar_data);

        if (chunk_data.i_scalar_variables[var_name].size() !=
            chunk_data.size()) {
          LOG(ERROR) << "WARNING: Scalar variable '" << var_name
                     << "' has a different size ("
                     << chunk_data.i_scalar_variables[var_name].size()
                     << ") than the total points (" << chunk_data.size()
                     << "). Data may be misaligned.";
        }
      } else {
        std::vector<double> scalar_data;
        MB_CHK_ERR(read_scalar_variable_chunk(var_name, start_idx, count,
                                              scalar_data));

        // The scalar data is read in the same order as coordinates, so
        // no filtering needed
        if (scalar_data.size() == 0)
          continue; // nothing to do.

        chunk_data.d_scalar_variables[var_name] = std::move(scalar_data);

        if (chunk_data.d_scalar_variables[var_name].size() !=
            chunk_data.size()) {
          LOG(ERROR) << "WARNING: Scalar variable '" << var_name
                     << "' has a different size ("
                     << chunk_data.d_scalar_variables[var_name].size()
                     << ") than the total points (" << chunk_data.size()
                     << "). Data may be misaligned.";
        }
      }
    }

    // Compute squares for specified fields
    if (!m_config.square_field_names.empty()) {
      for (const auto &field_name : m_config.square_field_names) {
        // Check if field exists in double variables
        if (chunk_data.d_scalar_variables.count(field_name)) {
          const auto &field_data = chunk_data.d_scalar_variables[field_name];
          std::vector<double> squared_data(field_data.size());

          for (size_t i = 0; i < field_data.size(); ++i) {
            squared_data[i] = field_data[i] * field_data[i];
          }

          std::string squared_name = field_name + "_squared";
          chunk_data.d_scalar_variables[squared_name] = std::move(squared_data);
          LOG(INFO) << "Computed squared field: " << squared_name;
        }
        // Check if field exists in integer variables
        else if (chunk_data.i_scalar_variables.count(field_name)) {
          const auto &field_data = chunk_data.i_scalar_variables[field_name];
          std::vector<int> squared_data(field_data.size());

          for (size_t i = 0; i < field_data.size(); ++i) {
            squared_data[i] = field_data[i] * field_data[i];
          }

          std::string squared_name = field_name + "_squared";
          chunk_data.i_scalar_variables[squared_name] = std::move(squared_data);
          LOG(INFO) << "Computed squared field: " << squared_name;
        } else {
          LOG(ERROR) << "Warning: Field '" << field_name
                     << "' not found for squaring";
        }
      }
    }

    LOG(INFO) << "Read " << chunk_data.size() << " points with "
              << chunk_data.d_scalar_variables.size()
              << " double scalar variables";
    if (!chunk_data.i_scalar_variables.empty()) {
      LOG(INFO) << " and " << chunk_data.i_scalar_variables.size()
                << " integer scalar variables";
    }
    if (!chunk_data.areas.empty()) {
      LOG(INFO) << " and area data";
    }

    return MB_SUCCESS;
  } catch (const std::exception &e) {
    LOG(ERROR) << "Error in read_local_chunk_distributed: " << e.what();
    return MB_FAILURE;
  }
}

moab::ErrorCode moab::ParallelPointCloudReader::read_points(PointData &points) {
  LOG(INFO) << "Starting point cloud reading...";

  // Initialize NetCDF file
  MB_CHK_ERR(initialize_netcdf());

  // Call the data reading
  MB_CHK_ERR(read_all_data(points));

  if (m_config.retain_point_cloud) {
    m_cached_points = points;
    m_have_cached_points = true;
  }

  return MB_SUCCESS;
}

/**
 * @brief Build MOAB mesh from point cloud data
 *
 * Creates MOAB vertices from point cloud coordinates and assigns
 * scalar data as MOAB tags. This method converts the internal
 * point cloud representation to MOAB entities.
 *
 * Algorithm:
 * 1. Create MOAB vertices for each point
 * 2. Create MOAB tags for each scalar variable
 * 3. Assign scalar data to vertices using tags
 * 4. Create vertex set and add to mesh set
 *
 * @param points Input point cloud data
 * @param mb MOAB interface instance
 * @param mesh_set Target mesh set
 * @param entities Output vector of created entity handles
 * @param default_area Default area for each point
 * @return MB_SUCCESS on success, MB_FAILURE on error
 */
ErrorCode ParallelPointCloudReader::build_mesh_from_point_cloud(
    const PointData &points, Interface *mb, EntityHandle mesh_set,
    std::vector<EntityHandle> &entities, double default_area) const {
  if (nullptr == mb)
    MB_SET_ERR(MB_FAILURE, "Invalid MOAB interface");
  if (0 == mesh_set)
    MB_SET_ERR(MB_FAILURE, "Invalid mesh set");

  const size_t total_points = points.size();
  if (0 == total_points)
    MB_SET_ERR(MB_FAILURE, "Point cloud is empty");

  std::vector<double> area_data;
  if (!points.areas.empty() && points.areas.size() == total_points) {
    LOG(INFO) << "Using area data from point cloud";
    area_data = points.areas;
  } else {
    LOG(INFO) << "Using default user-specified area = " << default_area;
    area_data.assign(total_points, default_area);
  }

  Tag area_tag = nullptr;
  MB_CHK_SET_ERR(mb->tag_get_handle("area", 1, MB_TYPE_DOUBLE, area_tag,
                                    MB_TAG_DENSE | MB_TAG_CREAT, &default_area),
                 "Failed to create area tag");

  entities.clear();
  entities.resize(total_points);

  for (size_t idx = 0; idx < total_points; ++idx) {
    std::array<double, 3> coords{0.0, 0.0, 0.0};
    auto lon = points.longitude(idx);
    auto lat = points.latitude(idx);
    RLLtoXYZ_Deg(lon, lat, coords);
    MB_CHK_SET_ERR(mb->create_vertex(coords.data(), entities[idx]),
                   "Failed to create vertex");
  }

  MB_CHK_SET_ERR(mb->tag_set_data(area_tag, entities.data(), total_points,
                                  area_data.data()),
                 "Failed to set area tag");
  MB_CHK_SET_ERR(mb->add_entities(mesh_set, entities.data(), total_points),
                 "Failed to add vertices to mesh set");

  return MB_SUCCESS;
}

void moab::ParallelPointCloudReader::cleanup_netcdf() {
  if (m_ncfile) {
    delete m_ncfile;
    m_ncfile = nullptr;
  }
}

} // namespace moab

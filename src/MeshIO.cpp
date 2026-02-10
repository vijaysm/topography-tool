/**
 * @file MeshIO.cpp
 * @brief Implementation of NetCDF I/O operations for point cloud meshes
 *
 * This file implements utilities for loading point cloud data from NetCDF files
 * and writing scalar fields from MOAB entities to NetCDF files. It supports
 * both structured and unstructured grid formats with parallel I/O capabilities
 * using PnetCDF for large datasets.
 *
 * Key Features:
 * - Parallel I/O using PnetCDF for scalable data access
 * - Automatic type conversion between MOAB tags and NetCDF variables
 * - Template-based file copying for consistent output format
 * - Chunked I/O for large 2D variables to avoid memory limits
 * - Support for multiple NetCDF data types (float, double, int, short, byte)
 * - Robust error handling with detailed error messages
 *
 * Author: Vijay Mahadevan
 * Date: 2025-2026
 */

#include "MeshIO.hpp"

// #include "ParallelPointCloudReader.hpp"
// #include "ScalarRemapper.hpp"
#include "easylogging.hpp"
#include "moab/ErrorHandler.hpp"

// C++ includes
#include <algorithm>
#include <array>
#include <filesystem>
#include <numeric>

namespace moab {

//===========================================================================
// Utility Functions
//===========================================================================

namespace {
/**
 * @brief Resolve string value or return fallback if empty
 *
 * Utility function that returns the input string if not empty,
 * otherwise returns the specified fallback value.
 *
 * @param value Input string to check
 * @param fallback Fallback string to return if input is empty
 * @return Either input value or fallback
 */
std::string resolve_or_default(const std::string &value,
                               const std::string &fallback) {
  return value.empty() ? fallback : value;
}

/**
 * @brief Convert vector from one numeric type to another
 *
 * Template function that converts elements from input vector type T
 * to output vector type U using static_cast. Preserves vector size
 * and performs element-wise conversion.
 *
 * @tparam T Input numeric type
 * @tparam U Output numeric type
 * @param input Input vector to convert
 * @param output Output vector (resized to match input)
 */
template <typename T, typename U>
void convert_to_type(const std::vector<T> &input, std::vector<U> &output) {
  output.resize(input.size(), U(0));
  std::transform(input.begin(), input.end(), output.begin(),
                 [](T val) { return static_cast<U>(val); });
}

} // namespace

//===========================================================================
// Point Cloud Loading Implementation
//===========================================================================

/**
 * @brief Load point cloud from NetCDF file into MOAB mesh
 *
 * Reads coordinate and area data from a NetCDF file and creates
 * MOAB vertex entities with appropriate tags. Converts longitude/latitude
 * coordinates to 3D Cartesian coordinates on unit sphere.
 *
 * Algorithm:
 * 1. Validate MOAB interface and resolve configuration options
 * 2. Open NetCDF file using PnetCDF for parallel access
 * 3. Read dimension size and validate
 * 4. Read coordinate arrays (lon, lat) and area data
 * 5. Convert each lon/lat pair to 3D Cartesian coordinates
 * 6. Create MOAB vertex entities
 * 7. Create and assign area tag to vertices
 * 8. Add vertices to specified mesh set
 * 9. Log loading statistics if verbose
 *
 * Coordinate Conversion:
 * - Input: Longitude (-180 to +180), Latitude (-90 to +90) in degrees
 * - Output: 3D Cartesian coordinates on unit sphere
 * - Uses RLLtoXYZ_Deg() for conversion
 *
 * Parallel I/O:
 * - Uses PnetCDF for parallel file access
 * - All processes read the same data (no domain decomposition)
 * - Suitable for moderate-sized datasets
 *
 * @param mb MOAB interface instance (must be valid)
 * @param mesh_set Target mesh set to contain loaded vertices
 * @param filename Path to NetCDF file to load
 * @param options Configuration options for loading
 * @param entities_out Output vector of created entity handles
 * @return MB_SUCCESS on success, MB_FAILURE on error
 *
 * @throws NetCDF exceptions for file I/O errors
 * @throws MOAB exceptions for mesh creation errors
 */
ErrorCode NetcdfMeshIO::load_point_cloud_from_file(
    Interface *mb, EntityHandle mesh_set, const std::string &filename,
    const NetcdfLoadOptions &options, std::vector<EntityHandle> &entities_out) {
  // Validate input parameters
  if (nullptr == mb)
    MB_SET_ERR(MB_FAILURE, "Invalid MOAB interface");

  // Resolve configuration options with defaults
  const std::string dim_name =
      resolve_or_default(options.dimension_name, "ncol");
  const std::string lon_name = resolve_or_default(options.lon_var_name, "lon");
  const std::string lat_name = resolve_or_default(options.lat_var_name, "lat");
  const std::string area_name =
      resolve_or_default(options.area_var_name, "area");

  try {
    // Open NetCDF file using PnetCDF for parallel access
    PnetCDF::NcmpiFile nc(MPI_COMM_WORLD, filename.c_str(),
                          PnetCDF::NcmpiFile::read);

    // Get dimension and validate
    PnetCDF::NcmpiDim dim = nc.getDim(dim_name);
    size_t ncol = dim.getSize();
    if (ncol <= 0)
      MB_SET_ERR(MB_FAILURE, "Dimension " << dim_name << " has no entries");

    // Allocate storage for coordinate and area data
    std::vector<double> lon(ncol), lat(ncol), area(ncol);

    // Get variable handles
    PnetCDF::NcmpiVar lon_var = nc.getVar(lon_name);
    PnetCDF::NcmpiVar lat_var = nc.getVar(lat_name);
    PnetCDF::NcmpiVar area_var = nc.getVar(area_name);

    // Check if chunked reading is needed
    bool use_chunked_reading = ncol > NetcdfMeshIO::MAX_ELEMENTS_PER_CHUNK;

    if (use_chunked_reading) {
      LOG(INFO) << "Using chunked reading for " << ncol << " points (exceeds "
                << NetcdfMeshIO::MAX_ELEMENTS_PER_CHUNK << " limit)";
    }

    // Prepare output entity vector
    entities_out.clear();
    entities_out.resize(ncol);

    // Read coordinate and area data with chunked approach if needed
    if (use_chunked_reading) {
      // Calculate chunk size
      size_t chunk_size = NetcdfMeshIO::MAX_ELEMENTS_PER_CHUNK;
      size_t num_chunks = (ncol + chunk_size - 1) / chunk_size;

      // LOG(INFO) << "Reading data in " << num_chunks << " chunks of up to " <<
      // chunk_size << " elements each";

      // Read data chunk by chunk directly into main arrays
      size_t current_start = 0;
      for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
        size_t current_count =
            std::min(chunk_size, static_cast<size_t>(ncol - current_start));

        LOG(INFO) << "Reading target chunk " << (chunk_idx + 1) << "/"
                  << num_chunks << " (elements " << current_start << " to "
                  << (current_start + current_count - 1) << ")";

        // Set up NetCDF hyperslab parameters for direct reading into main
        // arrays
        std::vector<MPI_Offset> start = {
            static_cast<MPI_Offset>(current_start)};
        std::vector<MPI_Offset> read_count = {
            static_cast<MPI_Offset>(current_count)};

        // Read directly into main arrays at the correct offset - no copying
        // needed!
        lon_var.getVar_all(start, read_count, lon.data() + current_start);
        lat_var.getVar_all(start, read_count, lat.data() + current_start);
        area_var.getVar_all(start, read_count, area.data() + current_start);

        // Convert lon/lat to 3D Cartesian coordinates on unit sphere
        std::vector<double> coords(3 * current_count);
#pragma omp parallel for shared(lon, lat, coords, current_start, current_count)
        for (size_t idx = current_start; idx < current_start + current_count;
             ++idx) {
          const size_t offset = (idx - current_start) * 3;
          RLLtoXYZ_Deg(lon[idx], lat[idx], coords.data() + offset);

          // Create MOAB vertex entity
          // MB_CHK_SET_ERR( mb->create_vertex( coords.data() + offset,
          // entities_out[idx] ), "Failed to create vertex" );
        }

        // Create MOAB vertex entity
        Range rngEnts;
        MB_CHK_SET_ERR(
            mb->create_vertices(coords.data(), current_count, rngEnts),
            "Failed to create vertices");
        std::copy(rngEnts.begin(), rngEnts.end(),
                  entities_out.data() + current_start);

        current_start += current_count;
      }
    } else {
      // Read all data at once for smaller datasets
      // LOG(INFO) << "Reading all " << ncol << " points at once";
      lon_var.getVar_all(lon.data());
      lat_var.getVar_all(lat.data());
      area_var.getVar_all(area.data());

      // Create MOAB vertices with chunked processing for large datasets
      for (size_t idx = 0; idx < ncol; ++idx) {
        // Convert lon/lat to 3D Cartesian coordinates on unit sphere
        std::array<double, 3> coords{0.0, 0.0, 0.0};
        RLLtoXYZ_Deg(lon[idx], lat[idx], coords);

        // Create MOAB vertex entity
        MB_CHK_SET_ERR(mb->create_vertex(coords.data(), entities_out[idx]),
                       "Failed to create vertex");
      }
    }

    // Add vertices to mesh set
    MB_CHK_SET_ERR(
        mb->add_entities(mesh_set, entities_out.data(), entities_out.size()),
        "Failed to add vertices to mesh set");

    // Create area tag for MOAB vertices
    Tag area_tag = nullptr;
    MB_CHK_SET_ERR(mb->tag_get_handle("area", 1, MB_TYPE_DOUBLE, area_tag,
                                      MB_TAG_DENSE | MB_TAG_CREAT),
                   "Failed to create area tag");

    // Assign area data to vertices
    MB_CHK_SET_ERR(mb->tag_set_data(area_tag, entities_out.data(),
                                    entities_out.size(), area.data()),
                   "Failed to assign area tag");

    // Log loading statistics if verbose
    if (options.verbose) {
      LOG(INFO) << "Loaded " << ncol << " points from NetCDF file " << filename
                << (options.context_label.empty()
                        ? std::string()
                        : " (" + options.context_label + ")");
    }

    // Close NetCDF file after reading data
    ncmpi_close(nc.getId());

    Range vertEnts;
    MB_CHK_SET_ERR(mb->get_entities_by_dimension(mesh_set, 0, vertEnts),
                   "Failed to get vertices");
    LOG(INFO) << "Loaded " << vertEnts.size()
              << " vertices from target NetCDF file " << filename;

  } catch (const PnetCDF::exceptions::NcmpiException &e) {
    MB_SET_ERR(MB_FAILURE,
               "NetCDF error while loading " << filename << ": " << e.what());
  }

  return MB_SUCCESS;
}

//===========================================================================
// Tag Data Fetching Utilities
//===========================================================================

namespace {
/**
 * @brief Fetch MOAB tag data as specified type
 *
 * Template function that reads tag data from MOAB entities and converts
 * it to the specified output type. Handles both double and integer tag
 * types with automatic conversion.
 *
 * Algorithm:
 * 1. Get tag data type from MOAB
 * 2. Based on tag type, read data directly or convert from int
 * 3. Return data in requested output type
 *
 * Supported Conversions:
 * - MB_TYPE_DOUBLE → T (direct read)
 * - MB_TYPE_INTEGER → T (int to T conversion)
 *
 * @tparam T Output data type
 * @param mb MOAB interface instance
 * @param tag MOAB tag handle
 * @param entities Vector of entity handles to read from
 * @param values Output vector of converted values
 * @return MB_SUCCESS on success, MB_FAILURE on error
 */
template <typename T>
ErrorCode fetch_tag_as_type(Interface *mb, Tag tag,
                            const std::vector<EntityHandle> &entities,
                            std::vector<T> &values) {
  // Get tag data type
  DataType type;
  MB_CHK_ERR(mb->tag_get_data_type(tag, type));

  const size_t count = entities.size();

  // Handle different tag types with appropriate conversion
  switch (type) {
  case MB_TYPE_DOUBLE: {
    // std::vector< double > buffer( count );
    // MB_CHK_ERR( mb->tag_get_data( tag, entities.data(), count, buffer.data()
    // ) ); convert_to_type( buffer, values ); Direct read for double tags
    values.resize(count);
    MB_CHK_ERR(mb->tag_get_data(tag, entities.data(), count, values.data()));
    return MB_SUCCESS;
  }
  case MB_TYPE_INTEGER: {
    // Read int tags and convert to output type
    std::vector<int> buffer(count);
    MB_CHK_ERR(mb->tag_get_data(tag, entities.data(), count, buffer.data()));
    convert_to_type(buffer, values);
    return MB_SUCCESS;
  }
  default:
    MB_SET_ERR(MB_FAILURE, "Unsupported tag data type for NetCDF write");
  }
}
} // namespace

//===========================================================================
// NetCDF Variable Writing Utilities
//===========================================================================

/**
 * @brief Template function to write variable data to NetCDF file
 *
 * Writes scalar field data from MeshData structure to a NetCDF variable.
 * Supports both 1D and 2D variables with automatic type conversion
 * and chunked I/O for large datasets to avoid memory limits.
 *
 * Algorithm:
 * 1. Get variable dimensions and calculate total size
 * 2. Find variable data in MeshData (double or integer fields)
 * 3. For 2D variables: use chunked I/O to avoid INT_MAX limits
 * 4. For 1D variables: write all data at once
 * 5. Convert data type to match NetCDF variable type
 * 6. Write data using PnetCDF parallel I/O
 *
 * Chunked I/O Strategy:
 * - Large 2D variables are written in latitude chunks
 * - NetcdfMeshIO::MAX_ELEMENTS_PER_CHUNK limits memory usage (250M elements)
 * - Prevents PnetCDF INT_MAX overflow errors
 *
 * Type Conversion:
 * - MeshData double/int → NetCDF variable type (T)
 * - Uses static_cast for safe conversion
 * - Preserves precision when possible
 *
 * @tparam T NetCDF variable data type
 * @param var NetCDF variable handle
 * @param mesh_data MeshData structure containing scalar fields
 * @return MB_SUCCESS on success, MB_FAILURE on error
 */
template <typename T>
static ErrorCode write_variable(const PnetCDF::NcmpiVar &var,
                                const ScalarRemapper::MeshData &mesh_data) {

  // Get variable dimensions
  std::vector<PnetCDF::NcmpiDim> dims = var.getDims();
  for (const auto &dim : dims) {
    LOG(INFO) << "Dimension " << dim.getName() << " has size " << dim.getSize();
  }

  // Check if it's a double variable in MeshData
  auto var_it = mesh_data.d_scalar_fields.find(var.getName());
  if (var_it != mesh_data.d_scalar_fields.end()) {
    const auto &values = var_it->second;

    // Handle 2D variables with chunked I/O
    if (dims.size() == 2) {
      size_t dim0_size = dims[0].getSize();
      size_t dim1_size = dims[1].getSize();
      size_t current_start = 0;
      size_t lat_chunk_size = NetcdfMeshIO::MAX_ELEMENTS_PER_CHUNK / dim1_size;
      std::vector<MPI_Offset> start, read_count;

      // Write data in latitude chunks to avoid memory limits
      for (size_t lat_offset = 0; lat_offset < dim0_size;
           lat_offset += lat_chunk_size) {
        size_t current_lat_count =
            std::min(lat_chunk_size, dim0_size - lat_offset);
        size_t chunk_elements = current_lat_count * dim1_size;
        LOG(INFO) << "\tWriting latitude chunk from " << lat_offset << " to "
                  << lat_offset + current_lat_count << ".";

        // Convert chunk data to NetCDF variable type
        std::vector<T> chunk_buffer(chunk_elements);
        std::transform(values.begin() + current_start,
                       values.begin() + current_start + chunk_elements,
                       chunk_buffer.begin(),
                       [](double d) { return static_cast<T>(d); });

        // Write chunk to NetCDF file
        start = {static_cast<MPI_Offset>(lat_offset),
                 static_cast<MPI_Offset>(0)};
        read_count = {static_cast<MPI_Offset>(current_lat_count),
                      static_cast<MPI_Offset>(dims[1].getSize())};
        var.putVar(start, read_count, chunk_buffer.data());
        current_start += chunk_elements;
      }
      // assert(current_start == total_size);
    } else {
      // 1D variable - check if chunked writing is needed for large datasets
      size_t values_size = values.size();

      if (values_size > NetcdfMeshIO::MAX_ELEMENTS_PER_CHUNK) {
        LOG(INFO) << "Using chunked writing for 1D variable '" << var.getName()
                  << "' with " << values_size << " elements";

        // Calculate chunk size
        size_t chunk_size = NetcdfMeshIO::MAX_ELEMENTS_PER_CHUNK;
        size_t num_chunks = (values_size + chunk_size - 1) / chunk_size;

        LOG(INFO) << "Writing 1D variable in " << num_chunks
                  << " chunks of up to " << chunk_size << " elements each";

        // Write data chunk by chunk
        for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
          size_t current_start = chunk_idx * chunk_size;
          size_t current_count =
              std::min(chunk_size, values_size - current_start);

          LOG(INFO) << "Writing chunk " << (chunk_idx + 1) << "/" << num_chunks
                    << " (elements " << current_start << " to "
                    << (current_start + current_count - 1) << ")";

          // Set up NetCDF hyperslab parameters
          std::vector<MPI_Offset> start = {
              static_cast<MPI_Offset>(current_start)};
          std::vector<MPI_Offset> write_count = {
              static_cast<MPI_Offset>(current_count)};

          // Convert and write chunk directly
          std::vector<T> chunk_buffer(current_count);
          std::transform(values.begin() + current_start,
                         values.begin() + current_start + current_count,
                         chunk_buffer.begin(),
                         [](double d) { return static_cast<T>(d); });
          var.putVar(start, write_count, chunk_buffer.data());
        }
      } else {
        // Write all data at once for smaller datasets
        std::vector<T> castValues(values.size());
        std::transform(values.begin(), values.end(), castValues.begin(),
                       [](double d) { return static_cast<T>(d); });
        var.putVar(castValues.data());
      }
    }
  } else {
    // Check if it's an integer variable in MeshData
    auto ivar_it = mesh_data.i_scalar_fields.find(var.getName());
    if (ivar_it != mesh_data.i_scalar_fields.end()) {
      const auto &values = ivar_it->second;

      if (dims.size() == 2) {
        // 2D integer variable with chunked I/O
        size_t dim0_size = dims[0].getSize();
        size_t dim1_size = dims[1].getSize();
        size_t current_start = 0;
        size_t lat_chunk_size =
            NetcdfMeshIO::MAX_ELEMENTS_PER_CHUNK / dim1_size;
        std::vector<MPI_Offset> start, read_count;

        for (size_t lat_offset = 0; lat_offset < dim0_size;
             lat_offset += lat_chunk_size) {
          size_t current_lat_count =
              std::min(lat_chunk_size, dim0_size - lat_offset);
          size_t chunk_elements = current_lat_count * dim1_size;
          LOG(INFO) << "\tWriting latitude chunk from " << lat_offset << " to "
                    << lat_offset + current_lat_count << ".";

          start = {static_cast<MPI_Offset>(lat_offset),
                   static_cast<MPI_Offset>(0)};
          read_count = {static_cast<MPI_Offset>(current_lat_count),
                        static_cast<MPI_Offset>(dims[1].getSize())};

          // Write directly for int type, convert for other types
          if constexpr (std::is_same_v<T, int>) {
            var.putVar(start, read_count, values.data() + current_start);
          } else {
            std::vector<T> chunk_buffer(chunk_elements);
            std::transform(values.begin() + current_start,
                           values.begin() + current_start + chunk_elements,
                           chunk_buffer.begin(),
                           [](int i) { return static_cast<T>(i); });
            var.putVar(start, read_count, chunk_buffer.data());
          }
          current_start += chunk_elements;
        }
        // assert(current_start == total_size);
      } else {
        // 1D integer variable - check if chunked writing is needed for large
        // datasets
        size_t values_size = values.size();

        if (values_size > NetcdfMeshIO::MAX_ELEMENTS_PER_CHUNK) {
          LOG(INFO) << "Using chunked writing for 1D integer variable '"
                    << var.getName() << "' with " << values_size << " elements";

          // Calculate chunk size
          size_t chunk_size = NetcdfMeshIO::MAX_ELEMENTS_PER_CHUNK;
          size_t num_chunks = (values_size + chunk_size - 1) / chunk_size;

          LOG(INFO) << "Writing 1D integer variable in " << num_chunks
                    << " chunks of up to " << chunk_size << " elements each";

          // Write data chunk by chunk
          for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
            size_t current_start = chunk_idx * chunk_size;
            size_t current_count =
                std::min(chunk_size, values_size - current_start);

            LOG(INFO) << "Writing chunk " << (chunk_idx + 1) << "/"
                      << num_chunks << " (elements " << current_start << " to "
                      << (current_start + current_count - 1) << ")";

            // Set up NetCDF hyperslab parameters
            std::vector<MPI_Offset> start = {
                static_cast<MPI_Offset>(current_start)};
            std::vector<MPI_Offset> write_count = {
                static_cast<MPI_Offset>(current_count)};

            // Write directly for int type, convert for other types
            if constexpr (std::is_same_v<T, int>) {
              var.putVar(start, write_count, values.data() + current_start);
            } else {
              std::vector<T> chunk_buffer(current_count);
              std::transform(values.begin() + current_start,
                             values.begin() + current_start + current_count,
                             chunk_buffer.begin(),
                             [](int i) { return static_cast<T>(i); });
              var.putVar(start, write_count, chunk_buffer.data());
            }
          }
        } else {
          // Write all data at once for smaller datasets
          std::vector<T> castValues(values.size());
          std::transform(values.begin(), values.end(), castValues.begin(),
                         [](int i) { return static_cast<T>(i); });
          var.putVar(castValues.data());
        }
      }
    }
  }

  return MB_SUCCESS;
};

//===========================================================================
// NetCDF Writing Implementation
//===========================================================================

/**
 * @brief Write scalar fields from MOAB entities to NetCDF file using MeshData
 *
 * Writes scalar field data from a MeshData structure to a NetCDF file
 * using a template file for format preservation. Supports both 1D and 2D
 * variables with automatic type conversion and chunked I/O for large datasets.
 *
 * Algorithm:
 * 1. Validate input parameters and MeshData
 * 2. Copy template file to output location
 * 3. Open output file in write mode with PnetCDF
 * 4. Begin independent data mode for parallel I/O
 * 5. For each requested variable:
 *    - Find variable in NetCDF file
 *    - Determine variable type and dimensions
 *    - Write data using appropriate type conversion
 *    - Use chunked I/O for large 2D variables
 * 6. End independent data mode and close file
 * 7. Log writing statistics if verbose
 *
 * Parallel I/O Strategy:
 * - Uses PnetCDF independent data mode
 * - Single process handles file operations (commented OpenMP directive)
 * - Suitable for both shared and distributed memory systems
 *
 * Memory Management:
 * - Chunked I/O prevents memory overflow for large 2D variables
 * - NetcdfMeshIO::MAX_ELEMENTS_PER_CHUNK limits memory usage (250M elements)
 * - Automatic type conversion preserves precision
 *
 * @param mb MOAB interface instance (must be valid)
 * @param template_file Template NetCDF file to copy format from
 * @param output_file Output NetCDF file path
 * @param request Configuration for which fields to write
 * @param mesh_data MeshData structure containing scalar fields
 * @return MB_SUCCESS on success, MB_FAILURE on error
 *
 * @throws Filesystem exceptions for file copy errors
 * @throws NetCDF exceptions for I/O errors
 * @throws Logic exceptions for unsupported data types
 */
ErrorCode NetcdfMeshIO::write_point_scalars_to_file(
    Interface *mb, const std::string &template_file,
    const std::string &output_file, const NetcdfWriteRequest &request,
    const ScalarRemapper::MeshData &mesh_data) {
  if (nullptr == mb)
    MB_SET_ERR(MB_FAILURE, "Invalid MOAB interface");
  if (!mesh_data.d_scalar_fields.size() && !mesh_data.i_scalar_fields.size())
    MB_SET_ERR(MB_FAILURE, "No entities provided for NetCDF write");

  const std::string dim_name =
      resolve_or_default(request.dimension_name, "ncol");

  try {
    auto start_time = std::chrono::high_resolution_clock::now();

    std::filesystem::copy_file(
        template_file, output_file,
        std::filesystem::copy_options::overwrite_existing);

    // Report writing time
    auto copy_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start_time);
    LOG(INFO) << "Time taken to copy target template file: "
              << copy_time.count() << " milliseconds";

    // #pragma omp single
    {
      PnetCDF::NcmpiFile out(MPI_COMM_WORLD, output_file.c_str(),
                             PnetCDF::NcmpiFile::write,
                             PnetCDF::NcmpiFile::classic5);

      // Begin the independent data mode
      int status = ncmpi_begin_indep_data(out.getId());
      if (status != NC_NOERR) {
        throw std::runtime_error(
            "Failed to begin independent data mode for NetCDF write");
      }

      auto vars = out.getVars();
      for (const auto &name : request.scalar_var_names) {
        for (const auto &ncvar : vars) {
          if (ncvar.first == name) {
            const auto &variable = ncvar.second;
            if (variable.getType() == PnetCDF::ncmpiFloat) {
              MB_CHK_ERR(write_variable<float>(variable, mesh_data));
            } else if (variable.getType() == PnetCDF::ncmpiDouble) {
              MB_CHK_ERR(write_variable<double>(variable, mesh_data));
            } else if (variable.getType() == PnetCDF::ncmpiInt) {
              MB_CHK_ERR(write_variable<int>(variable, mesh_data));
            } else if (variable.getType() == PnetCDF::ncmpiShort) {
              MB_CHK_ERR(write_variable<short>(variable, mesh_data));
            } else if (variable.getType() == PnetCDF::ncmpiByte) {
              MB_CHK_ERR(write_variable<signed char>(variable, mesh_data));
            } else {
              throw std::logic_error(
                  "Unsupported variable type for NetCDF write");
            }

            if (request.verbose) {
              LOG(INFO) << "Wrote NetCDF variable " << ncvar.first << " to "
                        << output_file;
            }
          }
        }
      }

      // End independent data mode if you are done with independent operations
      status = ncmpi_end_indep_data(out.getId());
      if (status != NC_NOERR) {
        throw std::runtime_error(
            "Failed to end independent data mode for NetCDF write");
      }

      // for( const auto& base_name : request.squared_var_names )
      // {
      //     const std::string var_name = base_name + "_squared";
      //     MB_CHK_ERR( write_variable( var_name ) );
      // }

      ncmpi_close(out.getId());
    }
  } catch (const std::filesystem::filesystem_error &e) {
    MB_SET_ERR(MB_FAILURE, "File copy error for NetCDF output: " << e.what());
  } catch (const PnetCDF::exceptions::NcmpiException &e) {
    MB_SET_ERR(MB_FAILURE, "NetCDF error while writing " << output_file << ": "
                                                         << e.what());
  } catch (const std::exception &e) {
    MB_SET_ERR(MB_FAILURE,
               "Exception while writing " << output_file << ": " << e.what());
  }

  return MB_SUCCESS;
}

/**
 * @brief Write scalar fields from MOAB entity tags to NetCDF file
 *
 * Writes scalar field data from MOAB entity tags to a NetCDF file.
 * Creates new variables in the output file if they don't exist
 * in the template. Supports automatic type conversion from MOAB
 * tag data to NetCDF variable types.
 *
 * Algorithm:
 * 1. Validate input parameters and entities
 * 2. Copy template file to output location
 * 3. Open output file in write mode
 * 4. Validate entity count matches NetCDF dimension
 * 5. Create new variables for requested fields if needed
 * 6. For each requested variable:
 *    - Get MOAB tag handle
 *    - Read tag data from entities with type conversion
 *    - Write data to NetCDF variable
 * 7. Compute and write squared fields if requested
 * 8. Log writing statistics if verbose
 *
 * Variable Creation Strategy:
 * - Uses TagValueType to determine NetCDF variable type
 * - Creates variables with proper dimension references
 * - Supports both float and double precision output
 *
 * Type Conversion Support:
 * - MOAB double/int tags → NetCDF variables
 * - Automatic conversion to TagValueType
 * - Preserves precision when possible
 *
 * @param mb MOAB interface instance (must be valid)
 * @param template_file Template NetCDF file to copy format from
 * @param output_file Output NetCDF file path
 * @param request Configuration for which fields to write
 * @param entities Vector of entity handles to read data from
 * @return MB_SUCCESS on success, MB_FAILURE on error
 *
 * @throws Filesystem exceptions for file copy errors
 * @throws NetCDF exceptions for I/O errors
 * @throws MOAB exceptions for tag access errors
 */
ErrorCode NetcdfMeshIO::write_point_scalars_to_file(
    Interface *mb, const std::string &template_file,
    const std::string &output_file, const NetcdfWriteRequest &request,
    const std::vector<EntityHandle> &entities) {
  if (nullptr == mb)
    MB_SET_ERR(MB_FAILURE, "Invalid MOAB interface");
  if (entities.empty())
    MB_SET_ERR(MB_FAILURE, "No entities provided for NetCDF write");

  const std::string dim_name =
      resolve_or_default(request.dimension_name, "ncol");

  try {
    auto start_time = std::chrono::high_resolution_clock::now();

    std::filesystem::copy_file(
        template_file, output_file,
        std::filesystem::copy_options::overwrite_existing);

    // Report writing time
    auto copy_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start_time);
    LOG(INFO) << "Time taken to copy target template file: "
              << copy_time.count() << " milliseconds";

    PnetCDF::NcmpiFile out(MPI_COMM_WORLD, output_file.c_str(),
                           PnetCDF::NcmpiFile::write);
    PnetCDF::NcmpiDim dim = out.getDim(dim_name);
    MPI_Offset ncol = dim.getSize();

    if (static_cast<size_t>(ncol) != entities.size()) {
      MB_SET_ERR(MB_FAILURE, "Entity count ("
                                 << entities.size()
                                 << ") does not match NetCDF dimension "
                                 << dim_name << " (" << ncol << ")");
    }

    std::vector<PnetCDF::NcmpiDim> dims;
    dims.push_back(dim);

    auto define_variable = [&](const std::string &var_name) {
      if constexpr (std::is_same_v<TagValueType, float>) {
        out.addVar(var_name, PnetCDF::ncmpiFloat, dims);
      } else {
        out.addVar(var_name, PnetCDF::ncmpiDouble, dims);
      }
    };

    for (const auto &name : request.scalar_var_names)
      define_variable(name);
    for (const auto &base_name : request.squared_var_names)
      define_variable(base_name + "_squared");

    out.enddef();

    auto write_variable = [&](const std::string &var_name) -> ErrorCode {
      Tag tag = 0;
      MB_CHK_SET_ERR(mb->tag_get_handle(var_name.c_str(), tag),
                     "Failed to get tag for " << var_name);

      std::vector<TagValueType> values;
      MB_CHK_ERR(fetch_tag_as_type<TagValueType>(mb, tag, entities, values));

      PnetCDF::NcmpiVar var = out.getVar(var_name);

      // Check if chunked writing is needed for large datasets
      size_t num_entities = values.size();
      bool use_chunked_writing =
          num_entities > NetcdfMeshIO::MAX_ELEMENTS_PER_CHUNK;

      if (use_chunked_writing) {
        LOG(INFO) << "Using chunked writing for " << num_entities
                  << " entities (exceeds "
                  << NetcdfMeshIO::MAX_ELEMENTS_PER_CHUNK << " limit)";

        // Calculate chunk size
        size_t chunk_size = NetcdfMeshIO::MAX_ELEMENTS_PER_CHUNK;
        size_t num_chunks = (num_entities + chunk_size - 1) / chunk_size;

        LOG(INFO) << "Writing " << var_name << " in " << num_chunks
                  << " chunks of up to " << chunk_size << " elements each";

        // Write data chunk by chunk
        for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
          size_t current_start = chunk_idx * chunk_size;
          size_t current_count =
              std::min(chunk_size, num_entities - current_start);

          LOG(INFO) << "Writing chunk " << (chunk_idx + 1) << "/" << num_chunks
                    << " (entities " << current_start << " to "
                    << (current_start + current_count - 1) << ")";

          // Set up NetCDF hyperslab parameters
          std::vector<MPI_Offset> start = {
              static_cast<MPI_Offset>(current_start)};
          std::vector<MPI_Offset> write_count = {
              static_cast<MPI_Offset>(current_count)};

          // Write chunk directly to NetCDF file
          var.putVar_all(start, write_count, values.data() + current_start);
        }
      } else {
        // Write all data at once for smaller datasets
        LOG(INFO) << "Writing all " << num_entities << " " << var_name
                  << " values at once";
        var.putVar_all(values.data());
      }

      if (request.verbose) {
        LOG(INFO) << "Wrote NetCDF variable " << var_name << " to "
                  << output_file;
      }
      return MB_SUCCESS;
    };

    for (const auto &name : request.scalar_var_names)
      MB_CHK_ERR(write_variable(name));

    for (const auto &base_name : request.squared_var_names) {
      const std::string var_name = base_name + "_squared";
      MB_CHK_ERR(write_variable(var_name));
    }

    ncmpi_close(out.getId());
  } catch (const std::filesystem::filesystem_error &e) {
    MB_SET_ERR(MB_FAILURE, "File copy error for NetCDF output: " << e.what());
  } catch (const PnetCDF::exceptions::NcmpiException &e) {
    MB_SET_ERR(MB_FAILURE, "NetCDF error while writing " << output_file << ": "
                                                         << e.what());
  }

  return MB_SUCCESS;
}

} // namespace moab

/**
 * @file MeshIO.hpp
 * @brief NetCDF I/O operations for point cloud meshes
 *
 * This file provides utilities for loading and writing point cloud data
 * to/from NetCDF files using MOAB mesh entities. It supports both
 * structured and unstructured grid formats with parallel I/O capabilities.
 *
 * Key Features:
 * - Load point clouds from NetCDF files with coordinate and area data
 * - Write scalar fields from MOAB tags to NetCDF files
 * - Support for multiple NetCDF data types (float, double, int, short, byte)
 * - Parallel I/O using PnetCDF for large datasets
 * - Template-based file copying for consistent output format
 *
 * Author: Vijay Mahadevan
 * Date: 2025-2026
 */

#ifndef NETCDF_MESH_IO_HPP
#define NETCDF_MESH_IO_HPP

#include "moab/Interface.hpp"
#include "ScalarRemapper.hpp"

#include <string>
#include <vector>

namespace moab
{

/**
 * @brief Type alias for tag values (double precision by default)
 *
 * This defines the precision used for scalar field operations
 * when writing to NetCDF files. Can be changed to float for
 * reduced memory usage when appropriate.
 */
typedef double TagValueType;

/**
 * @brief Configuration options for loading NetCDF point cloud files
 *
 * This structure provides flexible configuration for loading point cloud
 * data from NetCDF files, allowing customization of variable names and
 * logging behavior.
 */
struct NetcdfLoadOptions
{
    /** @brief Name of the dimension containing point count (default: "ncol") */
    std::string dimension_name = "ncol";

    /** @brief Name of longitude variable in NetCDF file (default: "lon") */
    std::string lon_var_name   = "lon";

    /** @brief Name of latitude variable in NetCDF file (default: "lat") */
    std::string lat_var_name   = "lat";

    /** @brief Name of area variable in NetCDF file (default: "area") */
    std::string area_var_name  = "area";

    /** @brief Optional context label for logging and debugging */
    std::string context_label;

    /** @brief Enable verbose logging for debugging purposes */
    bool verbose = false;
};

/**
 * @brief Configuration for writing scalar fields to NetCDF files
 *
 * This structure controls which scalar fields are written to NetCDF
 * files and how they are formatted. Supports both regular fields and
 * automatically computed squared fields.
 */
struct NetcdfWriteRequest
{
    /** @brief List of scalar field names to write to NetCDF file */
    std::vector< std::string > scalar_var_names;

    /** @brief List of field names to compute squared values for */
    std::vector< std::string > squared_var_names;

    /** @brief Name of the dimension containing point count (default: "ncol") */
    std::string dimension_name = "ncol";

    /** @brief Enable verbose logging for debugging purposes */
    bool verbose               = false;
};

// Forward declaration
class ScalarRemapper;
struct MeshData;

/**
 * @brief NetCDF I/O utilities for point cloud meshes
 *
 * This class provides static methods for loading point cloud data from
 * NetCDF files and writing scalar fields from MOAB entities to NetCDF files.
 * It supports both parallel and serial I/O operations with automatic
 * type conversion and memory management.
 *
 * Key Capabilities:
 * - Load point clouds with coordinates and area data
 * - Convert lon/lat to 3D Cartesian coordinates
 * - Write scalar fields from MOAB tags to NetCDF variables
 * - Support for multiple NetCDF data types with automatic conversion
 * - Template-based file copying for consistent output format
 * - Parallel I/O using PnetCDF for large datasets
 *
 * Usage Example:
 * @code
 * // Load point cloud
 * NetcdfLoadOptions options;
 * options.verbose = true;
 * std::vector<EntityHandle> entities;
 * NetcdfMeshIO::load_point_cloud_from_file(mb, mesh_set, "input.nc", options, entities);
 *
 * // Write scalar fields
 * NetcdfWriteRequest request;
 * request.scalar_var_names = {"temperature", "pressure"};
 * NetcdfMeshIO::write_point_scalars_to_file(mb, "template.nc", "output.nc", request, entities);
 * @endcode
 */
class NetcdfMeshIO
{
  public:
    /**
     * @brief Load point cloud from NetCDF file into MOAB mesh
     *
     * Reads coordinate and area data from a NetCDF file and creates
     * MOAB vertex entities with appropriate tags. Converts longitude/latitude
     * coordinates to 3D Cartesian coordinates on unit sphere.
     *
     * Algorithm:
     * 1. Open NetCDF file using PnetCDF for parallel access
     * 2. Read dimension size and validate
     * 3. Read coordinate arrays (lon, lat) and area data
     * 4. Convert each lon/lat pair to 3D Cartesian coordinates
     * 5. Create MOAB vertex entities
     * 6. Create and assign area tag to vertices
     * 7. Add vertices to specified mesh set
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
    static ErrorCode load_point_cloud_from_file( Interface* mb,
                                                 EntityHandle mesh_set,
                                                 const std::string& filename,
                                                 const NetcdfLoadOptions& options,
                                                 std::vector< EntityHandle >& entities_out );

    /**
     * @brief Write scalar fields from MOAB entities to NetCDF file using MeshData
     *
     * Writes scalar field data from a MeshData structure to a NetCDF file
     * using a template file for format preservation. Supports both 1D and 2D
     * variables with automatic type conversion and chunked I/O for large datasets.
     *
     * Algorithm:
     * 1. Copy template file to output location
     * 2. Open output file in write mode with PnetCDF
     * 3. Begin independent data mode for parallel I/O
     * 4. For each requested variable:
     *    - Find variable in NetCDF file
     *    - Determine variable type and dimensions
     *    - Write data using appropriate type conversion
     *    - Use chunked I/O for large 2D variables
     * 5. End independent data mode and close file
     *
     * Supported Data Types:
     * - Float (single precision)
     * - Double (double precision)
     * - Integer (32-bit)
     * - Short (16-bit)
     * - Byte (8-bit signed)
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
    static ErrorCode write_point_scalars_to_file( Interface* mb,
                                                  const std::string& template_file,
                                                  const std::string& output_file,
                                                  const NetcdfWriteRequest& request,
                                                  const moab::ScalarRemapper::MeshData& mesh_data );

    /**
     * @brief Write scalar fields from MOAB entity tags to NetCDF file
     *
     * Writes scalar field data from MOAB entity tags to a NetCDF file.
     * Creates new variables in the output file if they don't exist
     * in the template. Supports automatic type conversion from MOAB
     * tag data to NetCDF variable types.
     *
     * Algorithm:
     * 1. Copy template file to output location
     * 2. Open output file in write mode
     * 3. Validate entity count matches NetCDF dimension
     * 4. Create new variables for requested fields if needed
     * 5. For each requested variable:
     *    - Get MOAB tag handle
     *    - Read tag data from entities with type conversion
     *    - Write data to NetCDF variable
     * 6. Compute and write squared fields if requested
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
    static ErrorCode write_point_scalars_to_file( Interface* mb,
                                                  const std::string& template_file,
                                                  const std::string& output_file,
                                                  const NetcdfWriteRequest& request,
                                                  const std::vector< EntityHandle >& entities );
};

}  // namespace moab

#endif  // NETCDF_MESH_IO_HPP

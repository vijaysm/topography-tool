/**
 * @file mbda.cpp
 * @brief Main executable for mapping NetCDF point cloud data to target meshes
 *
 * This file implements the mbda (Mesh-Based Data Averaging) tool that maps
 * environmental data from NetCDF point clouds to target mesh elements using
 * disk-based averaging algorithms. The tool supports both structured and
 * unstructured grids with MPI parallelism and OpenMP threading.
 *
 * Key Features:
 * - Parallel point cloud reading using PnetCDF
 * - Multiple remapping algorithms (disk averaging, nearest neighbor)
 * - Support for spectral element meshes and USGS format data
 * - Automatic format detection and coordinate system handling
 * - Output to both NetCDF and H5M formats
 * - Comprehensive command-line interface with MOAB ProgOptions
 * - Thread-safe logging with configurable verbosity
 *
 * Algorithms Supported:
 * - ALG_DISKAVERAGE: Disk-based averaging with monotone properties
 * - ALG_NEAREST_NEIGHBOR: Fast nearest neighbor mapping
 *
 * Author: Vijay Mahadevan
 * Date: 2025-2026
 */

// Suppress warnings from easylogging++ header
#include "easylogging.hpp"

// Let us initialize easylogging++
// Can be in exactly one source file only
INITIALIZE_EASYLOGGINGPP

#include "MeshIO.hpp"
#include "ParallelPointCloudReader.hpp"
#include "ScalarRemapper.hpp"
#include "moab/Core.hpp"
#include "moab/ProgOptions.hpp"

// Standard includes
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <sstream>

#include <omp.h>

using namespace moab;

//===========================================================================
// Utility Functions
//===========================================================================

/**
 * @brief Parse comma-separated string into vector of strings
 *
 * Utility function that parses a comma-separated string and returns
 * a vector of individual strings with whitespace trimmed.
 *
 * @param input Input string containing comma-separated values
 * @return Vector of parsed strings (empty if input is empty)
 */
std::vector< std::string > parse_comma_separated( const std::string& input )
{
    std::vector< std::string > result;
    if( input.empty() ) return result;

    std::stringstream ss( input );
    std::string item;
    while( std::getline( ss, item, ',' ) )
    {
        // Trim whitespace from both ends
        item.erase( 0, item.find_first_not_of( " \t" ) );
        item.erase( item.find_last_not_of( " \t" ) + 1 );
        if( !item.empty() )
        {
            result.push_back( item );
        }
    }
    return result;
}

/**
 * @brief Get the number of OpenMP threads available
 *
 * Uses OpenMP parallel reduction to count the total number of threads.
 * This is a portable way to determine thread count across different
 * OpenMP implementations.
 *
 * @return Total number of OpenMP threads
 *
 * Source: https://stackoverflow.com/a
 * Posted by Moehre2, modified by community. See post 'Timeline' for change
 * history Retrieved 2025-11-10, License - CC BY-SA 4.0
 */
int get_num_threads( void )
{
    int num_threads = 0;
#pragma omp parallel reduction( + : num_threads )
    num_threads += 1;
    return num_threads;
}

//===========================================================================
// Main Function
//===========================================================================

/**
 * @brief Main function for mbda - Mesh-Based Data Averaging tool
 *
 * This is the main entry point for the mbda tool that maps NetCDF point cloud
 * data to target mesh elements using disk-based averaging algorithms that are
 * guaranteed to be monotone. The tool supports MPI parallelism and OpenMP
 * threading for scalable performance on large datasets.
 *
 * Algorithm Overview:
 * 1. Initialize MPI and configure logging
 * 2. Parse command-line arguments using MOAB ProgOptions
 * 3. Load target mesh (spectral or point cloud)
 * 4. Configure and read source point cloud data
 * 5. Perform scalar remapping using selected algorithm
 * 6. Write remapped data to output file (NetCDF or H5M)
 * 7. Clean up and exit
 *
 * Parallel Execution:
 * - MPI: Distributed memory parallelism for data distribution
 * - OpenMP: Shared memory parallelism for computational kernels
 * - Thread-safe logging with rank-specific output control
 *
 * Usage Examples:
 * @code
 * # Basic disk averaging remapping
 * mpiexec -n 4 ./mbda --source data.nc --target mesh.h5m --output output.h5m
 *
 * # Nearest neighbor remapping with custom fields
 * mpiexec -n 2 ./mbda --source data.nc --target mesh.nc --output output.nc \
 *   --remap-method nn --fields temperature,pressure --verbose
 *
 * # Spectral element mesh with area smoothing
 * mpiexec -n 8 ./mbda --source usgs_data.nc --target spectral_mesh.h5m \
 *   --output spectral_output.h5m --spectral --smoothing-area 0.01
 * @endcode
 *
 * @param argc Number of command-line arguments
 * @param argv Array of command-line argument strings
 * @return 0 on success, 1 on error
 *
 * @note OpenMP parallelism is available - set OMP_NUM_THREADS environment
 * variable
 * @note Requires MPI initialization for parallel execution
 */
int main( int argc, char* argv[] )
{

    // Configure easylogging++ (will be reconfigured after parsing args)
    START_EASYLOGGINGPP( argc, argv );

    // Configure thread-safe logging for OpenMP parallelism
    // el::Loggers::addFlag(el::LoggingFlag::DisableApplicationAbortOnFatalLog);
    el::Loggers::addFlag( el::LoggingFlag::MultiLoggerSupport );
    el::Loggers::addFlag( el::LoggingFlag::StrictLogFileSizeCheck );
    el::Loggers::addFlag( el::LoggingFlag::ColoredTerminalOutput );

    MPI_Init( &argc, &argv );

    try
    {
        //=======================================================================
        // Command-Line Argument Parsing
        //=======================================================================

        // Parse command-line arguments using MOAB's ProgOptions
        ProgOptions opts;

        // Required arguments
        std::string source_file = "";
        std::string target_file = "";
        std::string output_file = "";

        // Optional variable name overrides
        std::string dimension_variable        = "";
        std::string lon_variable              = "";
        std::string lat_variable              = "";
        std::string area_variable             = "";
        std::string fields_string_list        = "";
        std::string square_fields_string_list = "";
        bool reuse_source_mesh                = false;
        double user_smoothing_area            = 0.0;

        // Remapping options
        std::string remap_method = "ALG_DISKAVERAGE";
        bool verbose             = false;
        bool spectral_target     = false;

        // Define command-line options
        opts.addOpt< std::string >( "source", "Source NetCDF point cloud file", &source_file );
        opts.addOpt< std::string >( "target", "Target mesh file (nc or H5M)", &target_file );
        opts.addOpt< std::string >( "output", "Output mesh file with remapped data (ending with nc or h5m)",
                                    &output_file );

        opts.addOpt< std::string >( "dof-var",
                                    "DoF numbering variable name (bypasses format "
                                    "detection). Default: ncol",
                                    &dimension_variable );
        opts.addOpt< std::string >( "lon-var", "Longitude variable name (bypasses format detection). Default: lon",
                                    &lon_variable );
        opts.addOpt< std::string >( "lat-var", "Latitude variable name (bypasses format detection). Default: lat",
                                    &lat_variable );
        opts.addOpt< std::string >( "area-var", "Area variable name to read and store. Default: area", &area_variable );

        opts.addOpt< std::string >( "fields", "Comma-separated field names to remap", &fields_string_list );
        opts.addOpt< std::string >( "square-fields",
                                    "Comma-separated quadratic field names to "
                                    "remap (stored as: <field>_squared)",
                                    &square_fields_string_list );
        opts.addOpt< std::string >( "remap-method",
                                    "Remapping method: da (ALG_DISKAVERAGE) or nn "
                                    "(ALG_NEAREST_NEIGHBOR). Default: da",
                                    &remap_method );

        opts.addOpt< void >( "spectral",
                             "Assume that the target mesh requires online "
                             "spectral element mesh treatment. Default: false",
                             &spectral_target );
        opts.addOpt< void >( "reuse-source-mesh",
                             "Skip loading a target mesh file and reuse the "
                             "source point cloud as the target mesh",
                             &reuse_source_mesh );
        opts.addOpt< void >( "verbose,v", "Enable verbose output with timestamps. Default: false", &verbose );
        opts.addOpt< double >( "smoothing-area", "Smoothing area for the target mesh. Default: 0.0",
                               &user_smoothing_area );

        // Parse command-line arguments
        opts.parseCommandLine( argc, argv );

        // Validate command-line arguments
        if( reuse_source_mesh && user_smoothing_area < 1E-12 )
        {
            LOG( ERROR ) << "Error: Smoothing area needs to be specified when "
                            "using reuse-source-mesh";
            return 1;
        }

        //=======================================================================
        // Logging Configuration
        //=======================================================================

        // Configure logging format based on verbose flag
        el::Configurations defaultConf;
        defaultConf.setToDefault();

        if( verbose )
        {
            // Verbose: show level, timestamp, and message
            defaultConf.set( el::Level::Global, el::ConfigurationType::Format, "[%level: %datetime{%H:%m:%s}] %msg" );
        }
        else
        {
            // Quiet: only show message
            defaultConf.set( el::Level::Global, el::ConfigurationType::Format, "%msg" );
        }
        defaultConf.set( el::Level::Global, el::ConfigurationType::ToFile, "false" );
        defaultConf.set( el::Level::Global, el::ConfigurationType::ToStandardOutput, "true" );

        el::Loggers::reconfigureLogger( "default", defaultConf );

        // Validate required arguments
        if( source_file.empty() || target_file.empty() )
        {
            LOG( ERROR ) << "Source and target files are required.";
            opts.printHelp();
            return 1;
        }

        //=======================================================================
        // Configuration Summary
        //=======================================================================

        // Print configuration summary
        LOG( INFO ) << "==================================";
        LOG( INFO ) << "=== mbda: Configuration ===";
        LOG( INFO ) << "Source file: " << source_file;
        LOG( INFO ) << "Target file: " << target_file;
        if( !output_file.empty() )
        {
            LOG( INFO ) << "Output file: " << output_file;
        }
        if( !lon_variable.empty() && !lat_variable.empty() )
        {
            LOG( INFO ) << "Coordinate variable overrides: " << lon_variable << ", " << lat_variable;
        }
        if( !area_variable.empty() )
        {
            LOG( INFO ) << "Area variable: " << area_variable;
        }
        if( !fields_string_list.empty() )
        {
            LOG( INFO ) << "Field overrides: " << fields_string_list;
        }
        if( !square_fields_string_list.empty() )
        {
            LOG( INFO ) << "Square fields: " << square_fields_string_list;
        }
        LOG( INFO ) << "Remap method: " << remap_method << " ("
                    << ( spectral_target ? "Spectral Target" : "Generic Target" ) << ")";
        LOG( INFO ) << "Number of threads: " << get_num_threads();
        LOG( INFO ) << "====================================\n";

        //=======================================================================
        // MOAB Initialization and Target Mesh Loading
        //=======================================================================

        // Initialize MOAB core interface
        Core moab_core;
        Interface* mb = &moab_core;

        // Create mesh set for organizing mesh entities
        EntityHandle mesh_set;
        ErrorCode rval = mb->create_meshset( MESHSET_SET, mesh_set );
        if( MB_SUCCESS != rval )
        {
            LOG( ERROR ) << "Failed to create mesh set";
            return 1;
        }

        std::vector< EntityHandle > entities;
        const bool build_target_from_source = reuse_source_mesh && !spectral_target;

        // Load target mesh based on type
        if( spectral_target )
        {
            // Spectral element mesh: use parallel options for distributed
            // loading
            std::string read_opts = "DEBUG_IO=0";
            // if( size > 1 )  // If reading in parallel, need to tell it how
            //     read_opts +=
            //     ";PARALLEL=READ_PART;PARTITION=PARALLEL_PARTITION;PARALLEL_RESOLVE_SHARED_ENTS;DEBUG_PIO=0";

            // Load the spectral mesh
            MB_CHK_SET_ERR( mb->load_file( target_file.c_str(), &mesh_set, read_opts.c_str() ),
                            "Failed to load mesh file" );

            // Get all 2D elements in the mesh set
            MB_CHK_SET_ERR( mb->get_entities_by_dimension( mesh_set, 2, entities ),
                            "Failed to get elements by dimension" );
        }
        else if( !build_target_from_source )
        {
            // Point cloud mesh: use NetCDF mesh I/O
            NetcdfLoadOptions load_opts;
            load_opts.dimension_name = dimension_variable;
            load_opts.lon_var_name   = lon_variable;
            load_opts.lat_var_name   = lat_variable;
            load_opts.area_var_name  = area_variable;
            load_opts.context_label  = "source point-cloud mesh";
            load_opts.verbose        = verbose;

            MB_CHK_SET_ERR( NetcdfMeshIO::load_point_cloud_from_file( mb, mesh_set, target_file, load_opts, entities ),
                            "Failed to load NetCDF mesh file" );
        }
        else
        {
            // Reuse source mesh as target
            LOG( INFO ) << "Reuse-source-mesh is enabled; target mesh will be "
                           "built from the source point cloud.";
        }

        // Log target mesh loading status
        if( !reuse_source_mesh )
        {
            LOG( INFO ) << "Loaded target mesh: " << target_file;
        }

        //=======================================================================
        // Source Point Cloud Reading
        //=======================================================================

        // Configure the parallel point cloud reader
        ParallelPointCloudReader reader( mb, mesh_set );
        ParallelPointCloudReader::ReadConfig config;
        config.netcdf_filename  = source_file;
        config.print_statistics = true;
        config.verbose          = verbose;

        // Apply coordinate variable overrides (bypasses format detection)
        config.coord_var_names = { "lon", "lat" };

        // Apply field name overrides (replaces auto-detection)
        if( !fields_string_list.empty() )
        {
            config.scalar_var_names = parse_comma_separated( fields_string_list );
        }

        // Store square fields list for automatic computation
        if( !square_fields_string_list.empty() )
        {
            config.square_field_names = parse_comma_separated( square_fields_string_list );
        }

        // Configure the reader with parsed options
        reader.configure( config );

        // Read source point cloud data with timing
        auto start_time = std::chrono::high_resolution_clock::now();

        ParallelPointCloudReader::PointData local_points;
        MB_CHK_SET_ERR( reader.read_points( local_points ), "Failed to read points" );

        // Note: Commented code for building target mesh from source point cloud
        // This functionality can be enabled for self-remapping scenarios
        // if (build_target_from_source) {
        //     MB_CHK_SET_ERR(
        //         reader.build_mesh_from_point_cloud(local_points, mb,
        //         mesh_set, entities, user_smoothing_area), "Failed to
        //         construct target mesh from source point cloud");
        //     LOG(INFO) << "Constructed target mesh from source point cloud
        //     with " << entities.size() << " vertices.";
        // }

        // Print point cloud reading statistics
        if( reader.get_config().print_statistics )
        {
            auto read_duration = std::chrono::duration_cast< std::chrono::milliseconds >(
                std::chrono::high_resolution_clock::now() - start_time );

            LOG( INFO ) << "===================================\n";
            LOG( INFO ) << "=== Point Cloud Reading Results ===";
            LOG( INFO ) << "Total points in dataset: " << reader.get_point_count();
            LOG( INFO ) << "Points read: " << local_points.size();
            LOG( INFO ) << "Reading time: " << read_duration.count() / 1000.0 << " seconds";
        }

        //=======================================================================
        // Scalar Remapping
        //=======================================================================

        // Perform scalar remapping if output file specified
        if( !output_file.empty() )
        {
            LOG( INFO ) << "";
            LOG( INFO ) << "=== Starting Scalar Remapping ===";

            // Determine remapping method from command-line argument
            RemapperFactory::RemapMethod method = RemapperFactory::ALG_DISKAVERAGE;
            if( remap_method == "NEAREST_NEIGHBOR" || remap_method == "NN" )
            {
                method = RemapperFactory::ALG_NEAREST_NEIGHBOR;
            }

            // Create appropriate remapper using factory pattern
            auto remapper = RemapperFactory::create_remapper( method, mb, mesh_set );

            if( remapper )
            {
                // Configure remapping parameters
                ScalarRemapper::RemapConfig remap_config;

                // Note: PointCloudMeshView configuration for self-remapping
                // scenarios ParallelPointCloudReader::PointCloudMeshView
                // target_view (reader.cached_point_cloud(),
                // user_smoothing_area); remap_config.target_point_cloud_view =
                // &target_view;

                remap_config.scalar_var_names  = reader.get_config().scalar_var_names;
                remap_config.reuse_source_mesh = build_target_from_source;
                remap_config.user_search_area  = user_smoothing_area;

                // Add squared field names to the remapping list
                // These are automatically computed from base fields
                for( const auto& field_name : reader.get_config().square_field_names )
                {
                    remap_config.scalar_var_names.push_back( field_name + "_squared" );
                }

                // Configure format detection based on source data
                remap_config.is_usgs_format = reader.is_usgs_format();

                // Configure the remapper with parsed parameters
                MB_CHK_SET_ERR( remapper->configure( remap_config ), "Failed to configure remapper" );

                // Perform the actual scalar remapping from point cloud to mesh
                MB_CHK_SET_ERR( remapper->remap_scalars( local_points ), "Failed to perform scalar remapping" );

                // Write remapped data to MOAB tags on mesh elements
                if( !build_target_from_source )
                {
                    MB_CHK_SET_ERR( remapper->write_to_tags( "" ), "Failed to write remapped data to tags" );
                }

                LOG( INFO ) << "Successfully created remapped tags on mesh elements";
                LOG( INFO ) << "";

                //===================================================================
                // Output Writing
                //===================================================================

                // Serialize remapped results to disk with timing
                {
                    // Determine output format based on file extension and mesh
                    // type
                    bool write_nc_file             = false;
                    std::filesystem::path filePath = output_file;
                    if( !spectral_target && filePath.extension() != ".h5m" )
                    {
                        write_nc_file = true;
                    }

                    start_time = std::chrono::high_resolution_clock::now();

                    if( write_nc_file )
                    {
                        // Write NetCDF output format
                        NetcdfWriteRequest write_request;
                        write_request.scalar_var_names  = reader.get_config().scalar_var_names;
                        write_request.squared_var_names = reader.get_config().square_field_names;
                        write_request.dimension_name    = dimension_variable;
                        write_request.verbose           = verbose;

                        if( !build_target_from_source )
                        {
                            // Write from MOAB entities (mesh-based target)
                            MB_CHK_SET_ERR( NetcdfMeshIO::write_point_scalars_to_file( mb, target_file, output_file,
                                                                                       write_request, entities ),
                                            "Failed to write NetCDF output" );
                        }
                        else
                        {
                            // Write from MeshData (source-based target)
                            const auto& get_mesh_data = remapper->get_mesh_data();
                            MB_CHK_SET_ERR( NetcdfMeshIO::write_point_scalars_to_file( mb, source_file, output_file,
                                                                                       write_request, get_mesh_data ),
                                            "Failed to write NetCDF output" );
                        }
                    }
                    else
                    {
                        // Write H5M output format (mesh with tags)
                        MB_CHK_SET_ERR( mb->write_file( output_file.c_str(), nullptr, nullptr, &mesh_set, 1 ),
                                        "Failed to write remapped mesh" );
                    }

                    // Log output completion
                    LOG( INFO ) << "Saved mesh with remapped data to: " << output_file;

                    // Report writing time
                    auto write_time = std::chrono::duration_cast< std::chrono::milliseconds >(
                        std::chrono::high_resolution_clock::now() - start_time );
                    LOG( INFO ) << "Output written in " << write_time.count() / 1000.0 << " seconds";
                }
            }
            else
            {
                LOG( ERROR ) << "Failed to create remapper";
                return 1;
            }
        }

        //=======================================================================
        // Cleanup and Exit
        //=======================================================================

        // Log successful completion
        LOG( INFO ) << "====================================";
        LOG( INFO ) << "=== mbda completed successfully! ===";
        LOG( INFO ) << "====================================";
    }
    catch( const std::exception& e )
    {
        // Handle any exceptions that escaped the normal error handling
        LOG( FATAL ) << "Exception: " << e.what();
        return 1;
    }

    // Finalize MPI and exit
    MPI_Finalize();
    return 0;
}

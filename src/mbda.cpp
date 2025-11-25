// Suppress warnings from easylogging++ header
#include "easylogging.hpp"

// Let us initialize easylogging++
// Can be in exactly one source file only
INITIALIZE_EASYLOGGINGPP

#include "ParallelPointCloudReader.hpp"
#include "ScalarRemapper.hpp"
#include "MeshIO.hpp"
#include "moab/Core.hpp"
#include "moab/ProgOptions.hpp"

// Standard includes
#include <mpi.h>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <filesystem>

#include <omp.h>

using namespace moab;

// Helper function to parse comma-separated strings
std::vector<std::string> parse_comma_separated(const std::string& input) {
    std::vector<std::string> result;
    if (input.empty()) return result;

    std::stringstream ss(input);
    std::string item;
    while (std::getline(ss, item, ',')) {
        // Trim whitespace
        item.erase(0, item.find_first_not_of(" \t"));
        item.erase(item.find_last_not_of(" \t") + 1);
        if (!item.empty()) {
            result.push_back(item);
        }
    }
    return result;
}

// Source - https://stackoverflow.com/a
// Posted by Moehre2, modified by community. See post 'Timeline' for change history
// Retrieved 2025-11-10, License - CC BY-SA 4.0
int get_num_threads(void) {
    int num_threads = 0;
    #pragma omp parallel reduction(+:num_threads)
    num_threads += 1;
    return num_threads;
}


/**
 * @brief mbda - Maps NetCDF point cloud data to target mesh elements based on
 * the disk-based averaging algorithm that is guaranteed to be monotone.
 *
 * NOTE: OpenMP parallelism is available - set OMP_NUM_THREADS environment variable
 */
int main(int argc, char* argv[]) {

    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Configure easylogging++ (will be reconfigured after parsing args)
    START_EASYLOGGINGPP(argc, argv);

    // Thread-safe logging for OpenMP
    // el::Loggers::addFlag(el::LoggingFlag::DisableApplicationAbortOnFatalLog);
    el::Loggers::addFlag(el::LoggingFlag::MultiLoggerSupport);
    el::Loggers::addFlag(el::LoggingFlag::StrictLogFileSizeCheck);
    el::Loggers::addFlag(el::LoggingFlag::ColoredTerminalOutput);

    try {
        // Parse command-line arguments using MOAB's ProgOptions
        ProgOptions opts;

        // Required arguments
        std::string source_file = "";
        std::string target_file = "";
        std::string output_file = "";

        // Optional variable name overrides
        std::string dimension_variable = "";
        std::string lon_variable = "";
        std::string lat_variable = "";
        std::string area_variable = "";
        std::string fields_string_list = "";
        std::string square_fields_string_list = "";

        // Remapping options
        std::string remap_method = "ALG_DISKAVERAGE";
        bool verbose = false;
        bool spectral_target = false;

        opts.addOpt<std::string>("source", "Source NetCDF point cloud file", &source_file);
        opts.addOpt<std::string>("target", "Target mesh file (nc or H5M)", &target_file);
        opts.addOpt<std::string>("output", "Output mesh file with remapped data (ending with nc or h5m)", &output_file);

        opts.addOpt<std::string>("dof-var", "DoF numbering variable name (bypasses format detection). Default: ncol", &dimension_variable);
        opts.addOpt<std::string>("lon-var", "Longitude variable name (bypasses format detection). Default: lon", &lon_variable);
        opts.addOpt<std::string>("lat-var", "Latitude variable name (bypasses format detection). Default: lat", &lat_variable);
        opts.addOpt<std::string>("area-var", "Area variable name to read and store. Default: area", &area_variable);

        opts.addOpt<std::string>("fields", "Comma-separated field names to remap", &fields_string_list);
        opts.addOpt<std::string>("square-fields", "Comma-separated quadratic field names to remap (stored as: <field>_squared)", &square_fields_string_list);
        opts.addOpt<std::string>("remap-method", "Remapping method: da (ALG_DISKAVERAGE) or nn (ALG_NEAREST_NEIGHBOR). Default: da", &remap_method);

        opts.addOpt<void>("spectral", "Assume that the target mesh requires online spectral element mesh treatment. Default: false", &spectral_target);
        opts.addOpt<void>("verbose,v", "Enable verbose output with timestamps. Default: false", &verbose);

        opts.parseCommandLine(argc, argv);

        // Configure logging format based on verbose flag
        el::Configurations defaultConf;
        defaultConf.setToDefault();

        if (verbose) {
            // Verbose: show level, timestamp, and message
            defaultConf.set(el::Level::Global, el::ConfigurationType::Format,
                            "[%level: %datetime{%H:%m:%s}] %msg");
        } else {
            // Quiet: only show message
            defaultConf.set(el::Level::Global, el::ConfigurationType::Format, "%msg");
        }
        defaultConf.set(el::Level::Global, el::ConfigurationType::ToFile, "false");
        defaultConf.set(el::Level::Global, el::ConfigurationType::ToStandardOutput, "true");

        el::Loggers::reconfigureLogger("default", defaultConf);

        // Validate required arguments
        if (source_file.empty() || target_file.empty()) {
            if (rank == 0) {
                LOG(ERROR) << "Source and target files are required.";
                opts.printHelp();
            }
            MPI_Finalize();
            return 1;
        }

        if (rank == 0) {
            LOG(INFO) << "==================================";
            LOG(INFO) << "=== mbda: Configuration ===";
            LOG(INFO) << "Source file: " << source_file;
            LOG(INFO) << "Target file: " << target_file;
            if (!output_file.empty()) {
                LOG(INFO) << "Output file: " << output_file;
            }
            if (!lon_variable.empty() && !lat_variable.empty()) {
                LOG(INFO) << "Coordinate variable overrides: " << lon_variable << ", " << lat_variable;
            }
            if (!area_variable.empty()) {
                LOG(INFO) << "Area variable: " << area_variable;
            }
            if (!fields_string_list.empty()) {
                LOG(INFO) << "Field overrides: " << fields_string_list;
            }
            if (!square_fields_string_list.empty()) {
                LOG(INFO) << "Square fields: " << square_fields_string_list;
            }
            LOG(INFO) << "Remap method: " << remap_method << " (" << (spectral_target ? "Spectral Target" : "Generic Target") << ")";
            LOG(INFO) << "Number of threads: " << get_num_threads();
            LOG(INFO) << "====================================\n";
        }

        // Initialize MOAB
        Core moab_core;
        Interface* mb = &moab_core;

        // Load target mesh
        EntityHandle mesh_set;
        ErrorCode rval = mb->create_meshset(MESHSET_SET, mesh_set);
        if (MB_SUCCESS != rval) {
            LOG(ERROR) << "Failed to create mesh set";
            MPI_Finalize();
            return 1;
        }

        std::vector<EntityHandle> entities;
        if (spectral_target) {
            // Always use parallel options
            std::string read_opts = "DEBUG_IO=0";
            if( size > 1 )  // If reading in parallel, need to tell it how
                read_opts += ";PARALLEL=READ_PART;PARTITION=PARALLEL_PARTITION;PARALLEL_RESOLVE_SHARED_ENTS;DEBUG_PIO=0";
            // Load the mesh
            MB_CHK_SET_ERR( mb->load_file(target_file.c_str(), &mesh_set, read_opts.c_str()), "Failed to load mesh file");

            // Get all elements in the mesh set
            MB_CHK_SET_ERR( mb->get_entities_by_dimension(mesh_set, 2, entities), "Failed to get elements by dimension");

        } else {
            NetcdfLoadOptions load_opts;
            load_opts.dimension_name = dimension_variable;
            load_opts.lon_var_name   = lon_variable;
            load_opts.lat_var_name   = lat_variable;
            load_opts.area_var_name  = area_variable;
            load_opts.context_label  = "source point-cloud mesh";
            load_opts.verbose        = verbose && (rank == 0);

            MB_CHK_SET_ERR(NetcdfMeshIO::load_point_cloud_from_file(mb, mesh_set, target_file, load_opts, entities),
                           "Failed to load NetCDF mesh file");
        }
        if (rank == 0) {
            LOG(INFO) << "Loaded target mesh: " << target_file;
        }

        // Configure the reader
        ParallelPointCloudReader reader(mb, mesh_set);
        ParallelPointCloudReader::ReadConfig config;
        config.netcdf_filename = source_file;
        config.print_statistics = true;
        config.verbose = verbose;

        // Apply coordinate variable overrides (bypasses format detection)
        config.coord_var_names = {"lon", "lat"};

        // Apply field name overrides (replaces auto-detection)
        if (!fields_string_list.empty()) {
            config.scalar_var_names = parse_comma_separated(fields_string_list);
        }

        // Store square fields list
        if (!square_fields_string_list.empty()) {
            config.square_field_names = parse_comma_separated(square_fields_string_list);
        }

        reader.configure(config);

        // Read points
        auto start_time = std::chrono::high_resolution_clock::now();

        ParallelPointCloudReader::PointData local_points;
        MB_CHK_SET_ERR( reader.read_points(local_points), "Failed to read points");

        if (reader.get_config().print_statistics) {
            auto read_duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time);

            if (rank == 0) {
                LOG(INFO) << "===================================";
                LOG(INFO) << "=== Point Cloud Reading Results ===";
                LOG(INFO) << "Total points in dataset: " << reader.get_point_count();
                LOG(INFO) << "Points read: " << local_points.size();
                LOG(INFO) << "Reading time: " << read_duration.count()/1000.0 << " seconds";
            }
        }

        // Perform scalar remapping if output file specified
        if (!output_file.empty()) {
            if (rank == 0) {
                LOG(INFO) << "";
                LOG(INFO) << "=== Starting Scalar Remapping ===";
            }

            // Determine remapping method
            RemapperFactory::RemapMethod method = RemapperFactory::ALG_DISKAVERAGE;
            if (remap_method == "NEAREST_NEIGHBOR" || remap_method == "NN") {
                method = RemapperFactory::ALG_NEAREST_NEIGHBOR;
            }

            auto remapper = RemapperFactory::create_remapper(method, mb, mesh_set);

            if (remapper) {
                // Configure remapping
                ScalarRemapper::RemapConfig remap_config;
                remap_config.scalar_var_names = reader.get_config().scalar_var_names;

                // Add squared field names to the remapping list
                for (const auto& field_name : reader.get_config().square_field_names) {
                    remap_config.scalar_var_names.push_back(field_name + "_squared");
                }

                remap_config.max_neighbors = 1;
                remap_config.is_usgs_format = reader.is_usgs_format();

                MB_CHK_SET_ERR(remapper->configure(remap_config), "Failed to configure remapper");

                // Perform remapping
                MB_CHK_SET_ERR(remapper->remap_scalars(local_points), "Failed to perform scalar remapping");

                // Write remapped data to MOAB tags
                MB_CHK_SET_ERR(remapper->write_to_tags(""), "Failed to write remapped data to tags");

                if (rank == 0) {
                    LOG(INFO) << "Successfully created remapped tags on mesh elements";
                    LOG(INFO) << "";
                }

                // now let us serialize the results to disk
                {
                    bool write_nc_file = false;
                    std::filesystem::path filePath = output_file;
                    if (!spectral_target && filePath.extension() != ".h5m") {
                        write_nc_file = true;
                    }

                    start_time = std::chrono::high_resolution_clock::now();
                    if (write_nc_file) {
                        NetcdfWriteRequest write_request;
                        write_request.scalar_var_names = reader.get_config().scalar_var_names;
                        write_request.squared_var_names = reader.get_config().square_field_names;
                        write_request.dimension_name = dimension_variable;
                        write_request.verbose = verbose && (rank == 0);

                        MB_CHK_SET_ERR(
                            NetcdfMeshIO::write_point_scalars_to_file(mb, target_file, output_file, write_request, entities),
                            "Failed to write NetCDF output");
                    }
                    else {
                        // Save mesh with remapped data
                        std::string write_opts = "";
                        if (size > 1) {
                            write_opts = "PARALLEL=WRITE_PART";
                        }
                        MB_CHK_SET_ERR(mb->write_file(output_file.c_str(), nullptr, write_opts.c_str(), &mesh_set, 1),
                                    "Failed to write remapped mesh");
                    }

                    if (rank == 0) {
                        LOG(INFO) << "Saved mesh with remapped data to: " << output_file;
                    }

                    auto write_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time);
                    LOG(INFO) << "Output written in " << write_time.count() / 1000.0 << " seconds" ;
                }
            } else {
                LOG(ERROR) << "Failed to create remapper";
                MPI_Finalize();
                return 1;
            }
        }

        if (rank == 0) {
            LOG(INFO) << "====================================";
            LOG(INFO) << "=== mbda completed successfully! ===";
            LOG(INFO) << "====================================";
        }

    } catch (const std::exception& e) {
        LOG(FATAL) << "Exception: " << e.what();
        MPI_Finalize();
        return 1;
    }

    MPI_Finalize();
    return 0;
}

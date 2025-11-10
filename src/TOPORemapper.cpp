// Suppress warnings from easylogging++ header
#include "easylogging.hpp"

// Initialize easylogging++ - MUST be in exactly one source file
INITIALIZE_EASYLOGGINGPP

#include "ParallelPointCloudReader.hpp"
#include "ScalarRemapper.hpp"
#include "moab/Core.hpp"
#include "moab/ProgOptions.hpp"
#include <mpi.h>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <algorithm>

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

/**
 * @brief TOPORemapper - Maps NetCDF point cloud data to mesh elements
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
        std::string lon_var = "";
        std::string lat_var = "";
        std::string area_var = "";
        std::string fields_str = "";
        std::string square_fields_str = "";

        // Remapping options
        std::string remap_method = "PC_SPECTRAL";
        int spectral_order = 4;
        bool continuous_gll = true;
        bool apply_bubble = true;
        bool verbose = false;

        opts.addOpt<std::string>("source,s", "Source NetCDF point cloud file", &source_file);
        opts.addOpt<std::string>("target,t", "Target mesh file (H5M)", &target_file);
        opts.addOpt<std::string>("output,o", "Output mesh file with remapped data", &output_file);

        opts.addOpt<std::string>("lon-var,", "Longitude variable name (bypasses format detection)", &lon_var);
        opts.addOpt<std::string>("lat-var,", "Latitude variable name (bypasses format detection)", &lat_var);
        opts.addOpt<std::string>("area-var,", "Area variable name to read and store", &area_var);
        opts.addOpt<std::string>("fields,f", "Comma-separated field names to remap (replaces auto-detection)", &fields_str);
        opts.addOpt<std::string>("square-fields,", "Comma-separated fields to compute squares for (<field>_squared)", &square_fields_str);

        opts.addOpt<std::string>("remap-method,m", "Remapping method: PC_SPECTRAL or NEAREST_NEIGHBOR (default: PC_SPECTRAL)", &remap_method);
        opts.addOpt<int>("spectral-order,p", "Spectral element order for PC_SPECTRAL method (default: 4)", &spectral_order);
        opts.addOpt<void>("no-continuous-gll,", "Disable continuous GLL nodes", &continuous_gll);
        opts.addOpt<void>("no-bubble-correction,", "Disable bubble correction", &apply_bubble);
        opts.addOpt<void>("verbose,v", "Enable verbose output with timestamps", &verbose);

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
            LOG(INFO) << "";
            LOG(INFO) << "=== TOPORemapper Configuration ===";
            LOG(INFO) << "Source file: " << source_file;
            LOG(INFO) << "Target file: " << target_file;
            if (!output_file.empty()) {
                LOG(INFO) << "Output file: " << output_file;
            }
            if (!lon_var.empty() && !lat_var.empty()) {
                LOG(INFO) << "Coordinate variable overrides: " << lon_var << ", " << lat_var;
            }
            if (!area_var.empty()) {
                LOG(INFO) << "Area variable: " << area_var;
            }
            if (!fields_str.empty()) {
                LOG(INFO) << "Field overrides: " << fields_str;
            }
            if (!square_fields_str.empty()) {
                LOG(INFO) << "Square fields: " << square_fields_str;
            }
            LOG(INFO) << "Remap method: " << remap_method;
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

        // Load mesh with parallel options
        std::string read_opts = "DEBUG_IO=0";
        if( size > 1 )  // If reading in parallel, need to tell it how
            read_opts += ";PARALLEL=READ_PART;PARTITION=PARALLEL_PARTITION;PARALLEL_RESOLVE_SHARED_ENTS;DEBUG_PIO=0";
        MB_CHK_SET_ERR( mb->load_file(target_file.c_str(), &mesh_set, read_opts.c_str()), "Failed to load mesh file");

        if (rank == 0) {
            LOG(INFO) << "Loaded target mesh: " << target_file;
        }

        // Configure the reader
        ParallelPointCloudReader reader(mb, mesh_set);
        ParallelPointCloudReader::ReadConfig config;
        config.netcdf_filename = source_file;
        config.print_statistics = true;
        config.verbose = true;

        // Apply coordinate variable overrides (bypasses format detection)
        if (!lon_var.empty() && !lat_var.empty()) {
            config.coord_var_names = {lon_var, lat_var};
            config.bypass_format_detection = true;
        }

        // Apply field name overrides (replaces auto-detection)
        if (!fields_str.empty()) {
            config.scalar_var_names = parse_comma_separated(fields_str);
        }

        // Store area variable name if specified
        if (!area_var.empty()) {
            config.area_var_name = area_var;
        }

        // Store square fields list
        if (!square_fields_str.empty()) {
            config.square_field_names = parse_comma_separated(square_fields_str);
        }

        reader.configure(config);

        // Read points
        auto start_time = std::chrono::high_resolution_clock::now();

        ParallelPointCloudReader::PointData local_points;
        MB_CHK_SET_ERR( reader.read_points(local_points), "Failed to read points");

        if (reader.get_config().print_statistics) {
            auto read_time = std::chrono::high_resolution_clock::now();
            auto read_duration = std::chrono::duration_cast<std::chrono::milliseconds>(read_time - start_time);

            if (rank == 0) {
                LOG(INFO) << "";
                LOG(INFO) << "=== Point Cloud Reading Results ===";
                LOG(INFO) << "Total points in dataset: " << reader.get_point_count();
                LOG(INFO) << "Points read: " << local_points.size();
                LOG(INFO) << "Reading time: " << read_duration.count() << " ms";
            }
        }

        // Perform scalar remapping if output file specified
        if (!output_file.empty()) {
            if (rank == 0) {
                LOG(INFO) << "";
                LOG(INFO) << "=== Starting Scalar Remapping ===";
            }

            // Determine remapping method
            RemapperFactory::RemapMethod method = RemapperFactory::PC_AVERAGED_SPECTRAL_PROJECTION;
            if (remap_method == "NEAREST_NEIGHBOR" || remap_method == "NN") {
                method = RemapperFactory::NEAREST_NEIGHBOR;
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
                remap_config.spectral_order = spectral_order;
                remap_config.continuous_gll = continuous_gll;
                remap_config.apply_bubble_correction = apply_bubble;
                remap_config.use_element_centroids = true;
                remap_config.is_usgs_format = reader.is_usgs_format();

                MB_CHK_SET_ERR(remapper->configure(remap_config), "Failed to configure remapper");

                // Perform remapping
                MB_CHK_SET_ERR(remapper->remap_scalars(local_points), "Failed to perform scalar remapping");

                // Write remapped data to MOAB tags
                MB_CHK_SET_ERR(remapper->write_to_tags(""), "Failed to write remapped data to tags");

                if (rank == 0) {
                    LOG(INFO) << "Successfully created remapped tags on mesh elements";
                }

                // Save mesh with remapped data
                std::string write_opts = "";
                if (size > 1) {
                    write_opts = "PARALLEL=WRITE_PART";
                }
                MB_CHK_SET_ERR(mb->write_file(output_file.c_str(), nullptr, write_opts.c_str(), &mesh_set, 1),
                               "Failed to write remapped mesh");

                if (rank == 0) {
                    LOG(INFO) << "Saved mesh with remapped data to: " << output_file;
                }
            } else {
                LOG(ERROR) << "Failed to create remapper";
                MPI_Finalize();
                return 1;
            }
        }

        if (rank == 0) {
            LOG(INFO) << "";
            LOG(INFO) << "TOPORemapper completed successfully!";
        }

    } catch (const std::exception& e) {
        LOG(FATAL) << "Exception: " << e.what();
        MPI_Finalize();
        return 1;
    }

    MPI_Finalize();
    return 0;
}

#include "ParallelPointCloudReader.hpp"
#include "ScalarRemapper.hpp"
#include "moab/Core.hpp"
#include "moab/ParallelComm.hpp"
#include <mpi.h>
#include <iostream>
#include <chrono>
#include <iomanip>

using namespace moab;

/**
 * @brief Example demonstrating efficient parallel point cloud reading with MOAB mesh decomposition
 */
int main(int argc, char* argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    try {
        // Initialize MOAB
        Core moab_core;
        Interface* mb = &moab_core;

        // Create parallel communicator
        ParallelComm pcomm(mb, MPI_COMM_WORLD);

        // Load mesh file
        std::string mesh_filename = "/Users/mahadevan/Code/sigma/moab-pristine2/MeshFiles/unittest/quad_1000.h5m";
        if (argc > 1) mesh_filename = argv[1];

        EntityHandle mesh_set;
        ErrorCode rval = mb->create_meshset(MESHSET_SET, mesh_set);
        if (MB_SUCCESS != rval) {
            std::cerr << "Failed to create mesh set" << std::endl;
            MPI_Finalize();
            return 1;
        }

        // Load mesh with parallel options
        std::string read_opts;
        if( pcomm.size() > 1 )  // If reading in parallel, need to tell it how
            read_opts = "PARALLEL=READ_PART;PARTITION=PARALLEL_PARTITION;PARALLEL_RESOLVE_SHARED_ENTS;"
                      "DEBUG_IO=0;DEBUG_PIO=0";
        MB_CHK_SET_ERR( mb->load_file(mesh_filename.c_str(), &mesh_set, read_opts.c_str()), "Failed to load mesh file");

        if (rank == 0) {
            std::cout << "Loaded mesh file: " << mesh_filename << " on " << size << " processes" << std::endl;
        }

        // Configure the reader
        ParallelPointCloudReader reader(mb, &pcomm, mesh_set);
        ParallelPointCloudReader::ReadConfig config;
        std::string netcdf_filename = "pointcloud.nc";
        if (argc > 2) netcdf_filename = argv[2];
        config.netcdf_filename = netcdf_filename;
        // config.coord_var_names = {"xc", "yc"};  // Coordinate variables from NetCDF file
        // config.scalar_var_names = {"mask", "area", "frac"};  // Available scalar variables
        config.buffer_factor = 0.0;  // 1% buffer around mesh bounding box
        // config.chunk_size = 25000000;   // Process 25M points at a time
        config.use_collective_io = true;
        config.convert_lonlat_to_xyz = true;  // Convert from lon/lat degrees to Cartesian
        config.auto_detect_format = true;     // Auto-detect USGS vs standard format
        config.use_distributed_reading = true;       // Enable distributed reading approach
        config.print_statistics = true;
        config.verbose = true;

        reader.configure(config);

        // Read points
        auto start_time = std::chrono::high_resolution_clock::now();

        ParallelPointCloudReader::PointData local_points;
        MB_CHK_SET_ERR( reader.read_points(local_points), "Failed to read points");

        if (reader.get_config().print_statistics) {
            // Get total point count
            if (rank == 0) {
                std::cout << "\n=== Total points in dataset: " << reader.get_point_count() << std::endl;
            }

            auto read_time = std::chrono::high_resolution_clock::now();
            auto read_duration = std::chrono::duration_cast<std::chrono::milliseconds>(read_time - start_time);

            // Gather statistics
            size_t local_count = local_points.size();
            std::vector<size_t> all_counts(size);
            MPI_Gather(&local_count, 1, MPI_UNSIGNED_LONG, all_counts.data(), 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

            size_t total_read = 0;
            if (rank == 0) {
                size_t min_points = SIZE_MAX, max_points = 0;
                for (int i = 0; i < size; ++i) {
                    total_read += all_counts[i];
                    min_points = std::min(min_points, all_counts[i]);
                    max_points = std::max(max_points, all_counts[i]);
                }

                double load_balance = (max_points > 0) ? static_cast<double>(min_points) / max_points : 1.0;

                std::cout << "\n=== Point Cloud Reading Results ===" << std::endl;
                std::cout << "Total points read: " << total_read << std::endl;
                std::cout << "Reading time: " << read_duration.count() << " ms" << std::endl;
                std::cout << "Points per rank - Min: " << min_points << ", Max: " << max_points
                        << ", Avg: " << total_read / size << std::endl;
                std::cout << "Load balance factor: " << std::fixed << std::setprecision(3) << load_balance << std::endl;

                // Detailed per-rank statistics
                std::cout << "\nPer-rank point counts:" << std::endl;
                for (int i = 0; i < size; ++i) {
                    std::cout << "  Rank " << i << ": " << all_counts[i] << " points" << std::endl;
                }
            }
        }

        bool remap_data = false;
        std::string remap_output_filename = "";
        // let us check all other input options.
        for (int iarg = 3; iarg < argc; ++iarg)
        {
            if (std::string(argv[iarg]) == "--remap") {
                remap_data = true;
                if (iarg + 1 < argc) {
                    remap_output_filename = argv[iarg + 1];
                    ++iarg; // skip the next one.
                }
            }
        }

        // Perform scalar remapping from point cloud to mesh elements
        if (remap_data) {
            if (rank == 0) {
                std::cout << "\n=== Starting Scalar Remapping ===" << std::endl;
            }

            // Create remapper (PC averaged spectral projection by default)
            auto remapper = RemapperFactory::create_remapper(
                RemapperFactory::PC_AVERAGED_SPECTRAL_PROJECTION, mb, &pcomm, mesh_set);

            if (remapper) {
                // Configure remapping
                ScalarRemapper::RemapConfig remap_config;
                // if domain file, use the following for testing.
                // remap_config.scalar_var_names = {"mask", "area", "frac"};
                // if using USGS-rawdata, then use the following
                remap_config.scalar_var_names = reader.get_config().scalar_var_names; // {"htopo", "landfract"};
                remap_config.search_radius = 0.0;  // No search radius limit
                remap_config.max_neighbors = 1;    // Nearest neighbor only
                remap_config.spectral_order = 4;   // Spectral element order
                remap_config.continuous_gll = true; // Use continuous GLL nodes
                remap_config.apply_bubble_correction = true; // No bubble correction for now
                remap_config.use_element_centroids = true;
                remap_config.is_usgs_format = reader.is_usgs_format();

                MB_CHK_SET_ERR(remapper->configure(remap_config), "Failed to configure remapper");

                // Perform remapping
                MB_CHK_SET_ERR(remapper->remap_scalars(local_points), "Failed to perform scalar remapping");

                // ScalarRemapper::MeshData& mesh_data = remapper->get_mesh_data();
                // Write remapped data to MOAB tags
                MB_CHK_SET_ERR(remapper->write_to_tags("remapped_"), "Failed to write remapped data to tags");
                if (rank == 0) {
                    std::cout << "Successfully created remapped tags on mesh elements" << std::endl;
                }

                // Save mesh with remapped data
                if (!remap_output_filename.empty()) {
                    std::string write_opts = "PARALLEL=WRITE_PART";
                    MB_CHK_SET_ERR(mb->write_file(remap_output_filename.c_str(), nullptr, write_opts.c_str(), &mesh_set, 1), "Failed to write remapped mesh");
                    if (rank == 0) {
                        std::cout << "Saved mesh with remapped data to: " << remap_output_filename << std::endl;
                    }
                }
            } else {
                std::cerr << "Rank " << rank << ": Failed to create remapper" << std::endl;
            }
        }

        if (rank == 0) {
            std::cout << "\nRemapper processing completed successfully!" << std::endl;
            if (remap_output_filename.empty()) {
                std::cout << "Usage: " << argv[0] << " <mesh.h5m> <pointcloud.nc> [--remap] [output_mesh.h5m]" << std::endl;
                std::cout << "  --remap: Enable scalar remapping from point cloud to mesh" << std::endl;
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "Rank " << rank << ": Exception: " << e.what() << std::endl;
        MPI_Finalize();
        return 1;
    }

    MPI_Finalize();
    return 0;
}

// Additional utility functions for demonstration

// /**
//  * @brief Create a simple test NetCDF file for demonstration
//  */
// void create_test_netcdf_file(const std::string& /* filename */, size_t /* num_points */) {
//     // This would create a test NetCDF file with synthetic point cloud data
//     // Implementation omitted for brevity but would use NetCDF-C++ API
// }

// /**
//  * @brief Validate point cloud data integrity
//  */
// bool validate_point_data(const ParallelPointCloudReader::PointData& points) {
//     // Check for NaN values, coordinate bounds, etc.
//     for (const auto& coord : points.coordinates) {
//         for (int i = 0; i < 3; ++i) {
//             if (std::isnan(coord[i]) || std::isinf(coord[i])) {
//                 return false;
//             }
//         }
//     }

//     // Validate scalar variables
//     for (std::unordered_map<std::string, std::vector<double>>::const_iterator it = points.scalar_variables.begin();
//          it != points.scalar_variables.end(); ++it) {
//         if (it->second.size() != points.coordinates.size()) {
//             return false;
//         }
//         for (size_t i = 0; i < it->second.size(); ++i) {
//             if (std::isnan(it->second[i]) || std::isinf(it->second[i])) {
//                 return false;
//             }
//         }
//     }

//     // Validate vector variables
//     for (auto it = points.vector_variables.begin();
//          it != points.vector_variables.end(); ++it) {
//         if (it->second.size() != points.coordinates.size()) {
//             return false;
//         }
//         for (size_t i = 0; i < it->second.size(); ++i) {
//             for (int j = 0; j < 3; ++j) {
//                 if (std::isnan(it->second[i*3+j]) || std::isinf(it->second[i*3+j])) {
//                     return false;
//                 }
//             }
//         }
//     }

//     return true;
// }

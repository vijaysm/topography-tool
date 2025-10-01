#include "ParallelPointCloudReader.hpp"
#include "ScalarRemapper.hpp"
#include "moab/Core.hpp"
#include <mpi.h>
#include <iostream>
#include <chrono>
#include <iomanip>

using namespace moab;

/**
 * @brief Example demonstrating efficient point cloud reading with MOAB mesh
 * NOTE: OpenMP parallelism is available - set OMP_NUM_THREADS environment variable
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
        std::string read_opts = "DEBUG_IO=0";
        if( size > 1 )  // If reading in parallel, need to tell it how
            read_opts += "PARALLEL=READ_PART;PARTITION=PARALLEL_PARTITION;PARALLEL_RESOLVE_SHARED_ENTS;DEBUG_PIO=0;";
        MB_CHK_SET_ERR( mb->load_file(mesh_filename.c_str(), &mesh_set, read_opts.c_str()), "Failed to load mesh file");

        std::cout << "Loaded mesh file: " << mesh_filename << std::endl;

        // Configure the reader
        ParallelPointCloudReader reader(mb, mesh_set);
        ParallelPointCloudReader::ReadConfig config;
        std::string netcdf_filename = "pointcloud.nc";
        if (argc > 2) netcdf_filename = argv[2];
        config.netcdf_filename = netcdf_filename;
        config.print_statistics = true;
        config.verbose = true;

        reader.configure(config);

        // Read points
        auto start_time = std::chrono::high_resolution_clock::now();

        ParallelPointCloudReader::PointData local_points;
        MB_CHK_SET_ERR( reader.read_points(local_points), "Failed to read points");

        if (reader.get_config().print_statistics) {
            auto read_time = std::chrono::high_resolution_clock::now();
            auto read_duration = std::chrono::duration_cast<std::chrono::milliseconds>(read_time - start_time);

            std::cout << "\n=== Point Cloud Reading Results ===" << std::endl;
            std::cout << "Total points in dataset: " << reader.get_point_count() << std::endl;
            std::cout << "Points read: " << local_points.size() << std::endl;
            std::cout << "Reading time: " << read_duration.count() << " ms" << std::endl;
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
            std::cout << "\n=== Starting Scalar Remapping ===" << std::endl;

            // Create remapper (PC averaged spectral projection by default)
            auto remapper = RemapperFactory::create_remapper(
                RemapperFactory::PC_AVERAGED_SPECTRAL_PROJECTION, mb, mesh_set);

            if (remapper) {
                // Configure remapping
                ScalarRemapper::RemapConfig remap_config;
                // if domain file, use the following for testing.
                // remap_config.scalar_var_names = {"mask", "area", "frac"};
                // if using USGS-rawdata, then use the following
                remap_config.scalar_var_names = reader.get_config().scalar_var_names; // {"htopo", "landfract"};
                remap_config.max_neighbors = 1;    // Nearest neighbor only
                remap_config.spectral_order = 4;   // Spectral element order
                remap_config.continuous_gll = true; // Use continuous GLL nodes
                remap_config.apply_bubble_correction = true;
                remap_config.use_element_centroids = true;
                remap_config.is_usgs_format = reader.is_usgs_format();

                MB_CHK_SET_ERR(remapper->configure(remap_config), "Failed to configure remapper");

                // Perform remapping
                MB_CHK_SET_ERR(remapper->remap_scalars(local_points), "Failed to perform scalar remapping");

                // Write remapped data to MOAB tags
                MB_CHK_SET_ERR(remapper->write_to_tags("remapped_"), "Failed to write remapped data to tags");
                std::cout << "Successfully created remapped tags on mesh elements" << std::endl;

                // Save mesh with remapped data
                if (!remap_output_filename.empty()) {
                    std::string write_opts = "PARALLEL=WRITE_PART";
                    MB_CHK_SET_ERR(mb->write_file(remap_output_filename.c_str(), nullptr, write_opts.c_str(), &mesh_set, 1), "Failed to write remapped mesh");
                    std::cout << "Saved mesh with remapped data to: " << remap_output_filename << std::endl;
                }
            } else {
                std::cerr << "Failed to create remapper" << std::endl;
            }
        }

        std::cout << "\nRemapper processing completed successfully!" << std::endl;
        if (remap_output_filename.empty()) {
            std::cout << "Usage: " << argv[0] << " <mesh.h5m> <pointcloud.nc> [--remap] [output_mesh.h5m]" << std::endl;
            std::cout << "  --remap: Enable scalar remapping from point cloud to mesh" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << " Exception: " << e.what() << std::endl;
        MPI_Finalize();
        return 1;
    }

    MPI_Finalize();
    return 0;
}

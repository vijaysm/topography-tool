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

namespace moab
{

namespace
{
std::string resolve_or_default( const std::string& value, const std::string& fallback )
{
    return value.empty() ? fallback : value;
}

template <typename T>
std::vector< float > convert_to_float( const std::vector< T >& input )
{
    std::vector< float > output( input.size() );
    std::transform( input.begin(), input.end(), output.begin(), []( T val ) { return static_cast< float >( val ); } );
    return output;
}

std::vector< float > convert_to_float( const std::vector< float >& input )
{
    return input;
}

}  // namespace

ErrorCode NetcdfMeshIO::load_point_cloud_from_file( Interface* mb,
                                                     EntityHandle mesh_set,
                                                     const std::string& filename,
                                                     const NetcdfLoadOptions& options,
                                                     std::vector< EntityHandle >& entities_out )
{
    if( nullptr == mb ) MB_SET_ERR( MB_FAILURE, "Invalid MOAB interface" );

    const std::string dim_name  = resolve_or_default( options.dimension_name, "ncol" );
    const std::string lon_name  = resolve_or_default( options.lon_var_name, "lon" );
    const std::string lat_name  = resolve_or_default( options.lat_var_name, "lat" );
    const std::string area_name = resolve_or_default( options.area_var_name, "area" );

    try
    {
        PnetCDF::NcmpiFile nc( MPI_COMM_WORLD, filename.c_str(), PnetCDF::NcmpiFile::read );

        PnetCDF::NcmpiDim dim = nc.getDim( dim_name );
        MPI_Offset ncol       = dim.getSize();
        if( ncol <= 0 ) MB_SET_ERR( MB_FAILURE, "Dimension " << dim_name << " has no entries" );

        std::vector< double > lon( ncol ), lat( ncol ), area( ncol );

        PnetCDF::NcmpiVar lon_var  = nc.getVar( lon_name );
        PnetCDF::NcmpiVar lat_var  = nc.getVar( lat_name );
        PnetCDF::NcmpiVar area_var = nc.getVar( area_name );

        lon_var.getVar_all( lon.data() );
        lat_var.getVar_all( lat.data() );
        area_var.getVar_all( area.data() );

        ncmpi_close( nc.getId() );

        Tag area_tag = nullptr;
        MB_CHK_SET_ERR( mb->tag_get_handle( "area", 1, MB_TYPE_DOUBLE, area_tag, MB_TAG_DENSE | MB_TAG_CREAT ),
                        "Failed to create area tag" );

        entities_out.clear();
        entities_out.resize( ncol );

        for( MPI_Offset idx = 0; idx < ncol; ++idx )
        {
            std::array< double, 3 > coords { 0.0, 0.0, 0.0 };
            MB_CHK_SET_ERR( RLLtoXYZ_Deg( lon[idx], lat[idx], coords ), "Failed to convert lon/lat to Cartesian" );
            MB_CHK_SET_ERR( mb->create_vertex( coords.data(), entities_out[idx] ), "Failed to create vertex" );
        }

        MB_CHK_SET_ERR( mb->tag_set_data( area_tag, entities_out.data(), entities_out.size(), area.data() ),
                        "Failed to assign area tag" );
        MB_CHK_SET_ERR( mb->add_entities( mesh_set, entities_out.data(), entities_out.size() ),
                        "Failed to add vertices to mesh set" );

        if( options.verbose )
        {
            LOG( INFO ) << "Loaded " << ncol << " points from NetCDF file " << filename
                        << ( options.context_label.empty() ? std::string() : " (" + options.context_label + ")" );
        }
    }
    catch( const PnetCDF::exceptions::NcmpiException& e )
    {
        MB_SET_ERR( MB_FAILURE, "NetCDF error while loading " << filename << ": " << e.what() );
    }

    return MB_SUCCESS;
}

namespace
{
ErrorCode fetch_tag_as_float( Interface* mb,
                              Tag tag,
                              const std::vector< EntityHandle >& entities,
                              std::vector< float >& values )
{
    DataType type;
    MB_CHK_ERR( mb->tag_get_data_type( tag, type ) );

    const size_t count = entities.size();

    switch( type )
    {
        case MB_TYPE_DOUBLE:
        {
            std::vector< double > buffer( count );
            MB_CHK_ERR( mb->tag_get_data( tag, entities.data(), count, buffer.data() ) );
            values = convert_to_float( buffer );
            return MB_SUCCESS;
        }
        case MB_TYPE_INTEGER:
        {
            std::vector< int > buffer( count );
            MB_CHK_ERR( mb->tag_get_data( tag, entities.data(), count, buffer.data() ) );
            values = convert_to_float( buffer );
            return MB_SUCCESS;
        }
        default:
            MB_SET_ERR( MB_FAILURE, "Unsupported tag data type for NetCDF write" );
    }
}
}  // namespace


template<typename T>
static ErrorCode write_variable(const PnetCDF::NcmpiVar& var, const ScalarRemapper::MeshData& mesh_data ) {

    // 2D variable - handle USGS format with spatial chunking
    std::vector<PnetCDF::NcmpiDim> dims = var.getDims();
    size_t total_size = 1;
    for (const auto& dim : dims) {
        total_size *= dim.getSize();
        LOG(INFO) << "Dimension " << dim.getName() << " has size " << dim.getSize();
    }

    // Check if it's a double variable
    auto var_it = mesh_data.d_scalar_fields.find(var.getName());
    if (var_it != mesh_data.d_scalar_fields.end()) {
        const auto& values = var_it->second;

        if (dims.size() == 2)
        {
            size_t dim0_size = dims[0].getSize();
            size_t dim1_size = dims[1].getSize();
            const size_t MAX_ELEMENTS_PER_CHUNK = 250*1000*1000; // 100M elements, well below INT_MAX
            size_t current_start = 0;
            size_t lat_chunk_size = MAX_ELEMENTS_PER_CHUNK / dim1_size;
            std::vector<MPI_Offset> start, read_count;
            for (size_t lat_offset = 0; lat_offset < dim0_size; lat_offset += lat_chunk_size) {
                size_t current_lat_count = std::min(lat_chunk_size, dim0_size - lat_offset);
                size_t chunk_elements = current_lat_count * dim1_size;
                LOG(INFO) << "\tWriting latitude chunk from " << lat_offset << " to " << lat_offset + current_lat_count << "." ;

                std::vector<T> chunk_buffer(chunk_elements);
                std::transform(values.begin() + current_start, values.begin() + current_start + chunk_elements,
                                chunk_buffer.begin(), [](double d) { return static_cast<T>(d); });
                start = {static_cast<MPI_Offset>(lat_offset), static_cast<MPI_Offset>(0)};
                read_count = {static_cast<MPI_Offset>(current_lat_count), static_cast<MPI_Offset>(dims[1].getSize())};
                var.putVar(start, read_count, chunk_buffer.data());
                current_start += chunk_elements;
            }
            assert(current_start == total_size);
        }
        else {
            std::vector< T > castValues(values.size());
            std::transform(values.begin(), values.end(), castValues.begin(), [](double d) { return static_cast<T>(d); });
            var.putVar( castValues.data() );
        }
    } else {
        // Check if it's an integer variable
        auto ivar_it = mesh_data.i_scalar_fields.find(var.getName());
        if (ivar_it != mesh_data.i_scalar_fields.end()) {
            const auto& values = ivar_it->second;

            if (dims.size() == 2) {
                size_t dim0_size = dims[0].getSize();
                size_t dim1_size = dims[1].getSize();
                const size_t MAX_ELEMENTS_PER_CHUNK = 250*1000*1000; // 100M elements, well below INT_MAX
                size_t current_start = 0;
                size_t lat_chunk_size = MAX_ELEMENTS_PER_CHUNK / dim1_size;
                std::vector<MPI_Offset> start, read_count;
                for (size_t lat_offset = 0; lat_offset < dim0_size; lat_offset += lat_chunk_size) {
                    size_t current_lat_count = std::min(lat_chunk_size, dim0_size - lat_offset);
                    size_t chunk_elements = current_lat_count * dim1_size;
                    LOG(INFO) << "\tWriting latitude chunk from " << lat_offset << " to " << lat_offset + current_lat_count << "." ;

                    start = {static_cast<MPI_Offset>(lat_offset), static_cast<MPI_Offset>(0)};
                    read_count = {static_cast<MPI_Offset>(current_lat_count), static_cast<MPI_Offset>(dims[1].getSize())};
                    if constexpr (std::is_same_v<T, int>) {
                      var.putVar(start, read_count, values.data() + current_start);
                    }
                    else {
                      std::vector<T> chunk_buffer(chunk_elements);
                      std::transform(values.begin() + current_start, values.begin() + current_start + chunk_elements,
                                      chunk_buffer.begin(), [](int i) { return static_cast<T>(i); });
                      var.putVar(start, read_count, chunk_buffer.data());
                    }
                    current_start += chunk_elements;
                }
                assert(current_start == total_size);
            }
            else {
                std::vector< T > castValues(values.size());
                std::transform(values.begin(), values.end(), castValues.begin(), [](int i) { return static_cast<T>(i); });
                var.putVar( castValues.data() );
            }
        }
    }

    return MB_SUCCESS;
};


ErrorCode NetcdfMeshIO::write_point_scalars_to_file( Interface* mb,
                                                      const std::string& template_file,
                                                      const std::string& output_file,
                                                      const NetcdfWriteRequest& request,
                                                      const ScalarRemapper::MeshData& mesh_data )
{
    if( nullptr == mb ) MB_SET_ERR( MB_FAILURE, "Invalid MOAB interface" );
    if( !mesh_data.d_scalar_fields.size() && !mesh_data.i_scalar_fields.size() ) MB_SET_ERR( MB_FAILURE, "No entities provided for NetCDF write" );

    const std::string dim_name = resolve_or_default( request.dimension_name, "ncol" );

    try
    {
        std::filesystem::copy_file( template_file,
                                    output_file,
                                    std::filesystem::copy_options::overwrite_existing );

// #pragma omp single
        {
            PnetCDF::NcmpiFile out( MPI_COMM_WORLD, output_file.c_str(), PnetCDF::NcmpiFile::write, PnetCDF::NcmpiFile::classic5 );

            // Begin the independent data mode
            int status = ncmpi_begin_indep_data(out.getId());
            if (status != NC_NOERR) {
                throw std::runtime_error("Failed to begin independent data mode for NetCDF write");
            }

            auto vars = out.getVars();
            for( const auto& name : request.scalar_var_names )
            {
                for (const auto& ncvar : vars) {
                    if (ncvar.first == name) {
                        const auto& variable = ncvar.second;
                        if (variable.getType() == PnetCDF::ncmpiFloat) {
                            MB_CHK_ERR( write_variable<float>( variable, mesh_data ) );
                        } else if (variable.getType() == PnetCDF::ncmpiDouble) {
                            MB_CHK_ERR( write_variable<double>( variable, mesh_data ) );
                        } else if (variable.getType() == PnetCDF::ncmpiInt) {
                            MB_CHK_ERR( write_variable<int>( variable, mesh_data ) );
                        } else if (variable.getType() == PnetCDF::ncmpiShort) {
                            MB_CHK_ERR( write_variable<short>( variable, mesh_data ) );
                        } else if (variable.getType() == PnetCDF::ncmpiByte) {
                            MB_CHK_ERR( write_variable<signed char>( variable, mesh_data ) );
                        } else {
                            throw std::logic_error("Unsupported variable type for NetCDF write");
                        }

                        if( request.verbose )
                        {
                            LOG( INFO ) << "Wrote NetCDF variable " << ncvar.first << " to " << output_file;
                        }
                    }
                }
            }

            // End independent data mode if you are done with independent operations
            status = ncmpi_end_indep_data(out.getId());
            if (status != NC_NOERR) {
                throw std::runtime_error("Failed to end independent data mode for NetCDF write");
            }

            // for( const auto& base_name : request.squared_var_names )
            // {
            //     const std::string var_name = base_name + "_squared";
            //     MB_CHK_ERR( write_variable( var_name ) );
            // }

            ncmpi_close( out.getId() );
        }
    }
    catch( const std::filesystem::filesystem_error& e )
    {
        MB_SET_ERR( MB_FAILURE, "File copy error for NetCDF output: " << e.what() );
    }
    catch( const PnetCDF::exceptions::NcmpiException& e )
    {
        MB_SET_ERR( MB_FAILURE, "NetCDF error while writing " << output_file << ": " << e.what() );
    }
    catch (const std::exception& e) {
        MB_SET_ERR( MB_FAILURE, "Exception while writing " << output_file << ": " << e.what() );
    }

    return MB_SUCCESS;

// // #pragma omp single
//     {
//         try
//         {
//             std::filesystem::copy_file( template_file,
//                                         output_file,
//                                         std::filesystem::copy_options::overwrite_existing );

//             PnetCDF::NcmpiFile out( MPI_COMM_WORLD, output_file.c_str(), PnetCDF::NcmpiFile::write, PnetCDF::NcmpiFile::classic5 );

//             // Begin the independent data mode
//             int status = ncmpi_begin_indep_data(out.getId());
//             if (status != NC_NOERR) {
//                 throw std::runtime_error("Failed to begin independent data mode for NetCDF write");
//             }

//             auto vars = out.getVars();
//             for( const auto& name : request.scalar_var_names )
//             {
//                 for (const auto& ncvar : vars) {
//                     if (ncvar.first == name) {
//                         const auto& variable = ncvar.second;
//                         if (variable.getType() == PnetCDF::ncmpiFloat) {
//                             MB_CHK_ERR( write_variable<float>( variable, mesh_data ) );
//                         } else if (variable.getType() == PnetCDF::ncmpiDouble) {
//                             MB_CHK_ERR( write_variable<double>( variable, mesh_data ) );
//                         } else if (variable.getType() == PnetCDF::ncmpiInt) {
//                             MB_CHK_ERR( write_variable<int>( variable, mesh_data ) );
//                         } else if (variable.getType() == PnetCDF::ncmpiShort) {
//                             MB_CHK_ERR( write_variable<short>( variable, mesh_data ) );
//                         } else if (variable.getType() == PnetCDF::ncmpiByte) {
//                             MB_CHK_ERR( write_variable<signed char>( variable, mesh_data ) );
//                         } else {
//                             throw std::logic_error("Unsupported variable type for NetCDF write");
//                         }

//                         if( request.verbose )
//                         {
//                             LOG( INFO ) << "Wrote NetCDF variable " << ncvar.first << " to " << output_file;
//                         }
//                     }
//                 }
//             }

//             // End independent data mode if you are done with independent operations
//             status = ncmpi_end_indep_data(out.getId());
//             if (status != NC_NOERR) {
//                 throw std::runtime_error("Failed to end independent data mode for NetCDF write");
//             }

//             // for( const auto& base_name : request.squared_var_names )
//             // {
//             //     const std::string var_name = base_name + "_squared";
//             //     MB_CHK_ERR( write_variable( var_name ) );
//             // }

//             ncmpi_close( out.getId() );
//         }
//         catch( const std::filesystem::filesystem_error& e )
//         {
//             MB_SET_ERR( MB_FAILURE, "File copy error for NetCDF output: " << e.what() );
//         }
//         catch( const PnetCDF::exceptions::NcmpiException& e )
//         {
//             MB_SET_ERR( MB_FAILURE, "NetCDF error while writing " << output_file << ": " << e.what() );
//         }
//         catch (const std::exception& e) {
//             MB_SET_ERR( MB_FAILURE, "Exception while writing " << output_file << ": " << e.what() );
//         }
//     }

//     return MB_SUCCESS;
}

ErrorCode NetcdfMeshIO::write_point_scalars_to_file( Interface* mb,
                                                      const std::string& template_file,
                                                      const std::string& output_file,
                                                      const NetcdfWriteRequest& request,
                                                      const std::vector< EntityHandle >& entities )
{
    if( nullptr == mb ) MB_SET_ERR( MB_FAILURE, "Invalid MOAB interface" );
    if( entities.empty() ) MB_SET_ERR( MB_FAILURE, "No entities provided for NetCDF write" );

    const std::string dim_name = resolve_or_default( request.dimension_name, "ncol" );

    try
    {
        std::filesystem::copy_file( template_file,
                                    output_file,
                                    std::filesystem::copy_options::overwrite_existing );

        PnetCDF::NcmpiFile out( MPI_COMM_WORLD, output_file.c_str(), PnetCDF::NcmpiFile::write );
        PnetCDF::NcmpiDim dim = out.getDim( dim_name );
        MPI_Offset ncol       = dim.getSize();

        if( static_cast< size_t >( ncol ) != entities.size() )
        {
            MB_SET_ERR( MB_FAILURE,
                        "Entity count (" << entities.size() << ") does not match NetCDF dimension " << dim_name << " ("
                                       << ncol << ")" );
        }

        std::vector< PnetCDF::NcmpiDim > dims;
        dims.push_back( dim );

        auto define_variable = [&]( const std::string& var_name ) {
            out.addVar( var_name, PnetCDF::ncmpiFloat, dims );
        };

        for( const auto& name : request.scalar_var_names )
            define_variable( name );
        for( const auto& base_name : request.squared_var_names )
            define_variable( base_name + "_squared" );

        out.enddef();

        auto write_variable = [&]( const std::string& var_name ) -> ErrorCode {
            Tag tag = 0;
            MB_CHK_SET_ERR( mb->tag_get_handle( var_name.c_str(), tag ), "Failed to get tag for " << var_name );

            std::vector< float > values;
            MB_CHK_ERR( fetch_tag_as_float( mb, tag, entities, values ) );

            PnetCDF::NcmpiVar var = out.getVar( var_name );
            var.putVar_all( values.data() );

            if( request.verbose )
            {
                LOG( INFO ) << "Wrote NetCDF variable " << var_name << " to " << output_file;
            }
            return MB_SUCCESS;
        };

        for( const auto& name : request.scalar_var_names )
            MB_CHK_ERR( write_variable( name ) );

        for( const auto& base_name : request.squared_var_names )
        {
            const std::string var_name = base_name + "_squared";
            MB_CHK_ERR( write_variable( var_name ) );
        }

        ncmpi_close( out.getId() );
    }
    catch( const std::filesystem::filesystem_error& e )
    {
        MB_SET_ERR( MB_FAILURE, "File copy error for NetCDF output: " << e.what() );
    }
    catch( const PnetCDF::exceptions::NcmpiException& e )
    {
        MB_SET_ERR( MB_FAILURE, "NetCDF error while writing " << output_file << ": " << e.what() );
    }

    return MB_SUCCESS;
}

}  // namespace moab

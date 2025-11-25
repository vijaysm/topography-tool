#include "NetcdfMeshIO.hpp"

#include "ParallelPointCloudReader.hpp"
#include "easylogging.hpp"
#include "moab/ErrorHandler.hpp"

// #include <pnetcdf>

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

        Tag area_tag = 0;
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

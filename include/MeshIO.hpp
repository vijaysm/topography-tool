#ifndef NETCDF_MESH_IO_HPP
#define NETCDF_MESH_IO_HPP

#include "moab/Interface.hpp"
#include "ScalarRemapper.hpp"

#include <string>
#include <vector>

namespace moab
{

typedef double TagValueType;

struct NetcdfLoadOptions
{
    std::string dimension_name = "ncol";
    std::string lon_var_name   = "lon";
    std::string lat_var_name   = "lat";
    std::string area_var_name  = "area";
    std::string context_label;
    bool verbose = false;
};

struct NetcdfWriteRequest
{
    std::vector< std::string > scalar_var_names;
    std::vector< std::string > squared_var_names;
    std::string dimension_name = "ncol";
    bool verbose               = false;
};

// Forward declaration
class ScalarRemapper;
struct MeshData;

class NetcdfMeshIO
{
  public:
    static ErrorCode load_point_cloud_from_file( Interface* mb,
                                                 EntityHandle mesh_set,
                                                 const std::string& filename,
                                                 const NetcdfLoadOptions& options,
                                                 std::vector< EntityHandle >& entities_out );

    static ErrorCode write_point_scalars_to_file( Interface* mb,
                                                  const std::string& template_file,
                                                  const std::string& output_file,
                                                  const NetcdfWriteRequest& request,
                                                  const std::vector< EntityHandle >& entities );

    static ErrorCode write_point_scalars_to_file( Interface* mb,
                                                  const std::string& template_file,
                                                  const std::string& output_file,
                                                  const NetcdfWriteRequest& request,
                                                  const moab::ScalarRemapper::MeshData& mesh_data );
                                                  // const MeshData& mesh_data );
};

}  // namespace moab

#endif  // NETCDF_MESH_IO_HPP

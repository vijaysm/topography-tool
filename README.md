# mbda: MOAB remap tool using disk-based averaging algorithm


The **mbda** tool is designed to efficiently support the generation of atmospheric topography for model grids at ultra-high resolutions, including sub-kilometer and 100-meter scales. It addresses the limitations of legacy serial tools by providing a parallelized, scalable solution capable of processing massive datasets (e.g., 250m DEMs with over 13 billion points). The tool implements a disk-based averaging (“cell_avg”) algorithm that computes true area averages for both mesh-based and point-cloud grids, ensuring accurate and monotonic downsampling without introducing artificial extrema. It offers flexible input and output options, supporting various file formats and coordinate conventions, and provides a user-friendly command-line interface for specifying variables, remapping options, and target grid characteristics. The **mbda** tool is validated through a comprehensive suite of tests covering all required remap operations, ensuring robust performance across a range of atmospheric modeling workflows.

The expectation for **mbda** is that it will significantly improve the speed, memory efficiency, and accuracy of topography remapping compared to previous workflows, enabling practical use on modern high-resolution and RRM grids. By supporting disk averaging for point clouds and integrating seamlessly into the broader topography toolchain, **mbda** can eliminate the need for complex, multi-step legacy processes and provides a unified one-stop tool for both field and variance remapping. The tool is designed to be extensible for future requirements, with evolving documentation and example workflows to facilitate easy adoption.

## Usage

```bash
./mbda -s <source_file> -t <target_file> -o <output_file>
```

## Options

- `--help`: Show full help text
- `--source <source_file>`: Source NetCDF point cloud file
- `--target <target_file>`: Target mesh file (nc or H5M)
- `--output <output_file>`: Output mesh file with remapped data (ending with nc or h5m)
- `--dof-var <dof_var>`: DoF numbering variable name (bypasses format detection). *Default: ncol*
- `--lon-var <lon_var>`: Longitude variable name (bypasses format detection). *Default: lon*
- `--lat-var <lat_var>`: Latitude variable name (bypasses format detection). *Default: lat*
- `--area-var <area_var>`: Area variable name to read and store. *Default: area*
- `--fields <fields_str>`: Comma-separated field names to remap
- `--square-fields <square_fields_str>`: Comma-separated fields to remap squared fields (e.g., <field>_squared)
- `--remap-method <remap_method>`: Remapping method: da (ALG_DISKAVERAGE) or nn (ALG_NEAREST_NEIGHBOR). *Default: da*
- `--spectral`: Assume that the target mesh requires online spectral element mesh treatment. *Default: false*
- `--verbose,v`: Enable verbose output with timestamps. *Default: false*

## Example

```bash
./mbda -s /path/to/source.nc -t /path/to/target.h5m -o /path/to/output.h5m
```

## Notes

The tool can be used to remap a NetCDF point cloud data to a target mesh. The tool assumes that the target mesh file has dimensions of ncol (can be overridden by user with `--dof-var <ndofs>`). And the coordinates of the points of interest, typically GLL points are provided as `lon(dof_var)`, `lat(dof_var)`, where `lon_var` and `lat_var` names can be overridden by user with  area(ncol) variables available. If even one of these are missing, throw an error and exit. If all are present, load the file and add the loaded points to the mesh set.

The tool also assumes that the target mesh is a spectral element mesh. If the target mesh is not a spectral element mesh, the tool will throw an error and exit.

## Workflows

All sample workflows and descriptions have been provided in detail in the [Topography tool chain](https://e3sm.atlassian.net/wiki/spaces/DOC/pages/5251104838/Topography+tool+chain+-+description+and+upgrades+for+high-res) page. We provide details on how to achieve viable solutions for each of the test cases.

### Test 1

FVtoFV map: TERR and TERR^2  needed to compute SHG30 on ne3000pg1 grid.
- source grid:  GMTED2010_7.5_stitch_S5P_OPER_REF_DEM_15_NCL_24-3.r172800x86400.nc   (13.6G points, lat/lon grid)
- Target grid: ne3000pg1.scrip.nc  (54M grid points)
- Points and area given in variables:  grid_center_lon, grid_center_lat, grid_area

With USGS source grid
```bash
./mbda --target grids/ne3000pg1.scrip.nc --source grids/usgs-rawdata.nc --output remapped_data_da.nc --fields htopo --square-fields htopo --dof-var grid_size --lon-var grid_center_lon --lat-var grid_center_lat --area-var grid_area
```

With GMTED source grid
```bash
./mbda --target grids/ne3000pg1.scrip.nc --source grids/GMTED2010_7.5_stitch_S5P_OPER_REF_DEM_15_NCL_24-3.r172800x86400.nc --output remapped_data_da.nc --fields htopo --square-fields htopo --dof-var grid_size --lon-var grid_center_lon --lat-var grid_center_lat --area-var grid_area
```

### Test 2

FVtoSE map: TERR, TERR2
- source grid: GMTED2010_7.5_stitch_S5P_OPER_REF_DEM_15_NCL_24-3.r172800x86400.nc   (13.6G points, lat/lon grid)
- Target grid:  CAne32x128_Altamont100m_v2   (np4 GLL grid)   ( 300K elements,  3M grid points)
    - E3SM’s convention for these files is the (rarely used) “latlon” format:
    - SE Atmosphere Grid Overview (EAM & CAM)  This data can also be extracted from any mapping file (i.e. xc_a, yc_a, area_a extracted and renamed).
    - Points and area given for every GLL node are given in variables: “lat, lon, area”
- Permutter:
    - /global/cfs/cdirs/e3sm/taylorm/mapping/grids/CA100mnp4_homme_latlon.nc
    - /global/cfs/cdirs/e3sm/taylorm/mapping/grids/ne30np4_latlon_c20251028.nc
    - /global/cfs/cdirs/e3sm/taylorm/mapping/grids/ne256np4_latlon_c20190127.nc

With USGS source grid
```bash
./mbda --target grids/CA100mnp4_homme_latlon.nc --source grids/usgs-rawdata.nc --output remapped_data_da.nc --fields htopo --square-fields htopo
```

OR alternatively, if you want the tool to generate the spectral GLL jacobians and weights, use
```bash
./mbda --target grids/CAne32x128_Altamont100m_v2_scrip.nc --source grids/usgs-rawdata.nc --output remapped_data_da.h5m --fields htopo --square-fields htopo  --spectral
```

NOTE: The spectral option is only valid for SE grids (quads). The code still calculates the element averaged values on the SE coarse grid and this is not what you may want. Great for visualization however. And the output format is MOAB native h5m format as the target mesh was read in by MOAB as well (not custom reader). This helped in debugging the algorithm with domain files as targets to all other formats that MOAB natively supports already.

With GMTED source grid
```bash
./mbda --target grids/ne256np4_latlon_c20190127.nc --source grids/GMTED2010_7.5_stitch_S5P_OPER_REF_DEM_15_NCL_24-3.r172800x86400.nc  --output remapped_data_da.nc --fields htopo --square-fields htopo
```

### Test 3

FVtoFV map: SGH30, TERR and TERR^2
- source grid:
    - From old fortran tools:  USGS-topo-cube3000.nc, coordinates: lat,lon
    - From mbda: ( test 1 above).    cube3000.nc, with coordinates grid_center_lon, grid_center_lat
- target grid:  CAne32x128_Altamont100m_v2pg2.scrip.nc  (pg2 grid)
    - Points and area given in variables: grid_center_lon, grid_center_lat, grid_area

With USGS-topo-cube3000.nc source grid, we will now have to create a Kd-tree for point queries and can no longer use the fast point locator that we have been using with RLL grids. Hence, this workflow with have a O(nlog(n)) complexity. Here, we are going to project terr and SGH30 variables from USGS-topo-cube3000 grid to any target grid and also compute the projection of the terr^2 variable as well.
```bash
./mbda --target grids/ne256np4_latlon_c20190127.nc --source grids/USGS-topo-cube3000.nc  --output remapped_data_da.nc --fields terr,SGH30 --square-fields terr
```

### Test 4

FVtoFV map:  TERR and TERR^2
- map fields from GMTED2010_7.5_stitch_S5P_OPER_REF_DEM_15_NCL_24-3.r172800x86400.nc   to itself, using a 3.3km disk average.
- could also map to some MOAB grid with a similar resolution - like a cubed-sphere grid with 250m resolution.  The would be a one time operation, generating TERR and TERR_3km
- if we can get this to work, then we can use the 3.3km smoothed data on the GMTED2010 grid, and we would no longer need the cube3000 grid.

<u>**CURRENTLY UNSUPPORTED**:</u> This feature may require a refactor as we always explicitly load the target mesh. So if the target mesh is the massive RLL dataset, then the single node memory may not entirely suffice, or even index access into a `172800*86400*3=44B` double coordinate array in memory. It will require storing the target mesh as a logical tensor-product mesh as well, and hence requires some deeper changes.

## License

MIT License

## Author

Vijay Mahadevan

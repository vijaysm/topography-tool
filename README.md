# mbda: MOAB remap tool using disk-based averaging algorithm


The **mbda** tool is designed to efficiently support the generation of atmospheric topography for model grids at ultra-high resolutions, including sub-kilometer and 100-meter scales. It addresses the limitations of legacy serial tools by providing a parallelized (OpenMP threads), scalable solution capable of processing massive datasets (e.g., 250m DEMs with over 14 billion points). The tool implements a disk-based averaging (“cell_avg”) algorithm that computes true area averages for both mesh-based and point-cloud grids, ensuring accurate and monotonic downsampling without introducing artificial extrema. It offers flexible input and output options, supporting various file formats and coordinate conventions, and provides a user-friendly command-line interface for specifying variables, remapping options, and target grid characteristics. The **mbda** tool is being validated through a comprehensive suite of tests covering a range of atmospheric modeling workflows.

The expectation for **mbda** is that it will significantly improve the speed, memory efficiency, and accuracy of topography remapping compared to previous workflows, enabling practical use on modern high-resolution and RRM grids. By supporting disk averaging for point clouds and integrating seamlessly into the broader topography toolchain, **mbda** can eliminate the need for complex, multi-step legacy processes and provides a unified one-stop tool for both field and variance remapping. The tool is designed to be extensible for future requirements, with evolving documentation and example workflows to facilitate easy adoption.

The motivation for the disk averaging algorithm in **mbda** is also to support other climate initial-condition workflows where conservation is not a hard constraint. By specifying appropriate "area" factors, the averaging procedure can yield smoother reconstructions of initial condition data from very high-resolution datasets with fast wall-clock times.

## Usage

```bash
./mbda -s <source_file> -t <target_file> -o <output_file>
```

## Options

- `--help`: Show full help text
- `--source <source_file>`: Source NetCDF point cloud file
- `--target <target_file>`: Target mesh file (nc or h5m)
- `--output <output_file>`: Output mesh file with remapped data (ending with nc or h5m)
- `--dof-var <dof_var>`: DoF numbering variable name (bypasses format detection). *Default: ncol*
- `--lon-var <lon_var>`: Longitude variable name (bypasses format detection). *Default: lon*
- `--lat-var <lat_var>`: Latitude variable name (bypasses format detection). *Default: lat*
- `--area-var <area_var>`: Area variable name to read and store. *Default: area*
- `--fields <fields_str>`: Comma-separated field names to remap
- `--square-fields <square_fields_str>`: Comma-separated quadratic field names to remap (e.g., stored as `<field>_squared`)
- `--remap-method <remap_method>`: Remapping method: da (ALG_DISKAVERAGE) or nn (ALG_NEAREST_NEIGHBOR). *Default: da*
- `--spectral`: Assume that the target mesh requires online spectral element mesh treatment. *Default: false*
- `--reuse-source-mesh`: Skip loading a separate target mesh and reuse the cached source point cloud (self-smoothing workflows). Requires `--smoothing-area` and NetCDF inputs.
- `--smoothing-area <km2>`: Disk area (in square kilometers) used for averaging when reusing the source mesh. Determines the search radius for self-remapping kernels.
- `--verbose,v`: Enable verbose output with timestamps. *Default: false*

NOTE: If the target mesh file extension and output data file extension need to match. i.e., if target is ne256np4.h5m, then the output file has to be ne256np4_remapped.h5m.

## Example

```bash
./mbda -s /path/to/source.nc -t /path/to/target.h5m -o /path/to/output.nc
```

## Notes

The tool can be used to remap a NetCDF point cloud data to a target mesh. The tool assumes that the target mesh file has dimensions of ncol (can be overridden by user with `--dof-var <ndofs>`). And the coordinates of the points of interest, typically GLL points, are provided as `lon(dof_var)`, `lat(dof_var)`, where `lon_var` and `lat_var` names can be overridden by the user with  area(ncol) variables available. If even one of these is missing, throw an error and exit. If all are present, load the file and add the loaded points to the mesh set.

The tool also assumes that the target mesh is a spectral element mesh. If the target mesh is not a spectral element mesh, the tool will throw an error and exit.

## Workflows

All sample workflows and descriptions have been provided in detail in the [Topography tool chain](https://e3sm.atlassian.net/wiki/spaces/DOC/pages/5251104838/Topography+tool+chain+-+description+and+upgrades+for+high-res) page. We provide details on how to achieve viable solutions for these test cases.

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

Alternatively, if you want the tool to generate the spectral GLL Jacobians and weights, use
```bash
./mbda --target grids/CAne32x128_Altamont100m_v2_scrip.nc --source grids/usgs-rawdata.nc --output remapped_data_da.h5m --fields htopo --square-fields htopo  --spectral
```

NOTE: The spectral option is only valid for SE grids (quads). The code still calculates the element averaged values on the SE coarse grid and this is not what you may want. Great for visualization, however. And the output format is MOAB native h5m format as the target mesh was read in by MOAB as well (not a custom reader). This helped in debugging the algorithm with domain files as targets to all other formats that MOAB natively supports already.

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

With USGS-topo-cube3000.nc source grid, we will now have to create a Kd-tree for point queries and can no longer use the fast point locator that we have been using with RLL grids. Hence, this workflow with have a `O(nlog(n))` complexity. Here, we are going to project the `terr` and `SGH30` variables from USGS-topo-cube3000 grid to any target grid and also compute the projection of the terr^2 variable as well.
```bash
./mbda --target grids/ne256np4_latlon_c20190127.nc --source grids/USGS-topo-cube3000.nc  --output remapped_data_da.nc --fields terr,SGH30 --square-fields terr
```

### Test 4 – Self-smoothing on the source grid

FV→FV map where the source and target are identical (e.g., GMTED2010_7.5_stitch_S5P_OPER_REF_DEM_15_NCL_24-3 remapped to itself with a 3.3 km disk average). This workflow now leverages the cached point cloud directly instead of instantiating a new MOAB mesh:

1. Use `--reuse-source-mesh` so the reader exposes a transparent point-cloud view of the source coordinates/areas. No new vertices or tags are created, which keeps the memory footprint manageable even for $10^{10}$-point grids.
2. Provide `--smoothing-area` to control the constant disk radius used for averaging. This value determines the kernel size in km$^2$ for the scalar remapper.
3. Supply the usual `--fields` / `--square-fields` lists; the reader will auto-generate `<field>_squared` data if requested so the smoother can remap mean and variance in one pass.
4. When writing results, the NetCDF helper copies the original file to the output path and rewrites the existing variables in-place—no extra MOAB tags or new NetCDF variables are created. This matches the smoothing workflow’s requirement to simply overwrite the original variables with smoothed data.

Example:

```bash
./mbda --source grids/GMTED2010_7.5_stitch_S5P_OPER_REF_DEM_15_NCL_24-3.nc \
       --target grids/GMTED2010_7.5_stitch_S5P_OPER_REF_DEM_15_NCL_24-3.nc \
       --output GMTED2010_smoothed.nc \
       --fields htopo --square-fields htopo \
       --reuse-source-mesh --smoothing-area 7.4e-8
```

You can substitute any other FV target at similar resolution by omitting `--reuse-source-mesh`; in that case the classic mesh-loading path and NetCDF writer remain unchanged. The smoothing area of $7.4e-8$ km$^2$ corresponds to approximately a 3km disk radius (since $7.4e-8$ steradians on the unit sphere implies $r = \sqrt{7.4e-8 \times 6371^2 / \pi}$ km with radius of Earth taken as 6371 km).

## License

BSD 3-Clause License

## Author

Vijay Mahadevan

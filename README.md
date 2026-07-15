# CubePlot

Flat cube-net plotting experiments for cubed-sphere datasets.

## Main script

`plot_cube_net.py` plots a cubed-sphere dataset as a flat six-face cube net.

Features:
- face placement and rotation inferred from the dataset geometry
- Earth basemap mode or dataset-variable fill mode
- optional external Earth rasters via `--earth-image`
- automatic Blue Marble download via `--earth-image blue-marble`
- configurable figure background, annotation color, and overlay color
- auto-contrast face labels, arrows, and `u/v` labels for dark data fields
- portable `uv` script metadata at the top of the file

## Quick start

Run with the built-in Earth background:

```bash
uv run plot_cube_net.py c24.nc4 --background earth
```

You can also run the script directly:

```bash
./plot_cube_net.py c24.nc4 --background earth
```

The figure canvas defaults to white. You can also use:

```bash
uv run plot_cube_net.py c24.nc4 --background earth --canvas black
uv run plot_cube_net.py c24.nc4 --background earth --canvas transparent --output cube_plot.png
```

Run with the built-in Blue Marble helper:

```bash
uv run plot_cube_net.py c24.nc4 --background earth --earth-image blue-marble
```

The first Blue Marble run downloads and caches the NASA image automatically.

Run with a data field instead:

```bash
uv run plot_cube_net.py c24.nc4 --background data --variable SLP
```

For dark fields such as `PHIS`, you can either use the default auto-contrast annotations or force white styling explicitly:

```bash
uv run plot_cube_net.py c24.nc4 --background data --variable PHIS --output phis.png
uv run plot_cube_net.py c24.nc4 --background data --variable PHIS --annotation-color white --overlay-color white --output phis.png
```

Use a different colormap or time/level selection:

```bash
uv run plot_cube_net.py c24.nc4 --background data --variable T --level-index 10 --cmap viridis
uv run plot_cube_net.py c24.nc4 --background data --variable PS --time-index 0
```

Save to a file:

```bash
uv run plot_cube_net.py c24.nc4 --background earth --output cube_plot.png
```

## CLI options

```text
dataset
    Path to the cubed-sphere NetCDF file.

--background {earth,data}
    Use a shaded Earth basemap or a dataset variable fill.

--earth-image EARTH_IMAGE
    Optional path to a global Earth raster, or `blue-marble` to download and
    cache NASA Blue Marble automatically.

--variable VARIABLE
    Variable to plot in `data` mode. Default: `SLP`.

--time-index TIME_INDEX
    Time index for variables with a `time` dimension.

--level-index LEVEL_INDEX
    Level index for variables with a `lev` dimension.

--cmap CMAP
    Matplotlib colormap name for `data` mode. Default: `turbo`.

--canvas {white,black,transparent}
    Figure background color. Default: `white`.

--annotation-color {auto,black,white}
    Color for face numbers, arrows, and `u/v` labels. Default: `auto`.

--overlay-color {auto,black,white}
    Color for gridlines, coastlines, and borders. Default: `auto`.

--output OUTPUT
    Optional output image path.
```

## Notebook

`UXarray-grid.ipynb` includes the notebook workflow and imports `plot_cube_net` directly.

## Notes

- External rasters are assumed to be global equirectangular Earth images.
- Georeferenced rasters currently need to use `EPSG:4326`.
- The script suppresses the recurring xarray duplicate-dimension warning emitted when opening some cubed-sphere datasets.
- `*.nc4` files are gitignored in this repo.

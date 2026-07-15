# CubePlot

Flat cube-net plotting experiments for cubed-sphere datasets.

## Main script

`plot_cube_net.py` plots a cubed-sphere dataset as a flat six-face cube net.

Features:
- face placement and rotation inferred from the dataset geometry
- Earth basemap mode or dataset-variable fill mode
- optional external Earth rasters via `--earth-image`
- portable `uv` script metadata at the top of the file

## Quick start

Run with the built-in Earth background:

```bash
uv run plot_cube_net.py c24.nc4 --background earth
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

Run with a data field instead:

```bash
uv run plot_cube_net.py c24.nc4 --background data --variable SLP
```

Save to a file:

```bash
uv run plot_cube_net.py c24.nc4 --background earth --output cube_plot.png
```

## Notebook

`UXarray-grid.ipynb` includes the notebook workflow and imports `plot_cube_net` directly.

## Notes

- External rasters are assumed to be global equirectangular Earth images.
- Georeferenced rasters currently need to use `EPSG:4326`.
- `*.nc4` files are gitignored in this repo.

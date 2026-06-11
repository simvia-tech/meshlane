
<p align="center">
  <a href="https://github.com/nschloe/meshio"><img alt="meshio" src="https://nschloe.github.io/meshio/logo-with-text.svg" width="55%"></a>
</p>

<p align="center"><b>I/O for mesh files.</b></p>

<p align="center">
  <a href="https://pypi.org/project/meshio/"><img alt="PyPI version" src="https://img.shields.io/pypi/v/meshio.svg?logo=pypi&logoColor=white"></a>
  <a href="https://github.com/simvia-tech/meshio/actions"><img alt="CI-CD" src="https://img.shields.io/github/actions/workflow/status/simvia-tech/meshio/ci.yml?logo=github&label=CI-CD"></a>
  <a href="https://github.com/simvia-tech/meshio/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-green"></a>
  <a href="https://anaconda.org/conda-forge/meshio"><img alt="conda-forge" src="https://img.shields.io/conda/vn/conda-forge/meshio?logo=anaconda&logoColor=white"></a>
</p>

# meshio

**meshio** is a Python library to read and write a wide range of mesh file formats.
It supports many element types and handles **more than 30 formats** (see the table
below), making it a convenient interchange tool between simulation and meshing software.

Maintained with ❤️ by [Simvia](https://simvia.tech) as part of our open-source
simulation toolchain.

## ⚡ Features

- Read **and** write 30+ mesh formats from a single, unified API
- Rich element-type support (triangles, quads, tets, hexes, and more)
- In-memory `Mesh` object with `points`, `cells`, `point_data`, `cell_data`, …
- Time-series support through the XDMF format
- Command-line tool: `convert`, `info`, `compress`, `binary`/`ascii`
- ParaView plugin to open any meshio-supported file directly

### Supported formats

| Format | Extension | Read | Write |
|--------|-----------|:----:|:-----:|
| Abaqus | `.inp` | ✅ | ✅ |
| ANSYS msh | `.msh` | ✅ | ✅ |
| ANSYS inp | `.inp` | ✅ | ✅ |
| ANSYS cdb | `.cdb` | ✅ | ✅ |
| AVS-UCD | `.avs` | ✅ | ✅ |
| CGNS | `.cgns` | ✅ | ✅ |
| DOLFIN/FEniCS XML | `.xml` | ✅ | ✅ |
| Exodus | `.e`, `.exo` | ✅ | ✅ |
| FLAC3D | `.f3grid` | ✅ | ✅ |
| Gmsh (v2) | `.msh` | ✅ | ✅ |
| Gmsh (v4) | `.msh` | ✅ | ✅ |
| H5M (MOAB) | `.h5m` | ✅ | ✅ |
| Kratos/MDPA | `.mdpa` | ✅ | ✅ |
| MED/Salome | `.med` | ✅ | ✅ |
| Medit | `.mesh`, `.meshb` | ✅ | ✅ |
| nastran | `.bdf`, `.nas`, `.fem` | ✅ | ✅ |
| Netgen | `.vol`, `.vol.gz` | ✅ | ✅ |
| neuroglancer | `.precomputed` | ✅ | ✅ |
| OBJ | `.obj` | ✅ | ✅ |
| OFF | `.off` | ✅ | ✅ |
| OpenFOAM | `.foam` | ✅ | ✅ |
| PERMAS | `.post`, `.post.gz`, `.dato`, `.dato.gz` | ✅ | ✅ |
| PLY | `.ply` | ✅ | ✅ |
| STL | `.stl` | ✅ | ✅ |
| SU2 | `.su2` | ✅ | ✅ |
| SVG | `.svg` | ❌ | ✅ |
| Tecplot | `.dat` | ✅ | ✅ |
| TetGen | `.node` / `.ele` | ✅ | ✅ |
| UGRID | `.ugrid` | ✅ | ✅ |
| VTK (legacy) | `.vtk` | ✅ | ✅ |
| VTK XML (various) | `.vtu`, `.vts`, `.vtr`, `.vtp`, `.vti` | ✅ | ✅ |
| XDMF | `.xdmf`, `.xmf` | ✅ | ✅ |
| WKT CRS | `.wkt` | ✅ | ❌ |

## 🛠 Installation

meshio is [available on PyPI](https://pypi.org/project/meshio/) and
[conda-forge](https://anaconda.org/conda-forge/meshio):

```sh
pip install meshio[all]
# or
conda install -c conda-forge meshio
```

> `[all]` pulls in all optional dependencies (`netcdf4`, `h5py`, …).
> By default, meshio only depends on **numpy**.

### Verify installation

```sh
meshio --version
meshio --help
```

## 🚀 Usage

### Command line

```sh
meshio convert    input.msh output.vtk   # convert between two formats
meshio info       input.xdmf             # show some info about the mesh
meshio compress   input.vtu              # compress the mesh file
meshio decompress input.vtu              # decompress the mesh file
meshio binary     input.msh              # convert to binary format
meshio ascii      input.msh              # convert to ASCII format
```

…with any of the supported formats.

### Reading a mesh

```python
import meshio

mesh = meshio.read(
    filename,  # string, os.PathLike, or a buffer/open file
    # file_format="stl",  # optional if filename is a path; inferred from extension
)
# mesh.points, mesh.cells, mesh.cells_dict, ...

# mesh.vtk.read() is also possible
```

### Writing a mesh

```python
import meshio

# two triangles and one quad
points = [
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
    [2.0, 0.0],
    [2.0, 1.0],
]
cells = [
    ("triangle", [[0, 1, 2], [1, 3, 2]]),
    ("quad", [[1, 4, 5, 3]]),
]

mesh = meshio.Mesh(
    points,
    cells,
    # Optionally provide extra data on points, cells, etc.
    point_data={"T": [0.3, -1.2, 0.5, 0.7, 0.0, -3.0]},
    # Each item in cell data must match the cells array
    cell_data={"a": [[0.1, 0.2], [0.4]]},
)
mesh.write(
    "foo.vtk",  # str, os.PathLike, or buffer/open file
    # file_format="vtk",  # optional if first argument is a path; inferred from extension
)

# Alternative with the same options
meshio.write_points_cells("foo.vtk", points, cells)
```

For both input and output, you can optionally specify the exact `file_format`
(to enforce ASCII over binary VTK, for example).

### Time series

The [XDMF format](https://xdmf.org/index.php/XDMF_Model_and_Format) supports time
series with a shared mesh. Write time-series data with:

```python
with meshio.xdmf.TimeSeriesWriter(filename) as writer:
    writer.write_points_cells(points, cells)
    for t in [0.0, 0.1, 0.21]:
        writer.write_data(t, point_data={"phi": data})
```

…and read it back with:

```python
with meshio.xdmf.TimeSeriesReader(filename) as reader:
    points, cells = reader.read_points_cells()
    for k in range(reader.num_steps):
        t, point_data, cell_data = reader.read_data(k)
```

## 🔌 ParaView plugin

<img alt="gmsh paraview" src="https://nschloe.github.io/meshio/gmsh-paraview.png" width="60%">

*A Gmsh file opened with ParaView.*

If you downloaded a binary version of ParaView:

1. Install meshio for the Python major version ParaView uses (check `pvpython --version`).
2. Open ParaView.
3. Load `paraview-meshio-plugin.py` from your meshio installation
   (Linux: `~/.local/share/paraview-5.9/plugins/`) under
   *Tools → Manage Plugins → Load New*.
4. *Optional:* activate **Auto Load**.

You can now open all meshio-supported files in ParaView.

## 📊 Performance

Benchmarks below use a triangular mesh with ~900k points and ~1.8M triangles.
Red lines mark the size of the mesh in memory.

| File sizes | I/O speed | Max memory |
|:----------:|:---------:|:----------:|
| <img alt="file size" src="https://nschloe.github.io/meshio/filesizes.svg" width="100%"> | <img alt="performance" src="https://nschloe.github.io/meshio/performance.svg" width="100%"> | <img alt="memory usage" src="https://nschloe.github.io/meshio/memory.svg" width="100%"> |

## 🧪 Testing

To run meshio unit tests, check out this repository and run:

```sh
tox
```

## 🤝 Contributors

meshio was created by [Nico Schlömer](https://github.com/nschloe) and is built by
[many contributors](https://github.com/nschloe/meshio/graphs/contributors).
This fork is maintained by the **Simvia** team.

## 🔗 See Also

- [meshio (upstream)](https://github.com/nschloe/meshio)
- [Simvia  open-source simulation software](https://simvia.tech/software)
- [vs_code_aster repository](https://github.com/simvia-tech/vs-code-aster.git)
- [Simvia Docker Hub](https://hub.docker.com/r/simvia/code_aster)

## 📄 License

This project is published under the **MIT License**.
See the [LICENSE](https://en.wikipedia.org/wiki/MIT_License) file for details.

## 💬 Reach Us

We love feedback! Don't hesitate to open a
[GitHub issue](https://github.com/simvia-tech/meshio/issues/new), or reach out to us
on our website <https://simvia.tech/fr#contact>.
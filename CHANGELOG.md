<!--pytest-codeblocks:skipfile-->
# Changelog

This document only describes _breaking_ changes in meshio. If you are interested in bug
fixes, enhancements etc., best follow [the meshio project on
GitHub](https://github.com/nschloe/meshio). meshlane-specific changes are listed at the
top; the meshio history follows below.

## meshlane 5.6.0 (Oct 29, 2026)

### Added
- CGNS reader and writer, rewritten (HDF5 files). Reads both unstructured and
  structured zones. meshlane has no structured representation, so a structured zone is
  destructured on read: its i/j/k grid becomes unstructured hexahedron, quad or line
  cells (with a warning). Linear and quadratic elements, MIXED sections and polyhedra
  (NGON_n/NFACE_n) are supported; boundary conditions and families become groups, and
  flow solutions become point/cell data. The writer emits unstructured CGNS that passes
  `cgnscheck`. `meshlane.cgns.read_multi` / `write_multi` read and write multi-zone
  files, one mesh per zone. (#49)
- Ansys Fluent `.msh` and `.msh.h5` (HDF5) reader and writer: the reader rebuilds the
  volume cells from the faces (tetra, pyramid, wedge, hexahedron, polyhedra; triangles,
  quads and polygons in 2D), ASCII and binary, with the named zones as groups; the
  writer produces the face-based layout Fluent reads, binary or ASCII. (#48)
- OpenVolumeMesh `.ovm` (ASCII) reader and writer. Tetrahedra, pyramids, wedges and
  hexahedra are rebuilt from the half-face topology; shared edges and faces are written
  once, with half-face normals pointing into the cells as OpenVolumeMesh expects.
  Properties are skipped on reading, and point/cell data are not written. (#7, #40)
- OpenFOAM polyMesh writer, ASCII or binary (`binary=True`, or `meshlane binary
  case/case.foam`). Writing `case/case.foam` produces
  `case/constant/polyMesh`; faces are built and matched with numpy, oriented from owner
  to neighbour and ordered as OpenFOAM requires. Boundary patches come from the groups
  of surface cells (`cell_sets`, or MED/OpenFOAM `cell_tags`); ungrouped boundary faces
  go to `defaultFaces`. Quadratic cells are written with their corner nodes, and unused
  points are dropped. Files a previous mesh left in `polyMesh` (zones, sets,
  refinement levels) are moved to `polyMesh.orig`. The reader now keeps the patch
  types in `mesh.patch_types`. (#8, #42)
- gmsh: the missing wedge18 (prism18) node ordering, so quadratic prisms read and
  write correctly; the reverse table is now derived from the forward one so the two
  cannot drift. (#27, #28)

### Changed
- **Breaking:** `.msh` files are now written as **Gmsh** by default. Previously a mesh
  without Gmsh/MED-specific data written to `.msh` produced an Ansys Fluent file. To
  write Fluent, pass `file_format="ansys"` (CLI: `--output-format ansys`). Reading is
  unaffected. (#48)

### Fixed
- Ansys/APDL (`ansysInp`, `.cdb`): 2D plane meshes keep their type and dimension. The
  writer emits surface elements as PLANE182/PLANE183 (with `KEYOPT(1)` so triangles
  initialize) and writes boundary groups as node components instead of BEAM189 beams;
  the reader returns 2D points for plane element types. A 2D plane-stress mesh converted
  `.med` to `.cdb` (and back) no longer becomes 3D SHELL281/SOLID185/BEAM189. (#37, #47)
- MED: files mixing blocks with and without families (e.g. volume cells without
  `FAM` and boundary faces with groups, as in snappyHexMesh meshes exported from
  Salome) no longer fail with `Incompatible cell data 'cell_tags'`; blocks without
  families get family 0. The multi-mesh reader used by the CLI now reads polyhedra
  (`POE`) like `meshlane.read`, instead of failing with `KeyError: 'POE'`. Both
  readers share the same cell reading code. (#41)
- OpenFOAM reader: binary `faces` files as OpenFOAM writes them (`faceCompactList`)
  are now read; they failed with a `ValueError`. (#42)
- SU2: writing a mesh with boundary elements no longer fails with `cannot unpack
  non-iterable CellBlock object`. Marker names are kept: the reader stores them in
  `cell_sets` (and still in `cell_data["su2:tag"]`), the writer takes them from
  `cell_sets`, the MED/OpenFOAM `cell_tags` families or the first integer cell data.
  2D meshes with z = 0 coordinates are written as `NDIME= 2`. Files with Windows line
  endings or spaces around `=` are read. Reading and writing are vectorized. (#44)
- SU2: multi-zone files (`NZONE`/`IZONE`) are read and written as a list of meshes with
  `meshlane.su2.read_multi` / `write_multi`, like multi-mesh MED files; they used to
  give a corrupted mesh. `meshlane convert` and `meshlane info` handle them, e.g.
  `meshlane convert cht.su2 cht.med` (one MED mesh per zone) and back. (#44)
- SU2: the sections after the markers (periodicity, FFD boxes used by shape
  optimization) are kept in `mesh.su2_extra` and written back unchanged, instead of
  producing one warning per line and being lost. Quadratic cells are written with their
  corner nodes instead of being dropped; unused points are removed; cells that are
  neither domain nor boundary elements, and groups merged by marker name cleaning, are
  reported. (#44)

## meshlane 5.5.0 (Aug 24, 2026)

### Added
- numpy 2.x support: the `numpy<2` upper cap is removed, so meshlane runs on both
  numpy 1.20+ and numpy 2.x. (#31)

### Changed
- **Minimum Python is now 3.10** (3.8 and 3.9 dropped). Both are end-of-life, and
  numpy 2.x requires Python 3.10+. (#31)

### Fixed
- MED to Abaqus `.inp` via the CLI (`meshlane convert`) now preserves node and
  element groups (`*NSET`/`*ELSET`) as the multi-mesh read path previously dropped
  them. Element types are also written canonically (`C3D8`, `S4`, `B31`) instead
  of non-standard variants (`C3D8RH`, `CAX4P`, `B31H`). (#33, #25, #29)
- MED: text metadata (group names, units, description) is written as UTF-8, so
  Salome and code_aster read accented names correctly. (#32)
- Abaqus: the reader maps thermal and gasket element families, reads
  assembly-based input files (`*Part`/`*Instance`), and uses less memory on large
  files. (#30)

## meshlane 5.4.3 (Jul 23, 2026)

### Added
- CLI: `meshlane convert` handles multi-mesh MED files. Every mesh is read and
  written out (to MED); converting to a format that cannot hold more than one
  mesh reports a clear error. (#20)

### Fixed
- Ansys: the `.inp` reader collapses degenerate elements that ANSYS stores with
  repeated nodes (tets/wedges/pyramids written as hexes, triangles written as
  quads) to their real type. This fixes wrong cell types and zero-volume cells that were rejected by
  solvers. (#22)

## meshlane 5.4.2 (Jul 17, 2026)

### Added
- MED: read/write support for polyhedra (`MED_POLYHEDRON`) and variable-node
  polygon (`MED_POLYGON`) writing, so OpenFOAM (snappyHexMesh) polyhedral
  meshes convert to MED. (#12)
- CLI: `meshlane info` handles multi-mesh MED files; `meshlane convert` prints
  read/write progress. (#14)
- CLI: `meshlane convert --remove-duplicates` removes coincident (duplicate)
  cells. By default they are kept and a warning is emitted instead. (#17, #18)

### Fixed
- MED: 3D cells are written with consistent, correct orientation for external MED
  readers, for both linear and quadratic cells (tetra10, hexahedron20, ...), via
  the meshlane<->MED node ordering plus a topological pass for warped cells, so
  `foam -> med` and `inp -> med` meshes are accepted by MED tools (code_saturne, code_aster, etc.). (#9, #13, #16)
- MED: every cell group is written with its numeric `GEO` attribute, so
  meshlane's MED files load in readers such as code_aster. (#16)
- MED: Gmsh physical groups are preserved when writing MED, so a `.msh -> .med`
  conversion no longer drops its groups. (#11)
- Ansys: the `.inp` reader handles 1-integer `NBLOCK` and COMPACT `EBLOCK`
  formats, so real Ansys Workbench exports convert. (#15)

### Changed
- Abaqus: more robust `.inp` reader (membrane/surface elements, set-of-sets,
  `*ELSET, GENERATE`, encoding fallback). (#5)
- OpenFOAM: faster, memory-bounded polyMesh reader (binary + ASCII). (#3)
- MED: more robust family/group handling (HDF5 creation-order tracking, dynamic
  family generation). (#4)

### Note
Because of the MED orientation and `GEO` fixes, a MED file written by this
version differs from one written by 5.4.1 for 3D meshes (now correctly oriented
and loadable by external MED tools). MED <-> MED round-trips are unaffected.

## meshlane 5.4.1

First deploy on pypi.org using github action.

## meshlane 5.4.0

meshlane is a fork of meshio, maintained by Simvia. Breaking changes relative to the
meshio base it forked from:

- The Python package and import name is now `meshlane` (was `meshio`), and the
  command-line tool is `meshlane` (was `meshio`).
- `numpy` is temporarily pinned to `<2` pending a numpy 2.x migration.

Fork additions carried over (non-breaking): OpenFOAM polyMesh reader, Ansys/APDL `.inp`
reader, and MED/Salome improvements (multi-mesh, polygon support, more robust
Code_Aster round-trips).

## v5.1.0 (Dec 11, 2021)

- CellBlocks are no longer tuples, but classes. You can no longer iterate over them like
  ```python
  for cell_type, cell_data in cells:
      pass
  ```
  Instead, use
  ```python
  for cell_block in cells:
      cell_block.type
      cell_block.data
  ```

## v5.0.0 (Aug 06, 2021)

- meshio now only provides one command-line tool, `meshio`, with subcommands like
  `info`, `convert`, etc. This replaces the former `meshio-info`, `meshio-convert` etc.

## v4.4.0 (Apr 29, 2021)

- Polygons are now stored as `"polygon"` cell blocks, not `"polygonN"` (where `N` is the
  number of nodes per polygon). One can simply retrieve the number of points via
  `cellblock.data.shape[1]`.

## v4.0.0 (Feb 18, 2020)

- `mesh.cells` used to be a dictionary of the form

  ```python
  {
    "triangle": [[0, 1, 2], [0, 2, 3]],
    "quad": [[0, 7, 1, 10], ...]
  }
  ```

  From 4.0.0 on, `mesh.cells` is a list of tuples,

  ```python
  [
    ("triangle", [[0, 1, 2], [0, 2, 3]]),
    ("quad", [[0, 7, 1, 10], ...])
  ]
  ```

  This has the advantage that multiple blocks of the same cell type can be accounted
  for. Also, cell ordering can be preserved.

  You can now use the method `mesh.get_cells_type("triangle")` to get all cells of
  `"triangle"` type, or use `mesh.cells_dict` to build the old dictionary structure.

- `mesh.cell_data` used to be a dictionary of the form

  ```python
  {
    "triangle": {"a": [0.5, 1.3], "b": [2.17, 41.3]},
    "quad": {"a": [1.1, -0.3, ...], "b": [3.14, 1.61, ...]},
  }
  ```

  From 4.0.0 on, `mesh.cell_data` is a dictionary of lists,

  ```python
  {
    "a": [[0.5, 1.3], [1.1, -0.3, ...]],
    "b": [[2.17, 41.3], [3.14, 1.61, ...]],
  }
  ```

  Each data list, e.g., `mesh.cell_data["a"]`, can be `zip`ped with `mesh.cells`.

  An old-style `cell_data` dictionary can be retrieved via `mesh.cell_data_dict`.

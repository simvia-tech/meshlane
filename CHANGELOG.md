<!--pytest-codeblocks:skipfile-->
# Changelog

This document only describes _breaking_ changes in meshio. If you are interested in bug
fixes, enhancements etc., best follow [the meshio project on
GitHub](https://github.com/nschloe/meshio). meshlane-specific changes are listed at the
top; the meshio history follows below.

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

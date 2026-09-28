# Product Roadmap — meshlane

> Read, write, and convert meshes across 30+ formats.

meshlane (an actively maintained fork of [meshio](https://github.com/nschloe/meshio)) aims to move meshes between simulation tools without losing what a solver needs: coordinates, connectivity, groups, fields, and the semantics attached to them (element type intent, mesh dimensionality, and whether a group is a set of nodes, elements, or faces).

*Applies to v5.5.0.*

## Current Capabilities

- 30+ mesh formats through a single neutral `Mesh` model
- MED 4.1 read/write: multi-mesh files, node/element groups and families, fields with units and component names, multi-timestep results, polyhedra and variable-node polygons
- Native OpenFOAM polyMesh reader (ASCII and binary)
- Ansys MAPDL reader/writer (`.inp` / `.cdb`)
- Abaqus reader: assembly support (`*Part` / `*Instance`), ~15x faster on files fragmented into many `*ELEMENT` sections
- Groups and families carried through conversions between formats that support them (MED families, Abaqus / Ansys `*NSET` / `*ELSET`, Gmsh physical groups, ...)
- numpy 2.x migration: runs on numpy 1.20+ and numpy 2.x, Python 3.10+

## Roadmap

### Short Term: corrections & compatibility

**Goal:** *Close the gaps that block real industrial files today.*

- **Solver-faithful semantics:** preserve element type intent (e.g. PLANE vs SHELL), mesh dimensionality (2D/3D), and node / element / face groups across conversion: the gaps behind #35, #37, #38
- **MED version bridge:** convert files between MED versions (3.x <-> 6.x), designed to extend to other format-version pairs
- **meshlane check:** a diagnostic CLI that flags duplicated points, inverted or degenerate cells, and inconsistent groups before a mesh reaches a solver
- **meshlane-tutorials:** a public repository with tutorials, examples, benchmarks, and scripts

### Mid Term: robustness & reach

**Goal:** *Broaden format coverage and make conversions trustworthy release to release.*

- **Full CGNS reader and writer:** read multi-zone (multi-block) CGNS files, the current reader handles a single zone only
- **OpenFOAM polyMesh writer** (community-requested, reading is already supported)
- **Continuous solver validation:** routinely run conversions through the target software (code_aster, code_saturne, etc.) in CI, so round-trips stay correct across releases

### Long Term: scale & new formats

**Goal:** *Handle the largest meshes and keep widening format coverage.*

- Performance and memory improvements on large meshes
- Additional community-requested mesh formats

## Out of Scope (for now)

- Mesh generation and remeshing
- General-purpose or interactive mesh editing
- Solver case setup (materials, sections, loads, boundary conditions)

## Success Criteria

- Real industrial files (code_aster, code_saturne, Ansys, Abaqus, etc.) convert without losing groups, cell types, or dimensionality
- Conversions validated by loading the output in the target solver, not only by internal round-trips
- Fewer solver-side errors caused by mesh conversion
- Growing adoption and community contributions on GitHub and PyPI

## Open-Source Principles

- Transparent roadmap and changelog
- Semantic versioning
- Community-driven improvements (issues and pull requests welcome)
- Full meshio history and source attribution preserved

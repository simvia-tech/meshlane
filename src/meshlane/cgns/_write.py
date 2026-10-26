"""Write CGNS files (HDF5), following the node layout of the CGNS library so that
the files are valid for cgnscheck, the CGNS library and the tools built on it.

One base, one unstructured zone:

- meshes made of standard elements: one Elements_t section per cell type, volume
  cells first, split by volume group; each section belongs to the families of
  its group (FamilyName_t, AdditionalFamilyName_t);
- meshes with polyhedra: all cells are written as NFACE_n, and all faces once as
  NGON_n (CGNS 4 layout with ElementStartOffset); a face is oriented out of the
  first cell that uses it, which lists it with a positive sign, the other with a
  negative one;
- 2D meshes with polygons: all 2D cells are written as NGON_n.

Groups of faces/edges become BC_t nodes (PointList at FaceCenter/EdgeCenter),
groups of nodes ZoneSubRegion_t nodes (PointList at Vertex); every group of cells
has a Family_t. Point data and cell data (of the volume cells) become
FlowSolution_t nodes at Vertex and CellCenter.
"""

import numpy as np

from .._common import warn
from .._exceptions import WriteError

VERSION = 4.0  # first version with ElementStartOffset

# meshlane cell type -> CGNS code, for the standard elements
_CODES = {
    "vertex": 2,
    "line": 3,
    "line3": 4,
    "triangle": 5,
    "triangle6": 6,
    "quad": 7,
    "quad8": 8,
    "quad9": 9,
    "tetra": 10,
    "tetra10": 11,
    "pyramid": 12,
    "pyramid13": 21,
    "pyramid14": 13,
    "wedge": 14,
    "wedge15": 15,
    "wedge18": 16,
    "hexahedron": 17,
    "hexahedron20": 18,
    "hexahedron27": 19,
}
_NGON_n, _NFACE_n = 22, 23

# outward faces of the regular 3D cells, in meshlane node order
_OUTWARD = {
    "tetra": [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
    "pyramid": [[0, 3, 2, 1], [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]],
    "wedge": [[0, 2, 1], [3, 4, 5], [0, 1, 4, 3], [1, 2, 5, 4], [2, 0, 3, 5]],
    "hexahedron": [
        [0, 3, 2, 1],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
    ],
}

_MAX_NAME = 32


# -- CGNS nodes ------------------------------------------------------------------


def _set_string(obj, key, value, size):
    """Fixed-size, null-terminated string attribute, as the CGNS library writes."""
    import h5py

    string = h5py.h5t.C_S1.copy()
    string.set_size(size)
    string.set_strpad(h5py.h5t.STR_NULLTERM)
    if key in obj.attrs:
        del obj.attrs[key]
    attr = h5py.h5a.create(
        obj.id, key.encode(), string, h5py.h5s.create(h5py.h5s.SCALAR)
    )
    attr.write(np.array(value.encode()[: size - 1], dtype=f"S{size}"))


class _Writer:
    def __init__(self, f, compression, compression_opts):
        self.f = f
        self.compression = compression
        self.compression_opts = compression_opts

    def node(self, parent, name, label, value=None):
        """A CGNS node; value is a numpy array (2D arrays given in SIDS order,
        stored transposed as CGNS arrays are Fortran ordered), a string (C1) or
        None (MT)."""
        group = parent.create_group(name, track_order=True)
        if value is None:
            kind = "MT"
        elif isinstance(value, str):
            value = np.frombuffer(value.encode(), dtype=np.int8)
            kind = "C1"
        else:
            value = np.asarray(value)
            kind = {"int32": "I4", "int64": "I8", "float32": "R4", "float64": "R8"}[
                value.dtype.name
            ]
        _set_string(group, "name", name, 33)
        _set_string(group, "label", label, 33)
        _set_string(group, "type", kind, 3)
        group.attrs.create("flags", np.int32([1]))
        if value is not None:
            value = value.T if value.ndim > 1 else value
            big = value.size > 1024 and self.compression is not None
            group.create_dataset(
                " data",
                data=value,
                compression=self.compression if big else None,
                compression_opts=self.compression_opts if big else None,
            )
        return group


def _index(values):
    """Index array as I4 when it fits, else I8 (both are read by the library)."""
    values = np.asarray(values)
    if values.size == 0 or values.max() < 2**31 - 1:
        return values.astype(np.int32)
    return values.astype(np.int64)


class _Names:
    """CGNS node names: at most 32 characters, no '/', unique among siblings."""

    def __init__(self):
        self.used = set()
        self.renamed = {}

    def __call__(self, name):
        clean = str(name).replace("/", "_")[:_MAX_NAME] or "unnamed"
        k = 1
        while clean in self.used:
            suffix = f"~{k}"
            clean = str(name).replace("/", "_")[: _MAX_NAME - len(suffix)] + suffix
            k += 1
        self.used.add(clean)
        if len(str(name)) > _MAX_NAME or "/" in str(name):
            self.renamed[name] = clean
        return clean


# -- mesh layout ----------------------------------------------------------------


def _groups_of_block(cell_sets, block, size):
    """For each cell of a block, the tuple of the cell sets it belongs to."""
    member = {}
    for name, per_block in cell_sets.items():
        if block < len(per_block) and len(per_block[block]):
            member[name] = np.asarray(per_block[block], dtype=np.int64)
    combos = [()] * size
    if not member:
        return combos
    names = sorted(member)
    flags = np.zeros((size, len(names)), dtype=bool)
    for j, name in enumerate(names):
        flags[member[name], j] = True
    keys = {}
    out = []
    for row in flags:
        key = row.tobytes()
        if key not in keys:
            keys[key] = tuple(n for n, f in zip(names, row) if f)
        out.append(keys[key])
    return out


def _split_by_groups(combos):
    """{group tuple: local indices} in first-seen order."""
    out = {}
    for i, combo in enumerate(combos):
        out.setdefault(combo, []).append(i)
    return {k: np.array(v, dtype=np.int64) for k, v in out.items()}


def _face_keys(ptr, nodes):
    """Two order-independent hashes of every face given as CSR (sums of random
    values of its nodes), by chunks of faces to bound the memory; used with the
    face size to find the faces shared by two cells."""
    rng = np.random.default_rng(0)
    high = int(nodes.max(initial=0)) + 1
    r1 = rng.integers(0, 2**63, size=high, dtype=np.uint64)
    r2 = rng.integers(0, 2**63, size=high, dtype=np.uint64)
    num_faces = len(ptr) - 1
    h1 = np.empty(num_faces, dtype=np.uint64)
    h2 = np.empty(num_faces, dtype=np.uint64)
    chunk = 1 << 20
    with np.errstate(over="ignore"):
        for a in range(0, num_faces, chunk):
            b = min(a + chunk, num_faces)
            local = ptr[a : b + 1] - ptr[a]
            n = nodes[ptr[a] : ptr[b]]
            h1[a:b] = np.add.reduceat(r1[n], local[:-1])
            v = r2[n]
            h2[a:b] = np.add.reduceat(v * v, local[:-1])
    return h1, h2


def _ranges(starts, counts):
    """Concatenation of range(start, start + count) for every pair."""
    counts = np.asarray(counts, dtype=np.int64)
    ptr = np.concatenate([[0], np.cumsum(counts)])
    return np.repeat(np.asarray(starts, dtype=np.int64) - ptr[:-1], counts) + np.arange(
        ptr[-1]
    )


def _outward_faces(block):
    """(sizes, nodes) of the outward faces of every cell of a 3D block, and the
    number of faces of each cell."""
    if block.type.startswith("polyhedron"):
        faces = [f for cell in block.data for f in cell]
        counts = np.fromiter(
            (len(cell) for cell in block.data), np.int64, len(block.data)
        )
        sizes = np.fromiter((len(f) for f in faces), np.int64, len(faces))
        nodes = np.concatenate(faces) if faces else np.zeros(0, dtype=np.int64)
        return sizes, nodes, counts
    if block.type not in _OUTWARD:
        raise WriteError(
            f"Cannot write {block.type} cells with polyhedra (only linear cells)."
        )
    conn = np.asarray(block.data)
    table = _OUTWARD[block.type]
    sizes = np.tile([len(f) for f in table], len(conn)).astype(np.int64)
    nodes = np.concatenate([conn[:, f] for f in table], axis=1).ravel()
    counts = np.full(len(conn), len(table), dtype=np.int64)
    return sizes, nodes, counts


# -- writer ----------------------------------------------------------------------


def write(filename, mesh, compression=None, compression_opts=None):
    write_multi(filename, [mesh], ["Zone"], compression, compression_opts)


def write_multi(
    filename, meshes, mesh_names=None, compression=None, compression_opts=None
):
    """Write several meshes to one CGNS file: one base, one zone per mesh. The
    groups of all zones share the families of the base."""
    import h5py

    meshes = list(meshes)
    if mesh_names is None:
        mesh_names = [f"Zone{i + 1}" for i in range(len(meshes))]
    if len(mesh_names) != len(meshes):
        raise WriteError("One name per mesh is needed.")
    for mesh in meshes:
        points = np.asarray(mesh.points)
        if points.ndim != 2 or points.shape[1] not in (2, 3):
            raise WriteError(f"CGNS points must be 2D or 3D, got {points.shape}.")
        for block in mesh.cells:
            if block.type not in _CODES and not block.type.startswith(
                ("polygon", "polyhedron")
            ):
                raise WriteError(f"Cannot write {block.type} cells to CGNS.")
    phys_dim = max((np.asarray(m.points).shape[1] for m in meshes), default=3)
    cell_dim = max(
        (b.dim for m in meshes for b in m.cells),
        default=min((np.asarray(m.points).shape[1] for m in meshes), default=3),
    )

    # the CGNS library lists the children of a node in creation order, so every
    # group, the root included, must track it
    with h5py.File(filename, "w", track_order=True) as f:
        w = _Writer(f, compression, compression_opts)
        _set_string(f, "name", "HDF5 MotherNode", 33)
        _set_string(f, "label", "Root Node of HDF5 File", 33)
        _set_string(f, "type", "MT", 3)
        f.create_dataset(" format", data=np.frombuffer(b"IEEE_LITTLE_32\0", np.int8))
        hdf5 = f"HDF5 Version {h5py.version.hdf5_version}".encode()[:32]
        f.create_dataset(
            " hdf5version", data=np.frombuffer(hdf5.ljust(33, b"\0"), np.int8)
        )
        w.node(f, "CGNSLibraryVersion", "CGNSLibraryVersion_t", np.float32([VERSION]))
        base = w.node(f, "Base", "CGNSBase_t", np.int32([cell_dim, phys_dim]))

        families = _Families()
        zone_names = _Names()
        renamed = {}
        for mesh, mesh_name in zip(meshes, mesh_names):
            if len(mesh.points) == 0:
                continue  # a zone needs nodes: an empty mesh writes no zone
            renamed.update(
                _write_zone(
                    w, base, zone_names(mesh_name), mesh, cell_dim, phys_dim, families
                )
            )
        for fam in families.names.values():
            w.node(base, fam, "Family_t")
        renamed.update(zone_names.renamed)
        renamed.update(families.names_.renamed)
        if renamed:
            items = ", ".join(f"{a!r} -> {b!r}" for a, b in list(renamed.items())[:5])
            warn(f"CGNS names are limited to {_MAX_NAME} characters: {items}...")


class _Families:
    """Family_t names of a base, shared by its zones."""

    def __init__(self):
        self.names = {}
        self.names_ = _Names()

    def __call__(self, group):
        if group not in self.names:
            self.names[group] = self.names_(group)
        return self.names[group]


def _write_zone(w, base, zone_name, mesh, cell_dim, phys_dim, family):
    """Write one mesh as an unstructured zone; returns the renamed names."""
    points = np.asarray(mesh.points, dtype=float)
    if points.shape[1] < phys_dim:
        points = np.column_stack(
            [points, np.zeros((len(points), phys_dim - points.shape[1]))]
        )
    blocks = list(mesh.cells)
    cell_sets = mesh.cell_sets or {}
    has_polyhedra = any(b.type.startswith("polyhedron") for b in blocks)
    has_polygons = cell_dim == 2 and any(b.type.startswith("polygon") for b in blocks)
    if has_polyhedra:
        layout = _polyhedral_layout(blocks, cell_sets, cell_dim)
    else:
        layout = _standard_layout(blocks, cell_sets, cell_dim, has_polygons)

    zone = w.node(
        base, zone_name, "Zone_t", np.int64([[len(points), layout.num_cells, 0]])
    )
    w.node(zone, "ZoneType", "ZoneType_t", "Unstructured")
    grid = w.node(zone, "GridCoordinates", "GridCoordinates_t")
    for k, axis in enumerate("XYZ"[:phys_dim]):
        w.node(grid, f"Coordinate{axis}", "DataArray_t", points[:, k].copy())

    section_names = _Names()
    for sec in layout.sections:
        node = w.node(
            zone, section_names(sec["name"]), "Elements_t", np.int32([sec["code"], 0])
        )
        w.node(
            node,
            "ElementRange",
            "IndexRange_t",
            np.int64([sec["first"], sec["first"] + sec["count"] - 1]),
        )
        if "offsets" in sec:
            w.node(node, "ElementStartOffset", "DataArray_t", _index(sec["offsets"]))
        w.node(node, "ElementConnectivity", "DataArray_t", _index(sec["conn"]))
        for k, group in enumerate(sec.get("groups", ())):
            label = "FamilyName_t" if k == 0 else "AdditionalFamilyName_t"
            child = "FamilyName" if k == 0 else f"AdditionalFamilyName{k}"
            w.node(node, child, label, family(group))

    # groups of lower dimensional cells -> BCs
    bc_names = _Names()
    zone_bc = None
    location = {1: "EdgeCenter", 2: "FaceCenter"}
    for group, (dim, numbers) in layout.boundary_groups():
        if zone_bc is None:
            zone_bc = w.node(zone, "ZoneBC", "ZoneBC_t")
        bc = w.node(zone_bc, bc_names(group), "BC_t", "FamilySpecified")
        w.node(bc, "PointList", "IndexArray_t", _index(numbers)[None, :])
        w.node(bc, "GridLocation", "GridLocation_t", location.get(dim, "Vertex"))
        w.node(bc, "FamilyName", "FamilyName_t", family(group))

    # groups of nodes -> ZoneSubRegion
    region_names = _Names()
    for group, nodes in (mesh.point_sets or {}).items():
        nodes = np.asarray(nodes, dtype=np.int64)
        if not len(nodes):
            continue
        region = w.node(zone, region_names(group), "ZoneSubRegion_t", np.int32([1]))
        w.node(region, "GridLocation", "GridLocation_t", "Vertex")
        w.node(region, "PointList", "IndexArray_t", _index(nodes + 1)[None, :])

    _write_solutions(w, zone, mesh, points, layout)
    return {**section_names.renamed, **bc_names.renamed, **region_names.renamed}


def _expand(name, values):
    """Scalar arrays of a field: vectors are split into X/Y/Z (or _<i>)."""
    values = np.asarray(values)
    if values.ndim == 1:
        return [(name, values)]
    values = values.reshape(len(values), -1)
    suffixes = "XYZ" if values.shape[1] <= 3 else None
    return [
        (f"{name}{suffixes[i]}" if suffixes else f"{name}_{i}", values[:, i])
        for i in range(values.shape[1])
    ]


def _write_solutions(w, zone, mesh, points, layout):
    names = _Names()
    point_fields = []
    for name, values in (mesh.point_data or {}).items():
        values = np.asarray(values)
        if values.dtype.kind not in "fiub" or len(values) != len(points):
            continue
        point_fields += _expand(name, values)
    if point_fields:
        sol = w.node(zone, "FlowSolution", "FlowSolution_t")
        w.node(sol, "GridLocation", "GridLocation_t", "Vertex")
        for name, values in point_fields:
            w.node(sol, names(name), "DataArray_t", values.astype(np.float64))

    cell_fields, skipped = [], []
    for name, per_block in (mesh.cell_data or {}).items():
        values = layout.volume_values(per_block)
        if values is None:
            skipped.append(name)
            continue
        cell_fields += _expand(name, values)
    if skipped:
        warn(f"Cell data not on the volume cells, not written to CGNS: {skipped}.")
    if cell_fields:
        names = _Names()
        sol = w.node(zone, "FlowSolutionCells", "FlowSolution_t")
        w.node(sol, "GridLocation", "GridLocation_t", "CellCenter")
        for name, values in cell_fields:
            w.node(sol, names(name), "DataArray_t", values.astype(np.float64))


class _Layout:
    """Element sections of a mesh, and the CGNS element number of every cell."""

    def __init__(self, blocks, cell_sets, cell_dim):
        self.blocks = blocks
        self.cell_sets = cell_sets
        self.cell_dim = cell_dim
        self.sections = []
        self.numbers = [None] * len(blocks)  # block -> element number of each cell
        self.next = 1
        self.num_cells = 0

    def add(self, name, code, conn, count, block=None, local=None, **extra):
        sec = {"name": name, "code": code, "conn": conn, "count": count}
        sec["first"] = self.next
        sec.update(extra)
        self.sections.append(sec)
        if block is not None:
            if self.numbers[block] is None:
                self.numbers[block] = np.zeros(len(self.blocks[block].data), np.int64)
            self.numbers[block][local] = self.next + np.arange(count)
        self.next += count
        return sec

    def boundary_groups(self):
        """[(group, (dim, element numbers))] of the cell sets on lower
        dimensional blocks, one entry per group and dimension."""
        out = []
        for group, per_block in self.cell_sets.items():
            by_dim = {}
            for b, local in enumerate(per_block):
                if b >= len(self.blocks) or not len(local):
                    continue
                dim = self.blocks[b].dim
                if dim == self.cell_dim or self.numbers[b] is None:
                    continue
                by_dim.setdefault(dim, []).append(self.numbers[b][np.asarray(local)])
            for k, (dim, nums) in enumerate(sorted(by_dim.items(), reverse=True)):
                name = group if k == 0 else f"{group}_{dim}d"
                out.append((name, (dim, np.sort(np.concatenate(nums)))))
        return out

    def volume_values(self, per_block):
        """Values of a cell data field on the volume cells, in element order, or
        None when the field does not cover them."""
        values = np.full(self.num_cells, np.nan)
        covered = np.zeros(self.num_cells, dtype=bool)
        for b, block in enumerate(self.blocks):
            if block.dim != self.cell_dim or b >= len(per_block):
                continue
            data = np.asarray(per_block[b])
            if data.dtype.kind not in "fiub" or len(data) != len(block.data):
                return None
            data = data.reshape(len(data), -1)
            if data.shape[1] != 1:
                if values.ndim == 1:
                    values = np.full((self.num_cells, data.shape[1]), np.nan)
                values[self.numbers[b] - self.cell_first] = data
            else:
                values[self.numbers[b] - self.cell_first] = data[:, 0]
            covered[self.numbers[b] - self.cell_first] = True
        return values if covered.any() else None


def _standard_layout(blocks, cell_sets, cell_dim, polygons_2d):
    layout = _Layout(blocks, cell_sets, cell_dim)
    order = sorted(range(len(blocks)), key=lambda b: -blocks[b].dim)
    layout.cell_first = 1
    if polygons_2d:
        # all 2D cells as NGON_n, split by group
        cells = [b for b in order if blocks[b].dim == 2]
        parts = {}
        for b in cells:
            combos = _groups_of_block(cell_sets, b, len(blocks[b].data))
            for combo, local in _split_by_groups(combos).items():
                parts.setdefault(combo, []).append((b, local))
        for combo, items in parts.items():
            polygons, owners = [], []
            for b, local in items:
                data = blocks[b].data
                polygons += [np.asarray(data[i], dtype=np.int64) for i in local]
                owners.append((b, local))
            sizes = np.array([len(p) for p in polygons])
            conn = np.concatenate(polygons) + 1
            offsets = np.concatenate([[0], np.cumsum(sizes)])
            name = "_".join(combo) or "Polygons"
            sec = layout.add(name, _NGON_n, conn, len(polygons), offsets=offsets)
            sec["groups"] = combo
            start = sec["first"]
            for b, local in owners:
                if layout.numbers[b] is None:
                    layout.numbers[b] = np.zeros(len(blocks[b].data), np.int64)
                layout.numbers[b][local] = start + np.arange(len(local))
                start += len(local)
        layout.num_cells = layout.next - 1
        order = [b for b in order if blocks[b].dim != 2]

    for b in order:
        block = blocks[b]
        if block.dim == cell_dim:
            combos = _groups_of_block(cell_sets, b, len(block.data))
            parts = _split_by_groups(combos)
        else:
            parts = {(): np.arange(len(block.data))}
        for combo, local in parts.items():
            code, conn = _cgns_connectivity(block, local)
            name = "_".join(combo + (block.type,))
            sec = layout.add(name, code, conn, len(local), block=b, local=local)
            sec["groups"] = combo
        if block.dim == cell_dim:
            layout.num_cells = layout.next - 1
    return layout


def _cgns_connectivity(block, local):
    """CGNS code and flat 1-based connectivity of the cells local of a block."""
    from ._elements import PERMUTATIONS

    if block.type.startswith("polygon"):
        raise WriteError("Polygons are only written in 2D meshes or with polyhedra.")
    conn = np.asarray(block.data, dtype=np.int64)[local]
    perm = PERMUTATIONS.get(block.type)
    if perm is not None:
        # inverse of the CGNS -> meshlane permutation
        inverse = np.argsort(perm)
        conn = conn[:, inverse]
    return _CODES[block.type], (conn + 1).ravel()


def _polyhedral_layout(blocks, cell_sets, cell_dim):
    """All 3D cells as NFACE_n, all faces once as NGON_n."""
    if cell_dim != 3:
        raise WriteError("Polyhedra need a 3D mesh.")
    layout = _Layout(blocks, cell_sets, cell_dim)
    volume = [b for b, block in enumerate(blocks) if block.dim == 3]
    lower = [b for b, block in enumerate(blocks) if block.dim == 2]
    for b, block in enumerate(blocks):
        if block.dim < 2:
            warn(f"{block.type} cells are not written with polyhedra.")

    # half-faces of all cells (block by block) then the faces of the 2D blocks,
    # which need an NGON_n element too
    sizes, nodes, counts = [], [], []
    block_first = {}
    num_cells = 0
    for b in volume:
        s, n, c = _outward_faces(blocks[b])
        sizes.append(s)
        nodes.append(n)
        counts.append(c)
        block_first[b] = num_cells
        num_cells += len(c)
    counts = np.concatenate(counts) if counts else np.zeros(0, np.int64)
    num_half = int(counts.sum())
    for b in lower:
        data = blocks[b].data
        if isinstance(data, np.ndarray):
            sizes.append(np.full(len(data), data.shape[1], dtype=np.int64))
            nodes.append(data.ravel())
        else:
            sizes.append(np.array([len(x) for x in data], dtype=np.int64))
            nodes.append(np.concatenate([np.asarray(x) for x in data]))
    sizes = np.concatenate(sizes)
    nodes = np.concatenate(nodes)
    if nodes.max(initial=0) < 2**31 - 1:
        nodes = nodes.astype(np.int32)
    ptr = np.concatenate([[0], np.cumsum(sizes)])

    # find the faces: equal size and hashes; stable sort, so the first of a run
    # is the first use, which defines the face and its orientation
    h1, h2 = _face_keys(ptr, nodes)
    order = np.lexsort((sizes, h2, h1))
    new = np.ones(len(order), dtype=bool)
    new[1:] = (
        (h1[order[1:]] != h1[order[:-1]])
        | (h2[order[1:]] != h2[order[:-1]])
        | (sizes[order[1:]] != sizes[order[:-1]])
    )
    del h1, h2
    # a face is shared by at most two cells: further cell uses of the same nodes
    # (malformed cells listing a face twice) each get their own face
    start = np.flatnonzero(new)
    position = np.arange(len(order)) - np.repeat(
        start, np.diff(np.append(start, len(order)))
    )
    extra = (position >= 2) & (order < num_half)
    if extra.any():
        warn(f"{int(extra.sum())} face(s) used by more than two cells, written apart.")
        new |= extra
    del start, position, extra
    group = np.cumsum(new) - 1
    first = order[new]  # defining occurrence of each face
    # number the faces in order of first use: cells first, then 2D blocks
    rank = np.empty(len(first), dtype=np.int64)
    rank[np.argsort(first, kind="stable")] = np.arange(len(first))
    face_of = np.empty(len(order), dtype=np.int64)
    face_of[order] = rank[group]
    owner = np.empty(len(first), dtype=np.int64)
    owner[rank] = first
    del order, new, group, first, rank

    face_sizes = sizes[owner]
    face_ptr = np.concatenate([[0], np.cumsum(face_sizes)])
    ngon_conn = nodes[_ranges(ptr[owner], face_sizes)] + 1
    layout.add("NGonElements", _NGON_n, ngon_conn, len(owner), offsets=face_ptr)
    del ngon_conn

    # NFACE_n: positive sign for the cell that defines a face, negative otherwise
    half = np.arange(num_half)
    refs = face_of[:num_half] + 1
    refs[owner[face_of[:num_half]] != half] *= -1
    del half
    cell_ptr = np.concatenate([[0], np.cumsum(counts)])

    # one NFACE_n section per group combination
    parts = {}
    for b in volume:
        combos = _groups_of_block(cell_sets, b, len(blocks[b].data))
        for combo, local in _split_by_groups(combos).items():
            parts.setdefault(combo, []).append((b, local))
    layout.cell_first = layout.next
    for combo in sorted(parts):
        cells = np.concatenate([block_first[b] + local for b, local in parts[combo]])
        offsets = np.concatenate([[0], np.cumsum(counts[cells])])
        conn = refs[_ranges(cell_ptr[cells], counts[cells])]
        name = "_".join(combo) or "NFaceElements"
        sec = layout.add(name, _NFACE_n, conn, len(cells), offsets=offsets)
        sec["groups"] = combo
        number = sec["first"]
        for b, local in parts[combo]:
            if layout.numbers[b] is None:
                layout.numbers[b] = np.zeros(len(blocks[b].data), np.int64)
            layout.numbers[b][local] = number + np.arange(len(local))
            number += len(local)
    layout.num_cells = num_cells

    # 2D blocks: their faces are NGON_n elements
    k = num_half
    for b in lower:
        n = len(blocks[b].data)
        layout.numbers[b] = face_of[k : k + n] + 1
        k += n
    return layout

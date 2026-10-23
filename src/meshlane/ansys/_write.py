"""Write Ansys Fluent mesh files (.msh), in the face-based layout Fluent reads.

A Fluent mesh lists the nodes, every face once with the two cells on either side
(c0, c1) and the cell types; zones group the cells and the faces, and are named
in section 39. The right-hand normal of a face points to c0: an interior face is
written as seen from c1 (normal from c1 to c0), a boundary face with c0 its cell
and c1 = 0 (normal pointing into the domain), as Fluent and OpenFOAM's
foamMeshToFluent do.

- cell zones: one per combination of the groups (cell sets) of the volume cells,
  "fluid" otherwise;
- face zones: the interior faces ("default-interior", or the group they belong
  to), then one zone per group of boundary faces, and "wall" for the boundary
  faces in no group; zone types (wall, velocity-inlet, ... fluid, solid) can be
  chosen by name with ``zone_types``.
"""

import re

import numpy as np

from ..__about__ import __version__
from .._common import warn
from .._exceptions import WriteError

# Fluent face zone types (bc-type of section 13)
FACE_TYPES = {
    "interior": 2,
    "wall": 3,
    "pressure-inlet": 4,
    "inlet-vent": 4,
    "intake-fan": 4,
    "pressure-outlet": 5,
    "exhaust-fan": 5,
    "outlet-vent": 5,
    "symmetry": 7,
    "periodic-shadow": 8,
    "pressure-far-field": 9,
    "velocity-inlet": 10,
    "periodic": 12,
    "fan": 14,
    "porous-jump": 14,
    "radiator": 14,
    "mass-flow-inlet": 20,
    "interface": 24,
    "outflow": 36,
    "axis": 37,
}
CELL_TYPES = ("fluid", "solid")

# Fluent element types of the cells (section 12)
_ELEMENT = {
    "triangle": 1,
    "tetra": 2,
    "quad": 3,
    "hexahedron": 4,
    "pyramid": 5,
    "wedge": 6,
}
_POLYHEDRON, _POLYGON_2D = 7, 7

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
# quadratic cells are written with their corner nodes
_LINEAR = {
    "line3": ("line", 2),
    "triangle6": ("triangle", 3),
    "triangle7": ("triangle", 3),
    "quad8": ("quad", 4),
    "quad9": ("quad", 4),
    "tetra10": ("tetra", 4),
    "pyramid13": ("pyramid", 5),
    "pyramid14": ("pyramid", 5),
    "wedge15": ("wedge", 6),
    "wedge18": ("wedge", 6),
    "hexahedron20": ("hexahedron", 8),
    "hexahedron24": ("hexahedron", 8),
    "hexahedron27": ("hexahedron", 8),
}
_CHUNK = 1 << 20


# -- mesh preparation ------------------------------------------------------------


def _linear(block):
    """(cell type, data) of a block, quadratic cells reduced to their corners."""
    if block.type in _LINEAR:
        kind, k = _LINEAR[block.type]
        return kind, np.asarray(block.data)[:, :k]
    return block.type, block.data


def _ranges(starts, counts):
    """Concatenation of range(start, start + count) for every pair."""
    counts = np.asarray(counts, dtype=np.int64)
    ptr = np.concatenate([[0], np.cumsum(counts)])
    return np.repeat(np.asarray(starts, dtype=np.int64) - ptr[:-1], counts) + np.arange(
        ptr[-1]
    )


def _cell_faces(kind, data, dim):
    """(sizes, nodes, counts): the outward faces (3D) or counterclockwise edges
    (2D) of the cells of a block, as CSR, and the number of faces of each cell."""
    if dim == 3:
        if kind.startswith("polyhedron"):
            faces = [f for cell in data for f in cell]
            counts = np.fromiter((len(c) for c in data), np.int64, len(data))
            sizes = np.fromiter((len(f) for f in faces), np.int64, len(faces))
            nodes = np.concatenate(faces) if faces else np.zeros(0, np.int64)
            return sizes, nodes, counts
        if kind not in _OUTWARD:
            raise WriteError(f"Cannot write {kind} cells to Fluent.")
        conn = np.asarray(data)
        table = _OUTWARD[kind]
        sizes = np.tile([len(f) for f in table], len(conn)).astype(np.int64)
        nodes = np.concatenate([conn[:, f] for f in table], axis=1).ravel()
        return sizes, nodes, np.full(len(conn), len(table), dtype=np.int64)
    # 2D: the edges of each polygon, in its node order
    if isinstance(data, np.ndarray) and data.ndim == 2:
        k = data.shape[1]
        edges = np.stack([data, np.roll(data, -1, axis=1)], axis=2)
        return np.full(len(data) * k, 2, np.int64), edges.ravel(), np.full(len(data), k)
    polys = [np.asarray(p) for p in data]
    counts = np.array([len(p) for p in polys], dtype=np.int64)
    edges = (
        np.concatenate([np.stack([p, np.roll(p, -1)], axis=1).ravel() for p in polys])
        if polys
        else np.zeros(0, np.int64)
    )
    return np.full(int(counts.sum()), 2, np.int64), edges, counts


def _face_geometry(points, sizes, nodes):
    """Center and (unnormalized) normal of every face given as CSR; in 2D a face
    is an edge a -> b and its normal is (dy, -dx)."""
    ptr = np.concatenate([[0], np.cumsum(sizes)])
    num = len(sizes)
    dim = points.shape[1]
    centers = np.empty((num, dim))
    normals = np.empty((num, dim))
    for a in range(0, num, _CHUNK):
        b = min(a + _CHUNK, num)
        local = ptr[a : b + 1] - ptr[a]
        p = points[nodes[ptr[a] : ptr[b]]]
        centers[a:b] = np.add.reduceat(p, local[:-1], axis=0) / sizes[a:b, None]
        if dim == 2:
            d = p[local[:-1] + 1] - p[local[:-1]]
            normals[a:b] = np.column_stack([d[:, 1], -d[:, 0]])
        else:
            nxt = np.arange(1, len(p) + 1)
            nxt[local[1:] - 1] = local[:-1]
            normals[a:b] = np.add.reduceat(np.cross(p, p[nxt]), local[:-1], axis=0)
    return centers, normals


def _edge_sums(sizes, nodes, counts):
    """Per cell, the sum over its faces' edges u -> v of H[u] G[v] - H[v] G[u]
    (uint64, wrapping): exactly zero when every edge is walked once each way,
    that is when the faces of the cell are consistently oriented."""
    rng = np.random.default_rng(1)
    high = int(nodes.max(initial=0)) + 1
    h = rng.integers(0, 2**63, size=high, dtype=np.uint64)
    g = rng.integers(0, 2**63, size=high, dtype=np.uint64)
    ptr = np.concatenate([[0], np.cumsum(sizes)])
    nxt = np.arange(1, len(nodes) + 1)
    nxt[ptr[1:] - 1] = ptr[:-1]
    u, v = nodes, nodes[nxt]
    with np.errstate(over="ignore"):
        per_face = np.add.reduceat(h[u] * g[v] - h[v] * g[u], ptr[:-1])
        cell_ptr = np.concatenate([[0], np.cumsum(counts)])
        return np.add.reduceat(per_face, cell_ptr[:-1])


def _orient_outward(points, sizes, nodes, counts):
    """Orient the faces of every cell outward.

    A cell whose faces are consistently oriented (every edge walked once each
    way) is kept, or reversed as a whole when its volume (divergence theorem) is
    negative; the faces of the other cells (polyhedra given with arbitrary face
    orientations) are oriented one by one, away from the cell center."""
    if len(counts) == 0:
        return nodes, 0
    centers, normals = _face_geometry(points, sizes, nodes)
    cell = np.repeat(np.arange(len(counts)), counts)
    ncell = len(counts)
    cc = (
        np.column_stack(
            [
                np.bincount(cell, weights=centers[:, d], minlength=ncell)
                for d in range(points.shape[1])
            ]
        )
        / np.maximum(counts, 1)[:, None]
    )
    flux = np.einsum("ij,ij->i", normals, centers - cc[cell])
    volume = np.bincount(cell, weights=flux, minlength=ncell)
    if points.shape[1] == 3:
        consistent = _edge_sums(sizes, nodes, counts) == 0
    else:
        consistent = np.ones(ncell, dtype=bool)  # 2D cells: edges of a polygon
    face_flip = np.where(consistent[cell], (volume < 0)[cell], flux < 0)
    changed = int(
        np.count_nonzero(np.bincount(cell, weights=face_flip, minlength=ncell))
    )
    if not face_flip.any():
        return nodes, 0
    reverse = np.repeat(face_flip, sizes)
    ptr = np.concatenate([[0], np.cumsum(sizes)])
    within = np.arange(len(nodes)) - np.repeat(ptr[:-1], sizes)
    flipped = np.repeat(ptr[:-1] + sizes - 1, sizes) - within
    index = np.where(reverse, flipped, np.arange(len(nodes)))
    return nodes[index], changed


def _face_keys(sizes, nodes):
    """Two order-independent hashes of every face (sums of random values of its
    nodes); with the size, they find the faces shared by two cells."""
    rng = np.random.default_rng(0)
    high = int(nodes.max(initial=0)) + 1
    r1 = rng.integers(0, 2**63, size=high, dtype=np.uint64)
    r2 = rng.integers(0, 2**63, size=high, dtype=np.uint64)
    ptr = np.concatenate([[0], np.cumsum(sizes)])
    num = len(sizes)
    h1 = np.empty(num, dtype=np.uint64)
    h2 = np.empty(num, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for a in range(0, num, _CHUNK):
            b = min(a + _CHUNK, num)
            local = ptr[a : b + 1] - ptr[a]
            n = nodes[ptr[a] : ptr[b]]
            h1[a:b] = np.add.reduceat(r1[n], local[:-1])
            v = r2[n]
            h2[a:b] = np.add.reduceat(v * v, local[:-1])
    return h1, h2


def _combos(cell_sets, block, size, dims_ok):
    """Group combination (tuple of names) of each cell of a block."""
    names = sorted(
        name
        for name, per_block in cell_sets.items()
        if block < len(per_block) and len(per_block[block]) and dims_ok
    )
    if not names:
        return [()] * size
    flags = np.zeros((size, len(names)), dtype=bool)
    for j, name in enumerate(names):
        flags[np.asarray(cell_sets[name][block], dtype=np.int64), j] = True
    keys, inverse = np.unique(flags, axis=0, return_inverse=True)
    labels = [tuple(n for n, f in zip(names, row) if f) for row in keys]
    return [labels[i] for i in inverse.ravel()]


class _Names:
    """Fluent zone names: no blanks or brackets, unique."""

    def __init__(self):
        self.used = set()

    def __call__(self, name):
        clean = re.sub(r"[\s()\"']+", "_", str(name)).strip("_") or "zone"
        out, k = clean, 1
        while out in self.used:
            out = f"{clean}-{k}"
            k += 1
        self.used.add(out)
        return out


# -- writer ----------------------------------------------------------------------


def write(filename, mesh, binary=True, zone_types=None):
    """Write a Fluent mesh file.

    zone_types maps zone (group) names to Fluent zone types, e.g.
    ``{"inlet": "velocity-inlet", "outlet": "pressure-outlet", "plate": "solid"}``;
    boundary zones are walls and cell zones fluid otherwise.
    """
    zone_types = dict(zone_types or {})
    for name, kind in zone_types.items():
        if kind not in FACE_TYPES and kind not in CELL_TYPES:
            raise WriteError(f"Unknown Fluent zone type {kind!r} for zone {name!r}.")

    points = np.asarray(mesh.points, dtype=float)
    blocks = list(mesh.cells)
    if len(points) == 0 or not blocks:
        raise WriteError("Fluent meshes need nodes and cells.")
    dim = max(b.dim for b in blocks)
    if dim not in (2, 3):
        raise WriteError(f"Fluent meshes are 2D or 3D, got {dim}D cells.")
    if dim == 2 and points.shape[1] == 3:
        if np.any(points[:, 2] != 0.0):
            dim = 3  # a surface in space: faces only, no cells
        else:
            points = points[:, :2]
    if points.shape[1] != dim:
        raise WriteError(f"{dim}D cells need {dim}D points, got {points.shape[1]}D.")
    linearized = sorted({b.type for b in blocks if b.type in _LINEAR})
    if linearized:
        warn(
            f"Fluent meshes are linear: {', '.join(linearized)} written with their corners."
        )
    skipped = sorted({b.type for b in blocks if b.dim < dim - 1})
    if skipped:
        warn(f"{', '.join(skipped)} cells are not written to Fluent.")
    cell_sets = mesh.cell_sets or {}

    layout = _layout(points, blocks, cell_sets, dim, zone_types)
    with open(filename, "wb") as fh:
        _write_file(fh, points, layout, binary)


def _layout(points, blocks, cell_sets, dim, zone_types):
    """Cells (by zone), faces (by zone) and zone definitions."""
    names = _Names()
    volume = [b for b, blk in enumerate(blocks) if blk.dim == dim]
    lower = [b for b, blk in enumerate(blocks) if blk.dim == dim - 1]

    # cell zones: group combination of every volume cell
    parts = {}
    for b in volume:
        combos = _combos(cell_sets, b, len(blocks[b].data), True)
        for i, combo in enumerate(combos):
            parts.setdefault(combo, {}).setdefault(b, []).append(i)
    cell_zones = []
    for combo in sorted(parts, key=lambda c: (c != (), c)):
        name = names("_".join(combo) or "fluid")
        kind = next(
            (zone_types[g] for g in combo if zone_types.get(g) in CELL_TYPES), None
        )
        kind = zone_types.get(name, kind) or "fluid"
        cell_zones.append({"name": name, "type": kind, "cells": parts[combo]})

    # half-faces of the cells, zone by zone (cells are numbered in this order)
    sizes, nodes, counts, elements = [], [], [], []
    for zone in cell_zones:
        start = sum(len(c) for c in elements)
        for b, local in zone["cells"].items():
            kind, data = _linear(blocks[b])
            local = np.asarray(local)
            sub = (
                data[local]
                if isinstance(data, np.ndarray)
                else [data[i] for i in local]
            )
            s, n, c = _cell_faces(kind, sub, dim)
            sizes.append(s)
            nodes.append(n)
            counts.append(c)
            code = _ELEMENT.get(kind, _POLYHEDRON if dim == 3 else _POLYGON_2D)
            if dim == 2 and kind not in _ELEMENT:
                code = 3 if len(sub) and len(sub[0]) == 4 else _POLYGON_2D
            elements.append(np.full(len(local), code, dtype=np.int32))
        zone["range"] = (start, sum(len(c) for c in elements))
    if not sizes:  # surface mesh: no cells
        sizes, nodes, counts = ([np.zeros(0, np.int64)] for _ in range(3))
        elements = [np.zeros(0, np.int32)]
    sizes = np.concatenate(sizes)
    nodes = np.concatenate(nodes)
    counts = np.concatenate(counts)
    elements = np.concatenate(elements)
    nodes, flipped = _orient_outward(points, sizes, nodes, counts)
    if flipped:
        warn(
            f"{flipped} cell(s) with inward faces written with their faces turned outward."
        )
    num_half = len(sizes)

    # boundary faces and edges of the lower blocks, for their groups
    low_sizes, low_nodes, low_group = [], [], []
    for b in lower:
        kind, data = _linear(blocks[b])
        combos = _combos(cell_sets, b, len(data), True)
        if isinstance(data, np.ndarray) and data.ndim == 2:
            low_sizes.append(np.full(len(data), data.shape[1], np.int64))
            low_nodes.append(data.ravel())
        else:
            low_sizes.append(np.array([len(x) for x in data], np.int64))
            low_nodes.append(np.concatenate([np.asarray(x) for x in data]))
        low_group += combos
    all_sizes = np.concatenate([sizes] + low_sizes)
    all_nodes = np.concatenate([nodes] + low_nodes).astype(np.int64)

    # unique faces: equal size and hashes; the first use defines the face
    h1, h2 = _face_keys(all_sizes, all_nodes)
    order = np.lexsort((all_sizes, h2, h1))
    new = np.ones(len(order), dtype=bool)
    new[1:] = (
        (h1[order[1:]] != h1[order[:-1]])
        | (h2[order[1:]] != h2[order[:-1]])
        | (all_sizes[order[1:]] != all_sizes[order[:-1]])
    )
    # a face has at most two cells: further uses (malformed cells) are apart
    start = np.flatnonzero(new)
    position = np.arange(len(order)) - np.repeat(
        start, np.diff(np.append(start, len(order)))
    )
    extra = (position >= 2) & (order < num_half)
    if extra.any():
        warn(f"{int(extra.sum())} face(s) used by more than two cells, written apart.")
        new |= extra
    del h1, h2, start, position, extra
    group = np.cumsum(new) - 1
    face_of = np.empty(len(order), dtype=np.int64)
    face_of[order] = group
    first = order[new]  # first occurrence of each face
    num_faces = len(first)

    cell_of_half = np.repeat(np.arange(len(counts)), counts)
    owner = np.full(num_faces, -1, dtype=np.int64)
    neighbour = np.full(num_faces, -1, dtype=np.int64)
    half_faces = face_of[:num_half]
    # the first use of a face (in cell order) is its owner
    is_first = np.zeros(num_half, dtype=bool)
    is_first[np.unique(half_faces, return_index=True)[1]] = True
    owner[half_faces[is_first]] = cell_of_half[is_first]
    neighbour[half_faces[~is_first]] = cell_of_half[~is_first]

    # group of every face (from the lower blocks)
    face_group = {}
    for k, combo in enumerate(low_group):
        if combo:
            face_group.setdefault(int(face_of[num_half + k]), combo)

    # face zones: interior first, then boundary; faces only on lower blocks
    # (no cell) are boundary faces of a surface
    zone_of = {}
    zone_list = []

    def zone_id(combo, interior):
        key = (combo, interior)
        if key not in zone_of:
            if combo:
                name = "_".join(combo)
            else:
                name = "default-interior" if interior else "wall"
            name = names(name)
            kind = zone_types.get(name) or next(
                (zone_types[g] for g in combo if zone_types.get(g) in FACE_TYPES), None
            )
            # interior faces in a group are a two-sided wall (Fluent adds its
            # shadow zone): an interface between regions, kept as a group
            default = "interior" if interior and not combo else "wall"
            kind = kind or default
            if kind in CELL_TYPES:
                kind = default
            zone_of[key] = len(zone_list)
            zone_list.append({"name": name, "type": kind, "interior": interior})
        return zone_of[key]

    interior = neighbour >= 0
    face_zone = np.zeros(num_faces, dtype=np.int64)
    combos_by_face = np.full(num_faces, -1, dtype=np.int64)
    combo_index = {}
    for f, combo in face_group.items():
        combos_by_face[f] = combo_index.setdefault(combo, len(combo_index))
    combo_list = list(combo_index)
    # zones in a deterministic order: interior (default, then groups), boundary
    for is_interior in (True, False):
        sel = interior == is_interior
        if not sel.any():
            continue
        ids = np.unique(combos_by_face[sel])
        for c in ids:
            combo = combo_list[c] if c >= 0 else ()
            z = zone_id(combo, is_interior)
            face_zone[sel & (combos_by_face == c)] = z

    # face nodes, written as Fluent wants them: normal toward c0
    face_sizes = all_sizes[first]
    face_ptr = np.concatenate([[0], np.cumsum(face_sizes)])
    all_ptr = np.concatenate([[0], np.cumsum(all_sizes)])
    face_nodes = all_nodes[_ranges(all_ptr[first], face_sizes)]
    # interior: owner's outward face (normal from owner = c1 to neighbour = c0);
    # boundary: reversed (normal into the cell = c0). In 2D, Fluent's normal of
    # an edge a -> b is z x (b - a), the opposite of the outward normal of a
    # counterclockwise cell's edge: every edge is reversed.
    boundary = ~interior & (owner >= 0)
    reverse = np.repeat(boundary != (dim == 2), face_sizes)
    within = np.arange(len(face_nodes)) - np.repeat(face_ptr[:-1], face_sizes)
    flipped_index = np.repeat(face_ptr[:-1] + face_sizes - 1, face_sizes) - within
    face_nodes = face_nodes[
        np.where(reverse, flipped_index, np.arange(len(face_nodes)))
    ]
    c0 = np.where(interior, neighbour, owner) + 1  # 1-based, 0: no cell
    c1 = np.where(interior, owner, -1) + 1

    return {
        "dim": dim,
        "cell_zones": cell_zones,
        "elements": elements,
        "face_zones": zone_list,
        "face_zone": face_zone,
        "face_sizes": face_sizes,
        "face_ptr": face_ptr,
        "face_nodes": face_nodes + 1,
        "c0": c0,
        "c1": c1,
    }


# -- file output -----------------------------------------------------------------


def _hex_rows(rows):
    return "\n".join(" ".join(f"{v:x}" for v in row) for row in rows)


def _write_file(fh, points, layout, binary):
    dim = layout["dim"]
    num_points = len(points)
    cell_zones = layout["cell_zones"]
    face_zones = layout["face_zones"]
    num_cells = len(layout["elements"])
    num_faces = len(layout["c0"])
    # zone ids: nodes 1, cell zones next, then face zones
    node_zone = 1
    cell_ids = [2 + i for i in range(len(cell_zones))]
    face_ids = [2 + len(cell_zones) + i for i in range(len(face_zones))]

    w = fh.write
    w(f'(0 "Fluent mesh written by meshlane {__version__}")\n'.encode())
    w(f"(2 {dim})\n".encode())
    w(f"(10 (0 1 {num_points:x} 0 {dim}))\n".encode())
    w(f"(12 (0 1 {num_cells:x} 0 0))\n".encode())
    w(f"(13 (0 1 {num_faces:x} 0 0))\n".encode())

    # nodes
    if binary:
        w(f"(3010 ({node_zone:x} 1 {num_points:x} 1 {dim})(".encode())
        w(np.ascontiguousarray(points, dtype="<f8").tobytes())
        w(b")\nEnd of Binary Section 3010)\n")
    else:
        w(f"(10 ({node_zone:x} 1 {num_points:x} 1 {dim})(\n".encode())
        np.savetxt(fh, points, fmt="%.17g")
        w(b"))\n")

    # faces, zone by zone
    face_zone = layout["face_zone"]
    order = np.argsort(face_zone, kind="stable")
    sizes = layout["face_sizes"]
    ptr = layout["face_ptr"]
    nodes = layout["face_nodes"]
    c0, c1 = layout["c0"], layout["c1"]
    first = 1
    for z, zone in enumerate(face_zones):
        faces = order[face_zone[order] == z]
        n = len(faces)
        if n == 0:
            continue
        zone["range"] = (first, first + n - 1)
        zsizes = sizes[faces]
        uniform = len(np.unique(zsizes)) == 1 and zsizes[0] in (2, 3, 4)
        face_type = int(zsizes[0]) if uniform else 0
        bc = FACE_TYPES[zone["type"]]
        header = f"{face_ids[z]:x} {first:x} {first + n - 1:x} {bc:x} {face_type}"
        zn = nodes[_ranges(ptr[faces], zsizes)]
        if uniform:
            k = face_type
            rows = np.column_stack([zn.reshape(n, k), c0[faces], c1[faces]])
            flat = rows.ravel()
        else:
            # mixed zone: each face preceded by its number of nodes
            row_ptr = np.concatenate([[0], np.cumsum(zsizes + 3)])
            flat = np.empty(row_ptr[-1], dtype=np.int64)
            flat[row_ptr[:-1]] = zsizes
            body = _ranges(row_ptr[:-1] + 1, zsizes)
            flat[body] = zn
            flat[row_ptr[1:] - 2] = c0[faces]
            flat[row_ptr[1:] - 1] = c1[faces]
        if binary:
            w(f"(3013 ({header})(".encode())
            w(flat.astype("<i4").tobytes())
            w(b")\nEnd of Binary Section 3013)\n")
        else:
            w(f"(13 ({header})(\n".encode())
            if uniform:
                np.savetxt(fh, rows, fmt="%x")
            else:
                lines = np.split(flat, row_ptr[1:-1])
                w((_hex_rows(lines) + "\n").encode())
            w(b"))\n")
        first += n

    # cells, zone by zone
    elements = layout["elements"]
    for zone, zid in zip(cell_zones, cell_ids):
        a, b = zone["range"]
        types = elements[a:b]
        uniform = len(np.unique(types)) == 1
        element_type = int(types[0]) if uniform and types[0] != _POLYHEDRON else 0
        if uniform and types[0] == _POLYHEDRON:
            element_type = _POLYHEDRON
        header = f"{zid:x} {a + 1:x} {b:x} 1 {element_type:x}"
        if element_type != 0:
            w(f"(12 ({header}))\n".encode())
        elif binary:
            w(f"(3012 ({header})(".encode())
            w(types.astype("<i4").tobytes())
            w(b")\nEnd of Binary Section 3012)\n")
        else:
            w(f"(12 ({header})(\n".encode())
            text = " ".join(f"{t:x}" for t in types)
            w(text.encode() + b"\n))\n")

    # zone names
    for zone, zid in zip(cell_zones, cell_ids):
        w(f"(39 ({zid} {zone['type']} {zone['name']})())\n".encode())
    for zone, zid in zip(face_zones, face_ids):
        if "range" in zone:
            w(f"(39 ({zid} {zone['type']} {zone['name']})())\n".encode())

"""Structured CGNS zones, converted to unstructured cells.

A structured zone is a block of ni x nj x nk nodes (i varying fastest, as in the
HDF5 arrays, which are stored Fortran ordered). Its cells become hexahedra (three
directions with more than one node), quads (two) or lines (one), oriented so that
their volume (or area, in 2D) is positive.

BCs given as a PointRange on a face of the block become a point set and a group
of boundary faces (quads, or lines in 2D), which are generated since structured
zones have no face elements. Other ranges (edges, corners) only give a point set.
"""

import numpy as np

from .._common import warn
from .._exceptions import ReadError
from ._groups import family_names, location
from ._tree import child, children, data, name, resolve, text

# outward faces of a positive hexahedron, by (direction, side): its nodes are
# p, p+i, p+i+j, p+j, then the same plus k
_HEX_FACES = {
    (0, 0): [3, 0, 4, 7],
    (0, 1): [1, 2, 6, 5],
    (1, 0): [0, 1, 5, 4],
    (1, 1): [2, 3, 7, 6],
    (2, 0): [0, 3, 2, 1],
    (2, 1): [4, 5, 6, 7],
}
# outward edges of a counterclockwise quad p, p+a, p+a+b, p+b
_QUAD_EDGES = {(0, 0): [3, 0], (0, 1): [1, 2], (1, 0): [0, 1], (1, 1): [2, 3]}


def _vertex_size(zone):
    size = np.asarray(data(zone)).reshape(3, -1)
    return [int(n) for n in size[0]]


def coordinates(zone, phys_dim):
    """Points of a structured zone (i fastest), rind planes removed."""
    grids = children(zone, "GridCoordinates_t")
    if not grids:
        raise ReadError(f"Zone {name(zone)} has no GridCoordinates.")
    grid = next((g for g in grids if name(g) == "GridCoordinates"), grids[0])
    arrays = {
        name(a): np.asarray(data(a), dtype=float) for a in children(grid, "DataArray_t")
    }
    if "CoordinateX" not in arrays:
        found = ", ".join(sorted(arrays))
        raise ReadError(
            f"Zone {name(zone)}: only Cartesian coordinates are supported ({found})."
        )
    rind = grid.get("Rind")
    cols = []
    for axis in "XYZ"[:phys_dim]:
        values = arrays.get(f"Coordinate{axis}")
        if values is None:
            values = np.zeros_like(arrays["CoordinateX"])
        if rind is not None:
            values = _strip_rind(values, np.ravel(data(resolve(rind))))
        cols.append(values.ravel())
    return np.column_stack(cols)


def _strip_rind(values, rind):
    """Remove the rind planes of a structured array (HDF5 axes are k, j, i)."""
    index = []
    for axis in range(values.ndim):
        d = values.ndim - 1 - axis  # SIDS direction of this HDF5 axis
        lo, hi = int(rind[2 * d]), int(rind[2 * d + 1])
        index.append(slice(lo, values.shape[axis] - hi))
    return values[tuple(index)]


class Block:
    """Unstructured view of a structured zone."""

    def __init__(self, zone, points):
        self.zone = zone
        self.size = _vertex_size(zone)
        if int(np.prod(self.size)) != len(points):
            raise ReadError(f"Zone {name(zone)}: coordinates do not match its size.")
        self.points = points
        self.stride = [int(np.prod(self.size[:d])) for d in range(len(self.size))]
        self.active = [d for d, n in enumerate(self.size) if n > 1]
        self.flipped = False
        self.cell_type, self.cells = self._cells()

    def _base_nodes(self, ranges):
        """Index of the first node of every cell over the given per-direction
        index ranges (i fastest)."""
        grids = np.meshgrid(*[np.asarray(r) for r in ranges[::-1]], indexing="ij")
        p = np.zeros(grids[0].shape, dtype=np.int64)
        for d, g in zip(range(len(ranges) - 1, -1, -1), grids):
            p += g * self.stride[d]
        return p.ravel()

    def _cells(self):
        active, size, stride = self.active, self.size, self.stride
        ranges = [
            range(size[d] - 1) if d in active else range(1) for d in range(len(size))
        ]
        p = self._base_nodes(ranges)
        offsets = {0: [0]}
        if len(active) >= 1:
            a = stride[active[0]]
            offsets[1] = [0, a]
        if len(active) >= 2:
            b = stride[active[1]]
            offsets[2] = [0, a, a + b, b]
        if len(active) == 3:
            c = stride[active[2]]
            offsets[3] = offsets[2] + [o + c for o in offsets[2]]
        kind = {0: "vertex", 1: "line", 2: "quad", 3: "hexahedron"}[len(active)]
        cells = p[:, None] + np.array(offsets[len(active)], dtype=np.int64)
        cells = self._orient(kind, cells)
        return kind, cells

    def _orient(self, kind, cells):
        """Flip all cells when the block is left-handed (most cells negative)."""
        x = self.points
        if kind == "hexahedron":
            p = x[cells[:, [0, 1, 3, 4]]]
            v = np.einsum(
                "ij,ij->i",
                np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]),
                p[:, 3] - p[:, 0],
            )
            perm = [0, 3, 2, 1, 4, 7, 6, 5]
        elif kind == "quad" and x.shape[1] == 2:
            p = x[cells]
            v = np.sum(
                p[..., 0] * np.roll(p[..., 1], -1, axis=1)
                - np.roll(p[..., 0], -1, axis=1) * p[..., 1],
                axis=1,
            )
            perm = [0, 3, 2, 1]
        else:
            return cells
        if np.sum(np.sign(v)) < 0:
            self.flipped = True
            cells = cells[:, perm]
        return cells

    def cell_index(self, lo, hi):
        """Cells with lower corner in [lo, hi] (0-based, per direction)."""
        ranges = []
        for d, n in enumerate(self.size):
            if d in self.active:
                ranges.append(range(max(lo[d], 0), min(hi[d], n - 2) + 1))
            else:
                ranges.append(range(1))
        cell_size = [n - 1 if d in self.active else 1 for d, n in enumerate(self.size)]
        grids = np.meshgrid(*[np.asarray(r) for r in ranges[::-1]], indexing="ij")
        index = np.zeros(grids[0].shape, dtype=np.int64)
        stride = 1
        for d in range(len(cell_size)):
            index += grids[len(cell_size) - 1 - d] * stride
            stride *= cell_size[d]
        return index.ravel()

    def nodes(self, lo, hi):
        """Nodes with indices in [lo, hi] (0-based, per direction)."""
        return self._base_nodes([range(lo[d], hi[d] + 1) for d in range(len(lo))])

    def faces(self, lo, hi):
        """Boundary faces (outward) over a node range that is flat in exactly one
        active direction, or None."""
        flat = [d for d in self.active if lo[d] == hi[d]]
        if len(flat) != 1 or len(self.active) < 2:
            return None
        d = flat[0]
        n = self.size[d]
        side = 1 if lo[d] == n - 1 else 0
        cell_lo = list(lo)
        cell_hi = [h - 1 for h in hi]
        cell_lo[d] = cell_hi[d] = n - 2 if side else lo[d]
        if lo[d] not in (0, n - 1):
            side = 0  # interior range: faces of the cells on its upper side
        cells = self.cells[self.cell_index(cell_lo, cell_hi)]
        k = self.active.index(d)
        table = _HEX_FACES if len(self.active) == 3 else _QUAD_EDGES
        faces = cells[:, table[(k, side)]]
        return faces[:, ::-1] if self.flipped else faces


def _range(node, size, loc):
    """0-based (lo, hi) node range of a PointRange, per direction."""
    r = np.asarray(data(resolve(node)), dtype=np.int64).reshape(2, -1)
    lo, hi = r.min(axis=0) - 1, r.max(axis=0) - 1
    if loc not in ("Vertex", ""):
        # face or cell centered ranges: cell indices, one node more on the side
        normal = {"IFaceCenter": 0, "JFaceCenter": 1, "KFaceCenter": 2}.get(loc)
        for d in range(len(size)):
            if d != normal and size[d] > 1:
                hi[d] += 1
    return [int(x) for x in lo], [int(x) for x in hi]


def zone_groups(block, version):
    """Faces generated for the BCs, and ``[(name, kind, values)]`` groups: kind
    "faces" (indices in the face block), "region points" (1-based nodes) or
    "cells" (indices of cells)."""
    zone = block.zone
    faces, groups = [], []
    count = 0
    zone_bc = child(zone, "ZoneBC_t")
    for bc in children(zone_bc, "BC_t") if zone_bc is not None else []:
        group = name(bc) or next(iter(family_names(bc, version)), "")
        loc = location(bc, version)
        if "PointRange" in bc:
            lo, hi = _range(bc["PointRange"], block.size, loc)
        elif "PointList" in bc:
            points = (
                np.asarray(data(resolve(bc["PointList"])), dtype=np.int64).reshape(
                    -1, len(block.size)
                )
                - 1
            )
            nodes = points @ np.array(block.stride)
            groups.append((group, "region points", nodes + 1))
            continue
        else:
            continue
        if loc == "CellCenter":
            groups.append((group, "cells", block.cell_index(lo, [h - 1 for h in hi])))
            continue
        groups.append((group, "region points", block.nodes(lo, hi) + 1))
        f = block.faces(lo, hi)
        if f is not None and len(f):
            faces.append(f)
            groups.append((group, "faces", count + np.arange(len(f))))
            count += len(f)
    return faces, groups


def interface_nodes(block):
    """Nodes on the 1-to-1 interfaces of a zone (1-based), to be merged with the
    matching nodes of the neighbouring zones."""
    zone = block.zone
    out = []
    for zgc in children(zone, "ZoneGridConnectivity_t"):
        for conn in children(zgc, "GridConnectivity1to1_t"):
            if "PointRange" in conn:
                lo, hi = _range(conn["PointRange"], block.size, "Vertex")
                out.append(block.nodes(lo, hi) + 1)
        for conn in children(zgc, "GridConnectivity_t"):
            kind = child(conn, "GridConnectivityType_t")
            kind = text(kind).strip("\x00 ") if kind is not None else "Overset"
            if kind == "Overset":
                warn(f"Zone {name(zone)}: overset connectivity is not used.")
            elif "PointRange" in conn:
                lo, hi = _range(conn["PointRange"], block.size, "Vertex")
                out.append(block.nodes(lo, hi) + 1)
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)

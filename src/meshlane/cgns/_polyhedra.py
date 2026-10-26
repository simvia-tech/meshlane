"""Cells of NFACE_n sections: each cell is given by its faces (NGON_n elements).

The faces of each cell are oriented outward, then cells with the face structure
of a tetrahedron, pyramid, wedge or hexahedron become that cell type (meshlane
node ordering, positive volume); the others become ``polyhedron<N>`` cells (list
of outward faces). The rebuilding of regular cells follows the Fluent reader
(``ansys/_cells.py``).

NFACE_n signs each face of a cell to tell whether its normal points out of the
cell, but writers disagree on the meaning of the sign, and some files are not
consistent. Each cell is therefore oriented with the first of these that is
consistent (every edge of the cell walked once each way):

1. the signs, read with the file's convention (measured: faces whose normal
   points away from the cell center), the whole cell being flipped if its volume
   comes out negative; the signs stay right for non-convex cells;
2. the geometric orientation of each face (normal away from the cell center);
3. otherwise (open or malformed cells), the geometric orientation anyway.
"""

import numpy as np

_CHUNK = 1 << 20


def _edge_hashes(ptr, nodes):
    """Per face, the sum over its edges u -> v of H[u] G[v] - H[v] G[u] (uint64,
    wrapping). The term is antisymmetric, so the sum over the faces of a cell is
    exactly zero when each edge is walked once in each direction, and almost
    surely not otherwise."""
    rng = np.random.default_rng(0)
    high = int(nodes.max(initial=0)) + 1
    h = rng.integers(0, 2**63, size=high, dtype=np.uint64)
    g = rng.integers(0, 2**63, size=high, dtype=np.uint64)
    num_faces = len(ptr) - 1
    out = np.empty(num_faces, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for a in range(0, num_faces, _CHUNK):
            b = min(a + _CHUNK, num_faces)
            p0, p1 = ptr[a], ptr[b]
            local = ptr[a : b + 1] - p0
            u = nodes[p0:p1]
            nxt = np.arange(1, p1 - p0 + 1)
            nxt[local[1:] - 1] = local[:-1]
            v = u[nxt]
            out[a:b] = np.add.reduceat(h[u] * g[v] - h[v] * g[u], local[:-1])
    return out


def _consistent(cell_ptr, hashes, reverse):
    """Whether each cell is consistently oriented, faces flipped where reverse."""
    signed = np.where(reverse, np.uint64(0) - hashes, hashes)
    sums = np.add.reduceat(signed, cell_ptr[:-1]) if len(signed) else signed
    return sums == 0


def _face_geometry(points, ptr, nodes):
    """Center and (unnormalized, Newell) normal of every face, by chunks of faces
    to bound the memory of large meshes."""
    num_faces = len(ptr) - 1
    centers = np.empty((num_faces, 3))
    normals = np.empty((num_faces, 3))
    for a in range(0, num_faces, _CHUNK):
        b = min(a + _CHUNK, num_faces)
        p0, p1 = ptr[a], ptr[b]
        local = ptr[a : b + 1] - p0
        size = np.diff(local)
        p = points[nodes[p0:p1]]
        nxt = np.arange(1, p1 - p0 + 1)
        nxt[local[1:] - 1] = local[:-1]
        centers[a:b] = np.add.reduceat(p, local[:-1], axis=0) / size[:, None]
        normals[a:b] = np.add.reduceat(np.cross(p, p[nxt]), local[:-1], axis=0)
    return centers, normals


class _CellFaces:
    """The faces of every cell, sorted by cell then by face size, each with a flag
    telling whether its stored node order must be reversed to point outward."""

    def __init__(self, points, face_ptr, face_nodes, cell_ptr, cell_faces, positive):
        num_cells = len(cell_ptr) - 1
        self.face_ptr = face_ptr
        self.face_nodes = face_nodes
        self.face_size = np.diff(face_ptr)
        cell = np.repeat(np.arange(num_cells), np.diff(cell_ptr))
        face = np.asarray(cell_faces, dtype=np.int64)

        centers, normals = _face_geometry(points, face_ptr, face_nodes)
        count = np.maximum(np.bincount(cell, minlength=num_cells), 1)
        cell_centers = (
            np.column_stack(
                [
                    np.bincount(cell, weights=centers[face, d], minlength=num_cells)
                    for d in range(3)
                ]
            )
            / count[:, None]
        )
        # dot of each face normal with (face center - cell center): positive
        # when the normal points out of the cell; dot / 6 is the volume of the
        # cone from the cell center to the face
        dots = np.empty(len(face))
        for a in range(0, len(face), _CHUNK):
            f, c = face[a : a + _CHUNK], cell[a : a + _CHUNK]
            dots[a : a + _CHUNK] = np.einsum(
                "ij,ij->i", normals[f], centers[f] - cell_centers[c]
            )
        del centers, normals, cell_centers
        geometric = dots > 0
        out_if_positive = np.mean(geometric == positive) >= 0.5 if len(face) else True
        by_sign = positive if out_if_positive else ~positive
        volume = np.bincount(
            cell, weights=np.where(by_sign, dots, -dots), minlength=num_cells
        )
        del dots
        hashes = _edge_hashes(face_ptr, face_nodes)[face]
        sign_ok = _consistent(cell_ptr, hashes, ~by_sign)
        # consistent signs: flip the cell if they make it inside out
        by_sign ^= np.repeat(sign_ok & (volume < 0), np.diff(cell_ptr))
        use_sign = np.repeat(sign_ok, np.diff(cell_ptr))
        rev = ~np.where(use_sign, by_sign, geometric)
        del hashes, geometric, by_sign, volume, use_sign

        order = np.lexsort((self.face_size[face], cell))
        self.face = face[order]
        self.rev = rev[order]
        cell = cell[order]
        self.num = np.bincount(cell, minlength=num_cells)
        self.offset = np.concatenate([[0], np.cumsum(self.num)])
        size = self.face_size[self.face]
        self.num_tri = np.bincount(cell, weights=size == 3, minlength=num_cells)
        self.num_quad = np.bincount(cell, weights=size == 4, minlength=num_cells)

    def slot(self, cells, j, k):
        """Outward nodes (len(cells), k) of the j-th face of each cell, which must
        have k nodes."""
        h = self.offset[cells] + j
        f = self.face[h]
        nodes = self.face_nodes[self.face_ptr[f][:, None] + np.arange(k)]
        return np.where(self.rev[h][:, None], nodes[:, ::-1], nodes)

    def outward_faces(self):
        """For every face used by a single cell: whether its node order must be
        reversed to point out of the domain."""
        used = np.bincount(self.face, minlength=len(self.face_size))
        single = used[self.face] == 1
        rev = np.zeros(len(self.face_size), dtype=bool)
        rev[self.face[single]] = self.rev[single]
        return used == 1, rev


def _isin_rows(a, b):
    """a[i, j] in b[i, :] for every row i."""
    return (a[:, :, None] == b[:, None, :]).any(axis=2)


def _unique_rows(conn):
    """True for the rows of conn whose entries are pairwise distinct."""
    s = np.sort(conn, axis=1)
    return (np.diff(s, axis=1) != 0).all(axis=1)


def _apex(base, face):
    """The single node of face not in base, and whether there is exactly one."""
    mask = ~_isin_rows(face, base)
    ok = mask.sum(axis=1) == 1
    return face[np.arange(len(face)), mask.argmax(axis=1)], ok


def _top(base, sides):
    """For each node of the base face, its neighbour off the base (along the edges
    of the side faces), and whether it is unique for every node."""
    e0 = np.concatenate(sides, axis=1)
    e1 = np.concatenate([np.roll(s, -1, axis=1) for s in sides], axis=1)
    in0 = _isin_rows(e0, base)
    in1 = _isin_rows(e1, base)
    top = np.empty_like(base)
    ok = np.ones(len(base), dtype=bool)
    for k in range(base.shape[1]):
        b = base[:, k : k + 1]
        cand = np.where((e0 == b) & ~in1, e1, np.where((e1 == b) & ~in0, e0, -1))
        hi = cand.max(axis=1)
        lo = np.where(cand < 0, np.iinfo(cand.dtype).max, cand).min(axis=1)
        ok &= (hi >= 0) & (hi == lo)
        top[:, k] = hi
    return top, ok


def _build_regular(cf, cells, kind):
    """Connectivity of the cells of one kind, and which ones are valid."""
    if kind == "tetra":
        base = cf.slot(cells, 0, 3)[:, ::-1]
        apex, ok = _apex(base, cf.slot(cells, 1, 3))
        conn = np.column_stack([base, apex])
    elif kind == "pyramid":
        # faces sorted by size: four triangles then the quad base
        base = cf.slot(cells, 4, 4)[:, ::-1]
        apex, ok = _apex(base, cf.slot(cells, 0, 3))
        conn = np.column_stack([base, apex])
    else:
        sizes = [3, 3, 4, 4, 4] if kind == "wedge" else [4] * 6
        slots = [cf.slot(cells, j, k) for j, k in enumerate(sizes)]
        base = slots[0][:, ::-1]
        top, ok = _top(base, slots[1:])
        conn = np.column_stack([base, top])
    return conn, ok & _unique_rows(conn)


def _polyhedra(cf, cells):
    """``polyhedron<N>`` blocks (each cell: list of outward faces, N: its number of
    nodes). The outward node lists of all faces are gathered in one array and each
    face is a view of it, to keep millions of cells affordable."""
    if len(cells) == 0:
        return []
    num = cf.num[cells]
    cell_ptr = np.concatenate([[0], np.cumsum(num)])
    h = np.repeat(cf.offset[cells] - cell_ptr[:-1], num) + np.arange(cell_ptr[-1])
    f = cf.face[h]
    size = cf.face_size[f]
    face_ptr = np.concatenate([[0], np.cumsum(size)])
    total = int(face_ptr[-1])
    index = np.int32 if max(total, len(cf.face_nodes)) < 2**31 else np.int64

    within = np.arange(total, dtype=index)
    within -= np.repeat(face_ptr[:-1].astype(index), size)
    reverse = np.repeat(cf.rev[h], size)
    flipped = np.repeat((size - 1).astype(index), size)
    np.subtract(flipped, within, out=flipped)
    np.copyto(within, flipped, where=reverse)
    del flipped, reverse
    within += np.repeat(cf.face_ptr[f].astype(index), size)
    flat = cf.face_nodes[within]
    del within
    if flat.max(initial=0) < 2**31:
        flat = flat.astype(np.int32)

    # number of distinct nodes of each cell
    base = int(flat.max(initial=0)) + 1
    key = np.repeat(np.repeat(np.arange(len(cells), dtype=np.int64), num), size)
    key *= base
    key += flat
    key.sort()
    first = np.empty(len(key), dtype=bool)
    if len(key):
        first[0] = True
        np.not_equal(key[1:], key[:-1], out=first[1:])
    num_nodes = np.bincount(key[first] // base, minlength=len(cells))
    del key, first

    views = []
    for a in range(0, len(f), _CHUNK):
        p = face_ptr[a : a + _CHUNK + 1].tolist()
        views.extend(flat[p[k] : p[k + 1]] for k in range(len(p) - 1))
    cp = cell_ptr.tolist()
    blocks = []
    for n in np.unique(num_nodes):
        sel = np.flatnonzero(num_nodes == n)
        data = np.empty(len(sel), dtype=object)
        for i, c in enumerate(sel.tolist()):
            data[i] = views[cp[c] : cp[c + 1]]
        blocks.append((f"polyhedron{n}", data, cells[sel]))
    return blocks


def build_cells(points, face_ptr, face_nodes, cell_ptr, cell_faces, positive):
    """Cells of an NFACE_n section.

    face_ptr/face_nodes: CSR of all NGON_n faces (0-based nodes); cell_ptr/
    cell_faces: CSR of the faces (0-based face indices) of each cell, positive:
    whether NFACE_n gives each of them a positive sign.

    Returns ``(blocks, boundary, reverse)``: blocks is a list of
    ``(cell_type, data, cell indices)``; boundary marks the faces used by one
    cell, and reverse whether their node order must be flipped to point out of
    the domain.
    """
    cf = _CellFaces(points, face_ptr, face_nodes, cell_ptr, cell_faces, positive)
    num, tri, quad = cf.num, cf.num_tri, cf.num_quad
    kinds = {
        "tetra": (num == 4) & (tri == 4),
        "pyramid": (num == 5) & (tri == 4) & (quad == 1),
        "wedge": (num == 5) & (tri == 2) & (quad == 3),
        "hexahedron": (num == 6) & (quad == 6),
    }
    blocks = []
    leftover = [np.flatnonzero(~np.logical_or.reduce(list(kinds.values())))]
    for kind, mask in kinds.items():
        cells = np.flatnonzero(mask)
        if len(cells) == 0:
            continue
        conn, ok = _build_regular(cf, cells, kind)
        if ok.any():
            blocks.append((kind, conn[ok], cells[ok]))
        leftover.append(cells[~ok])
    blocks.extend(_polyhedra(cf, np.sort(np.concatenate(leftover))))
    boundary, reverse = cf.outward_faces()
    return blocks, boundary, reverse

"""Rebuild cell connectivity from the faces of a Fluent mesh.

Fluent mesh files do not store the nodes of the volume cells: every face lists its
nodes and the cells on either side (c0, c1), and the cells are only known through
their faces. This module groups the faces by cell, orients them outward and turns
each cell into a meshlane cell (meshlane node ordering, positive volume):

- 3D: tetra, pyramid, wedge and hexahedron when the faces allow it, otherwise a
  general ``polyhedron<N>`` (list of outward faces), e.g. for polyhedral meshes or
  hexahedra with hanging nodes;
- 2D: triangle, quad, or ``polygon<N>`` (counterclockwise nodes).

The work is vectorized per cell type, so meshes with millions of cells stay fast.
"""

from collections import defaultdict

import numpy as np

from .._exceptions import ReadError


class Faces:
    """Faces stored as CSR (``ptr``, ``nodes``), with the 0-based cells on either
    side (``c0``, ``c1``, -1 when there is none)."""

    def __init__(self, ptr, nodes, c0, c1):
        self.ptr = ptr
        self.nodes = nodes
        self.c0 = c0
        self.c1 = c1
        self.size = np.diff(ptr)

    def __len__(self):
        return len(self.c0)

    def subset(self, idx):
        """The faces idx, as a new Faces."""
        size = self.size[idx]
        ptr = np.concatenate([[0], np.cumsum(size)])
        nodes = self.nodes[
            np.repeat(self.ptr[idx] - ptr[:-1], size) + np.arange(ptr[-1])
        ]
        return Faces(ptr, nodes, self.c0[idx], self.c1[idx])


def _face_centers_normals(points, faces):
    """Center and (unnormalized) normal of each face. In 2D a face is a segment
    a -> b and its normal is the right-hand normal (dy, -dx)."""
    start = faces.ptr[:-1]
    p = points[faces.nodes]
    centers = np.add.reduceat(p, start, axis=0) / faces.size[:, None]
    if points.shape[1] == 2:
        a = points[faces.nodes[start]]
        b = points[faces.nodes[start + 1]]
        d = b - a
        return centers, np.column_stack([d[:, 1], -d[:, 0]])
    # Newell's method: sum of p_i x p_{i+1} over the face edges
    nxt = np.arange(1, len(faces.nodes) + 1)
    nxt[faces.ptr[1:] - 1] = start
    normals = np.add.reduceat(np.cross(p, points[faces.nodes[nxt]]), start, axis=0)
    return centers, normals


def stored_normal_points_out_of_c0(points, faces, n_cells, sample=50_000):
    """Whether the right-hand normal of the stored node order points away from c0.

    Fluent writers agree on one convention per file, but it is easy to get wrong,
    so it is measured: each face votes by comparing its normal with the direction
    from the center of c0 to the face center. Only the faces of the first
    ``sample`` cells vote, which is plenty and keeps large meshes cheap."""
    if n_cells > sample:
        c0, c1 = faces.c0, faces.c1
        near = ((c0 >= 0) & (c0 < sample)) | ((c1 >= 0) & (c1 < sample))
        faces = faces.subset(np.flatnonzero(near))
        faces.c0 = np.where(faces.c0 < sample, faces.c0, -1)
        faces.c1 = np.where(faces.c1 < sample, faces.c1, -1)
        n_cells = sample
    centers, normals = _face_centers_normals(points, faces)
    cells = np.concatenate([faces.c0, faces.c1])
    fc = np.concatenate([centers, centers])
    keep = cells >= 0
    cells, fc = cells[keep], fc[keep]
    count = np.bincount(cells, minlength=n_cells)
    count[count == 0] = 1
    cell_centers = (
        np.column_stack(
            [
                np.bincount(cells, weights=fc[:, d], minlength=n_cells)
                for d in range(fc.shape[1])
            ]
        )
        / count[:, None]
    )
    has_c0 = faces.c0 >= 0
    s = np.einsum(
        "ij,ij->i", normals[has_c0], centers[has_c0] - cell_centers[faces.c0[has_c0]]
    )
    return np.sum(np.sign(s)) >= 0


class _CellFaces:
    """The faces of every cell, sorted by cell then by face size, each with a flag
    telling whether its stored node order must be reversed to point outward."""

    def __init__(self, faces, n_cells, out_of_c0):
        has0 = faces.c0 >= 0
        has1 = faces.c1 >= 0
        f0 = np.flatnonzero(has0)
        f1 = np.flatnonzero(has1)
        cell = np.concatenate([faces.c0[f0], faces.c1[f1]])
        face = np.concatenate([f0, f1])
        rev = np.concatenate(
            [np.full(len(f0), not out_of_c0), np.full(len(f1), out_of_c0)]
        )
        order = np.lexsort((faces.size[face], cell))
        self.face = face[order]
        self.rev = rev[order]
        cell = cell[order]
        self.faces = faces
        self.num = np.bincount(cell, minlength=n_cells)
        self.offset = np.concatenate([[0], np.cumsum(self.num)])
        size = faces.size[self.face]
        self.num_tri = np.bincount(cell, weights=size == 3, minlength=n_cells)
        self.num_quad = np.bincount(cell, weights=size == 4, minlength=n_cells)

    def slot(self, cells, j, k):
        """Outward nodes (len(cells), k) of the j-th face of each cell, which must
        have k nodes."""
        h = self.offset[cells] + j
        f = self.face[h]
        nodes = self.faces.nodes[self.faces.ptr[f][:, None] + np.arange(k)]
        return np.where(self.rev[h][:, None], nodes[:, ::-1], nodes)

    def outward(self, cell):
        """Outward faces of one cell, as a list of node arrays."""
        out = []
        for h in range(self.offset[cell], self.offset[cell + 1]):
            f = self.face[h]
            nodes = self.faces.nodes[self.faces.ptr[f] : self.faces.ptr[f + 1]]
            out.append(nodes[::-1].copy() if self.rev[h] else nodes.copy())
        return out


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


def _build_3d(cf, cells, kind):
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


def _chain(edges):
    """Walk the outward (counterclockwise) edges (n, m, 2) of 2D cells into node
    loops (n, m), and tell which cells form a single closed loop."""
    n, m, _ = edges.shape
    rows = np.arange(n)
    loop = np.empty((n, m), dtype=edges.dtype)
    loop[:, 0] = edges[:, 0, 0]
    loop[:, 1] = edges[:, 0, 1]
    ok = np.ones(n, dtype=bool)
    for k in range(2, m + 1):
        match = edges[:, :, 0] == loop[:, k - 1 : k]
        ok &= match.sum(axis=1) == 1
        nxt = edges[rows, match.argmax(axis=1), 1]
        if k < m:
            loop[:, k] = nxt
        else:
            ok &= nxt == loop[:, 0]
    return loop, ok & _unique_rows(loop)


def _chain_one(faces):
    """Node loop of one 2D cell from its outward edges, or None."""
    nxt = {int(f[0]): int(f[1]) for f in faces}
    if len(nxt) != len(faces):
        return None
    start = int(faces[0][0])
    loop = [start]
    while len(loop) < len(faces):
        loop.append(nxt.get(loop[-1], -1))
    if nxt.get(loop[-1]) != start or len(set(loop)) != len(loop):
        return None
    return loop


def _polyhedra(cf, cells):
    """``polyhedron<N>`` blocks (each cell: list of outward faces, N: its number of
    nodes) for the given cells.

    Polyhedral meshes can have millions of cells, so the outward node lists of all
    their faces are gathered at once in one array, each face being a view of it,
    and the temporary index arrays are kept small (32-bit when possible, updated
    in place).
    """
    if len(cells) == 0:
        return []
    faces = cf.faces
    num = cf.num[cells]
    cell_ptr = np.concatenate([[0], np.cumsum(num)])
    h = np.repeat(cf.offset[cells] - cell_ptr[:-1], num) + np.arange(cell_ptr[-1])
    f = cf.face[h]
    size = faces.size[f]
    face_ptr = np.concatenate([[0], np.cumsum(size)])
    total = int(face_ptr[-1])
    index = np.int32 if max(total, len(faces.nodes)) < 2**31 else np.int64

    # position in faces.nodes of each node of each outward face
    within = np.arange(total, dtype=index)
    within -= np.repeat(face_ptr[:-1].astype(index), size)
    reverse = np.repeat(cf.rev[h], size)
    flipped = np.repeat((size - 1).astype(index), size)
    np.subtract(flipped, within, out=flipped)
    np.copyto(within, flipped, where=reverse)
    del flipped, reverse
    within += np.repeat(faces.ptr[f].astype(index), size)
    nodes = faces.nodes[within]
    del within
    flat = nodes.astype(index) if nodes.max(initial=0) < 2**31 else nodes
    del nodes

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

    # one view per face, made by chunks to avoid a huge list of Python ints
    views = []
    chunk = 1 << 20
    for a in range(0, len(f), chunk):
        p = face_ptr[a : a + chunk + 1].tolist()
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


def build_cells(points, faces, n_cells):
    """Rebuild the cells from their faces.

    Returns ``(blocks, out_of_c0)`` where blocks is a list of
    ``(cell_type, data, cell_ids)`` (cell_ids: the 0-based Fluent cell of each row)
    and out_of_c0 is the face convention found by
    :func:`stored_normal_points_out_of_c0`.
    """
    out_of_c0 = stored_normal_points_out_of_c0(points, faces, n_cells)
    cf = _CellFaces(faces, n_cells, out_of_c0)
    num, tri, quad = cf.num, cf.num_tri, cf.num_quad

    blocks = []
    leftover = []
    if points.shape[1] == 3:
        kinds = {
            "tetra": (num == 4) & (tri == 4),
            "pyramid": (num == 5) & (tri == 4) & (quad == 1),
            "wedge": (num == 5) & (tri == 2) & (quad == 3),
            "hexahedron": (num == 6) & (quad == 6),
        }
        for kind, mask in kinds.items():
            cells = np.flatnonzero(mask)
            if len(cells) == 0:
                continue
            conn, ok = _build_3d(cf, cells, kind)
            if ok.any():
                blocks.append((kind, conn[ok], cells[ok]))
            leftover.append(cells[~ok])
        regular = np.zeros(n_cells, dtype=bool)
        for mask in kinds.values():
            regular |= mask
        leftover.append(np.flatnonzero(~regular & (num >= 4)))
        blocks.extend(_polyhedra(cf, np.sort(np.concatenate(leftover))))
        skipped = int(np.count_nonzero((num > 0) & (num < 4)))
    else:
        for kind, m in (("triangle", 3), ("quad", 4)):
            cells = np.flatnonzero(num == m)
            if len(cells) == 0:
                continue
            edges = np.stack([cf.slot(cells, j, 2) for j in range(m)], axis=1)
            conn, ok = _chain(edges)
            if ok.any():
                blocks.append((kind, conn[ok], cells[ok]))
            leftover.append(cells[~ok])
        leftover.append(np.flatnonzero(num > 4))
        by_n = defaultdict(list)
        skipped = int(np.count_nonzero((num > 0) & (num < 3)))
        for c in np.sort(np.concatenate(leftover)):
            loop = _chain_one(cf.outward(c))
            if loop is None:
                skipped += 1
            else:
                by_n[len(loop)].append((c, loop))
        for n_nodes in sorted(by_n):
            items = by_n[n_nodes]
            kind = {3: "triangle", 4: "quad"}.get(n_nodes, f"polygon{n_nodes}")
            conn = np.array([loop for _, loop in items], dtype=int)
            ids = np.array([c for c, _ in items], dtype=int)
            blocks.append((kind, conn, ids))

    if skipped:
        raise ReadError(f"{skipped} cell(s) could not be rebuilt from their faces.")
    return _merge(blocks), out_of_c0


def _merge(blocks):
    """Merge the blocks of the same regular type (2D cells rebuilt one by one end
    up in a second block), keeping the rows sorted by cell id."""
    merged = {}
    for kind, data, ids in blocks:
        if kind in merged and not kind.startswith("polyhedron"):
            d0, i0 = merged[kind]
            data, ids = np.concatenate([d0, data]), np.concatenate([i0, ids])
            order = np.argsort(ids, kind="stable")
            data, ids = data[order], ids[order]
        merged[kind] = (data, ids)
    return [(kind, data, ids) for kind, (data, ids) in merged.items()]

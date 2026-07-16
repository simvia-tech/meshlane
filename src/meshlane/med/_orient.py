"""Make 3D cells consistently oriented before writing them to MED.

Many MED tools need two neighbouring cells to list their shared face in
opposite directions. meshlane's fixed per-type node permutation already
gets this right for normal cells, but not for a few badly warped cells
(common in cfMesh / snappyHexMesh meshes). Those few break the mesh
connectivity and MED tools may reject the mesh with a fatal
face to cell connectivity error.

This module repairs them: it finds which cells share each face, then flips the
offending cells so every shared face is listed oppositely by its two cells. It
works purely on the mesh topology and not on cell volumes, to stay robust to any
warping. A final per-region volume check fixes the overall sign so no cell ends
up inside-out.

It handles two kinds of 3D cell: regular cells, with a fixed node count per
type (tetra 4, pyramid 5,...) stored as a rectangular array; and polyhedra,
with a variable number of faces and nodes per cell.
"""
import numpy as np

# The faces of each cell type, in MED's node order (from MEDCoupling's
# CellModel). Reversed so every cell type and the polyhedra share one
# outward-facing convention.
_RAW_FACES = {
    "tetra": [[0, 2, 1], [0, 3, 2], [2, 3, 1], [1, 3, 0]],
    "pyramid": [[0, 3, 2, 1], [0, 4, 3], [3, 4, 2], [2, 4, 1], [1, 4, 0]],
    "wedge": [[0, 2, 1], [3, 4, 5], [0, 3, 5, 2], [2, 5, 4, 1], [1, 4, 3, 0]],
    "hexahedron": [[0, 3, 2, 1], [4, 5, 6, 7], [0, 4, 7, 3],
                   [3, 7, 6, 2], [2, 6, 5, 1], [1, 5, 4, 0]],
}
_FACES = {t: [f[::-1] for f in fs] for t, fs in _RAW_FACES.items()}

# orientation-reversing node permutations (self-inverse) per MED cell type
_FLIP_PERM = {
    "tetra": [0, 2, 1, 3],
    "pyramid": [0, 3, 2, 1, 4],
    "wedge": [0, 2, 1, 3, 5, 4],
    "hexahedron": [0, 3, 2, 1, 4, 7, 6, 5],
}

ORIENTABLE_TYPES = set(_FACES)


def _chirality(ordered):
    """Return a 0/1 'winding tag' for each face, computed from its node order
    alone. Two cells share a face using the same node ids: if they list them the
    same way round the tags match (windings agree, inconsistent for MED); if
    opposite ways the tags differ (consistent)."""
    nf, k = ordered.shape
    idx = np.arange(nf)
    mn = np.argmin(ordered, axis=1)
    nxt = ordered[idx, (mn + 1) % k]
    prv = ordered[idx, (mn - 1) % k]
    return (nxt < prv).astype(np.int8)


def _cell_faces(cell_type, conn):
    """Yield (size, ordered_faces[n,size]) for a regular cell block."""
    for face in _FACES[cell_type]:
        yield len(face), conn[:, face]


def _signed_volume(cell_type, conn, points):
    """Signed volume of each cell, computed from its faces. Positive when the
    cell is correctly oriented, negative when inside-out (only the sign is
    used)."""
    V = np.zeros(len(conn))
    for face in _FACES[cell_type]:
        fp = points[conn[:, face]]
        c = fp.mean(axis=1)
        k = len(face)
        for i in range(k):
            V += np.einsum("ij,ij->i", c,
                           np.cross(fp[:, i], fp[:, (i + 1) % k])) / 6.0
    return V


def _uf_find(parent, rel, x):
    """Union-find 'find' with path compression and parity accumulation.
    Returns (root, parity of x relative to root). Mutates parent/rel in place."""
    root = x
    p = 0
    while parent[root] != root:
        p ^= rel[root]
        root = parent[root]
    cur = x
    pc = 0
    while parent[cur] != root:
        nxt = parent[cur]
        nb = rel[cur]
        parent[cur] = root
        rel[cur] = pc ^ p
        pc ^= nb
        cur = nxt
    return root, p


def _solve_parity(n, A, B, par):
    """Union-find with parity. Returns (flip_bit[n], roots[n], n_nonorientable).
    flip_bit is each cell's orientation relative to its component's root."""
    parent = list(range(n))
    rank = [0] * n
    rel = [0] * n  # parity relative to parent

    violations = 0
    for a, b, pab in zip(A.tolist(), B.tolist(), par.tolist()):
        ra, pa = _uf_find(parent, rel, a)
        rb, pb = _uf_find(parent, rel, b)
        want = pab ^ pa ^ pb
        if ra == rb:
            if (pa ^ pb) != pab:
                violations += 1
            continue
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        rel[rb] = want
        if rank[ra] == rank[rb]:
            rank[ra] += 1

    flip = np.empty(n, dtype=np.int8)
    roots = np.empty(n, dtype=np.int64)
    for x in range(n):
        r, p = _uf_find(parent, rel, x)
        flip[x] = p
        roots[x] = r
    return flip, roots, violations


def _bucket_face(by_size, size, ordered, cids):
    """Append a face (ordered node array) and its owning cell ids into the
    per-size accumulator ``by_size`` (size -> ([ordered, ...], [cids, ...]))."""
    o, c = by_size.setdefault(size, ([], []))
    o.append(np.asarray(ordered, dtype=np.int64))
    c.append(cids)


def consistent_orientation_flips(blocks, points):
    """Compute per-block boolean flip masks that make 3D-cell orientation
    globally consistent for MED.

    blocks: list of dicts, in the order cells are laid out, each either
        {"kind": "regular", "type": <med cell type>, "conn": (n,k) int array}
        {"kind": "polyhedron", "faces": [ [face_node_array, ...], ... ]}
      All node ids are 0-based, in the final MED node ordering.
    points: (n_points, 3) float array.

    Returns a list (aligned with blocks) of boolean masks (True = flip that
    cell), or None if there is nothing to do or the mesh is already consistent.
    """
    starts = []
    counts = []
    gid = 0
    for blk in blocks:
        starts.append(gid)
        if blk["kind"] == "regular":
            c = len(blk["conn"])
        else:
            c = len(blk["faces"])
        counts.append(c)
        gid += c
    n_cells = gid
    if n_cells == 0:
        return None

    # enumerate faces grouped by size -> ordered arrays + owning cell id
    by_size = {}
    for blk, start in zip(blocks, starts):
        if blk["kind"] == "regular":
            conn = blk["conn"]
            ids = np.arange(start, start + len(conn), dtype=np.int64)
            for size, ofaces in _cell_faces(blk["type"], conn):
                _bucket_face(by_size, size, ofaces, ids)
        else:  # polyhedron: use its stored outward faces
            for j, faces in enumerate(blk["faces"]):
                cid = start + j
                for face in faces:
                    face = np.asarray(face, dtype=np.int64)
                    _bucket_face(by_size, len(face), face[None, :],
                                 np.array([cid], np.int64))

    # build shared-face parity constraints
    eA, eB, eP = [], [], []
    for size, (olist, clist) in by_size.items():
        ordered = np.concatenate(olist, 0)
        cids = np.concatenate(clist)
        key = np.sort(ordered, axis=1)
        order = np.lexsort(key.T[::-1])
        ks = key[order]
        cs = cids[order]
        ch = _chirality(ordered)[order]
        same = np.all(ks[1:] == ks[:-1], axis=1)
        pair = np.where(same)[0]
        if len(pair) == 0:
            continue
        eA.append(cs[pair])
        eB.append(cs[pair + 1])
        eP.append((ch[pair] == ch[pair + 1]).astype(np.int8))  # same wind -> 1
    if not eA:
        return None
    A = np.concatenate(eA)
    B = np.concatenate(eB)
    par = np.concatenate(eP)

    if int(par.sum()) == 0:
        return None  # already consistent, skip the union-find entirely

    flip, roots, _ = _solve_parity(n_cells, A, B, par)

    # A whole region can still come out inside-out. For each region, flip it if
    # most of its cells have negative volume.
    signed = np.zeros(n_cells)
    have_vol = np.zeros(n_cells, dtype=bool)
    for blk, start in zip(blocks, starts):
        if blk["kind"] == "regular":
            conn = blk["conn"]
            fp = _FLIP_PERM[blk["type"]]
            end = start + len(conn)
            f = flip[start:end].astype(bool)
            oriented = conn.copy()
            oriented[f] = conn[f][:, fp]
            signed[start:end] = _signed_volume(blk["type"], oriented, points)
            have_vol[start:end] = True

    order = np.argsort(roots, kind="stable")
    r_sorted = roots[order]
    bounds = np.flatnonzero(np.r_[True, r_sorted[1:] != r_sorted[:-1], True])
    comp_flip_root = {}
    votes = np.where(have_vol, np.sign(signed), 0.0)
    for i in range(len(bounds) - 1):
        seg = order[bounds[i]:bounds[i + 1]]
        root = roots[seg[0]]
        if votes[seg].sum() < 0:  # component mostly inside-out -> flip it
            comp_flip_root[root] = 1
    if comp_flip_root:
        extra = np.array([comp_flip_root.get(r, 0) for r in roots], dtype=np.int8)
        flip = flip ^ extra

    masks = []
    for blk, start, cnt in zip(blocks, starts, counts):
        masks.append(flip[start:start + cnt].astype(bool))
    return masks


def apply_regular_flip(cell_type, conn, mask):
    """Return conn with flipped rows reversed by the type's node permutation."""
    if mask is None or not mask.any():
        return conn
    out = conn.copy()
    out[mask] = conn[mask][:, _FLIP_PERM[cell_type]]
    return out


def apply_polyhedron_flip(faces_list, mask):
    """Return polyhedra with each flipped cell's faces reversed (winding
    flipped) so its outward normals invert consistently."""
    if mask is None or not mask.any():
        return faces_list
    out = []
    for poly, fl in zip(faces_list, mask):
        if fl:
            out.append([np.asarray(f)[::-1] for f in poly])
        else:
            out.append(poly)
    return out

"""
I/O for OpenFOAM polyMesh format (reader).

Supports both ASCII and binary (LSB, label=32, scalar=64) formats.
Handles general polyhedra in addition to tetra/pyramid/wedge/hexahedron.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from .._helpers import register_format
from .._mesh import CellBlock, Mesh

logger = logging.getLogger(__name__)


def _detect_format(path: Path) -> tuple[str, int, int]:
    """
    Reads the FoamFile header and returns (format, label_bytes, scalar_bytes).

    format       : 'ascii' or 'binary'
    label_bytes  : 4 (label=32) or 8 (label=64)
    scalar_bytes : 4 (scalar=32) or 8 (scalar=64)
    """
    fmt = "ascii"
    label_bytes = 8
    scalar_bytes = 8

    with open(path, "rb") as f:
        for raw in f:
            try:
                line = raw.decode("ascii", errors="replace").strip()
            except Exception:
                break

            m = re.match(r"format\s+(\w+)\s*;", line)
            if m:
                fmt = m.group(1).lower()

            m = re.match(r'arch\s+"([^"]+)"\s*;', line)
            if m:
                arch = m.group(1)
                ml = re.search(r"label=(\d+)", arch)
                ms = re.search(r"scalar=(\d+)", arch)
                if ml:
                    label_bytes = int(ml.group(1)) // 8
                if ms:
                    scalar_bytes = int(ms.group(1)) // 8

            # End of header
            if line == "}":
                break

    return fmt, label_bytes, scalar_bytes


class _RaggedArray:
    """
    Rows of variable length stored in CSR (compressed sparse row) form.

    A polyMesh has two ragged integer relations -- faces (node ids per face)
    and cell topology (face ids per cell) -- both of which would cost several
    gigabytes as a Python ``list[list[int]]`` on an industrial mesh (tens of
    millions of rows). CSR stores them as two flat numpy arrays instead:

    * ``conn`` -- every row's values concatenated back to back;
    * ``off``  -- length ``n_rows + 1``, so row ``i`` is
      ``conn[off[i]:off[i + 1]]``.

    Indexing (``a[i]``) and ``len(a)`` behave like the list-of-lists they
    replace, so consumers need no special-casing.
    """

    __slots__ = ("conn", "off")

    def __init__(self, conn: np.ndarray, off: np.ndarray):
        self.conn = conn
        self.off = off

    @classmethod
    def from_lists(cls, rows: list) -> "_RaggedArray":
        """Build from a Python list-of-lists (used for small ASCII inputs)."""
        if len(rows) == 0:
            return cls(np.empty(0, dtype=np.int64), np.zeros(1, dtype=np.int64))
        sizes = np.fromiter((len(r) for r in rows), dtype=np.int64, count=len(rows))
        off = np.empty(len(rows) + 1, dtype=np.int64)
        off[0] = 0
        np.cumsum(sizes, out=off[1:])
        conn = np.fromiter(
            (int(v) for r in rows for v in r), dtype=np.int64, count=int(off[-1])
        )
        return cls(conn, off)

    def __len__(self) -> int:
        return len(self.off) - 1

    def __getitem__(self, i: int) -> np.ndarray:
        return self.conn[self.off[i] : self.off[i + 1]]

    def sizes(self) -> np.ndarray:
        """Length of every row, as a numpy array."""
        return np.diff(self.off)

    def to_lists(self) -> list:
        """Materialise back to a Python list-of-lists (used by tests)."""
        return [
            self.conn[self.off[i] : self.off[i + 1]].tolist() for i in range(len(self))
        ]


def _data_start(raw: bytes) -> tuple[int, int]:
    """
    Return (N, offset just after the outer '(') for a binary OpenFOAM List.

    Layout (verified against real polyMesh output)::

        FoamFile { ... }
        // * * * ...
        N
        (<binary data ...

    For contiguous lists (vectorField, labelList) the binary data starts
    immediately after '('. ``N`` is the last integer between the header's
    closing '}' and that '('.
    """
    end = raw.find(b"}")  # end of the FoamFile header block
    if end == -1:
        raise ValueError("No FoamFile header found")
    lp = raw.find(b"(", end)  # '(' opening the outer list
    if lp == -1:
        raise ValueError("No '(' opening the data list found")
    nums = re.findall(rb"\d+", raw[end:lp])
    if not nums:
        raise ValueError("No element count found before '('")
    return int(nums[-1]), lp + 1


def _read_binary_points(path: Path, scalar_bytes: int = 8) -> np.ndarray:
    """Binary vectorField:  N ( <N*3*scalar bytes binary> )."""
    raw = path.read_bytes()
    n, start = _data_start(raw)

    dtype = "<f4" if scalar_bytes == 4 else "<f8"
    expected = n * 3 * scalar_bytes
    if len(raw) - start < expected:
        raise ValueError(
            f"points: expected {expected} bytes, got {len(raw) - start} "
            f"(n={n}, start={start}, file_size={len(raw)})"
        )
    return (
        np.frombuffer(raw, dtype=dtype, count=n * 3, offset=start)
        .reshape(n, 3)
        .astype(float)
    )


def _read_binary_labels(path: Path, label_bytes: int = 4) -> np.ndarray:
    """Binary labelList (owner/neighbour):  N ( <N*label bytes binary> )."""
    raw = path.read_bytes()
    n, start = _data_start(raw)

    dtype = "<i4" if label_bytes == 4 else "<i8"
    expected = n * label_bytes
    if len(raw) - start < expected:
        raise ValueError(
            f"labels: expected {expected} bytes, got {len(raw) - start} "
            f"(n={n}, start={start})"
        )
    return np.frombuffer(raw, dtype=dtype, count=n, offset=start).astype(np.int64)


def _read_binary_faces(path: Path, label_bytes: int = 4) -> _RaggedArray:
    """
    Binary OpenFOAM faceList -> CSR ``_RaggedArray``.

    A ``List<face>`` is non-contiguous, so each face is serialised as a
    ``labelList``::

        N
        (
        <M> ( <M*label bytes binary> )
        <M> ( <M*label bytes binary> )
        ...
        )

    Two passes, both memory bounded:

    1. Sequential scan recording, per face, its node count and the byte offset
       of its binary blob. ``raw.find(b"(")`` only ever scans the short ASCII
       gap between one face's ')' and the next face's '(' -- never the binary
       blob -- so binary bytes that happen to equal '(' are never misread.
    2. Vectorised gather of all blob bytes via a byte mask (the blob ranges are
       disjoint, so a +1/-1 diff + cumsum marks them), then a single
       ``view`` to the label dtype.
    """
    raw = path.read_bytes()
    nfaces, pos = _data_start(raw)

    counts = np.empty(nfaces, dtype=np.int32)
    offsets = np.empty(nfaces, dtype=np.int64)  # byte offset of each blob
    find = raw.find
    p = pos
    for i in range(nfaces):
        lp = find(b"(", p)  # scans only the ASCII gap
        if lp == -1:
            raise ValueError(f"faces: missing '(' for face {i}")
        counts[i] = int(raw[p:lp])  # ASCII count, tolerates ws
        blob = lp + 1
        offsets[i] = blob
        p = blob + int(counts[i]) * label_bytes + 1  # skip blob and ')'

    off = np.empty(nfaces + 1, dtype=np.int64)
    off[0] = 0
    np.cumsum(counts, out=off[1:])

    nbytes = counts.astype(np.int64) * label_bytes
    diff = np.zeros(len(raw) + 1, dtype=np.int8)
    np.add.at(diff, offsets, 1)
    np.add.at(diff, offsets + nbytes, -1)
    mask = np.cumsum(diff[:-1], dtype=np.int8).astype(bool)

    buf = np.frombuffer(raw, dtype=np.uint8)
    conn = buf[mask].view("<i4" if label_bytes == 4 else "<i8").astype(np.int64)
    return _RaggedArray(conn, off)


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//.*", "", text)
    return text


def _skip_header(lines: list[str]) -> list[str]:
    """Remove the FoamFile { ... } block."""
    in_header = False
    depth = 0
    result = []
    for line in lines:
        s = line.strip()
        if "FoamFile" in s:
            in_header = True
        if in_header:
            depth += s.count("{") - s.count("}")
            if depth <= 0:
                in_header = False
            continue
        result.append(line)
    return result


def _read_foam_lines(path: Path) -> list[str]:
    """Read a FoamFile, strip comments and header, return content lines."""
    text = _strip_comments(path.read_text(errors="replace"))
    return _skip_header(text.splitlines())


def _parse_points_ascii(lines: list[str]) -> np.ndarray:
    coords = []
    in_block = False
    n = None
    num = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if n is None and s.isdigit():
            n = int(s)
            continue
        if s == "(" and n is not None:
            in_block = True
            continue
        if s == ")" and in_block:
            break
        if in_block:
            nums = num.findall(s)
            if len(nums) == 3:
                coords.append([float(v) for v in nums])
    pts = np.array(coords, dtype=float)
    if n is not None and len(pts) != n:
        logger.warning("points: expected %d, parsed %d", n, len(pts))
    return pts


def _parse_faces_ascii(lines: list[str]) -> list[list[int]]:
    faces = []
    in_block = False
    n = None
    face_re = re.compile(r"(\d+)\s*\(([^)]+)\)")
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if n is None and s.isdigit():
            n = int(s)
            continue
        if s == "(" and n is not None:
            in_block = True
            continue
        if s == ")" and in_block:
            break
        if in_block:
            m = face_re.match(s)
            if m:
                try:
                    faces.append(list(map(int, m.group(2).split())))
                except ValueError:
                    logger.warning("Skipping malformed face line: %r", s)
    return faces


def _parse_int_list_ascii(lines: list[str]) -> np.ndarray:
    tokens = []
    in_block = False
    n = None
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if n is None and s.isdigit():
            n = int(s)
            continue
        if s == "(":
            in_block = True
            continue
        if s == ")":
            break
        if in_block:
            tokens.extend(s.split())
    return np.array(tokens, dtype=int)


def _parse_boundary(lines: list[str]) -> dict:
    """Returns {patch_name: {type, nFaces, startFace}}."""
    patches = {}
    flat = "\n".join(lines)
    pattern = re.compile(r"(\w+)\s*\{([^}]*)\}", re.DOTALL)
    for match in pattern.finditer(flat):
        name = match.group(1)
        body = match.group(2)

        def _get(key, body=body):
            m = re.search(rf"{key}\s+([^\s;]+)\s*;", body)
            return m.group(1) if m else None

        n_faces = _get("nFaces")
        start_face = _get("startFace")
        if n_faces is not None and start_face is not None:
            patches[name] = {
                "type": _get("type"),
                "nFaces": int(n_faces),
                "startFace": int(start_face),
            }
    return patches


def _read_points(path: Path) -> np.ndarray:
    fmt, label_bytes, scalar_bytes = _detect_format(path)
    if fmt == "binary":
        logger.info("Reading binary points from %s", path.name)
        return _read_binary_points(path, scalar_bytes)
    return _parse_points_ascii(_read_foam_lines(path))


def _read_faces(path: Path) -> _RaggedArray:
    fmt, label_bytes, scalar_bytes = _detect_format(path)
    if fmt == "binary":
        logger.info("Reading binary faces from %s (label=%d B)", path.name, label_bytes)
        return _read_binary_faces(path, label_bytes)
    return _RaggedArray.from_lists(_parse_faces_ascii(_read_foam_lines(path)))


def _read_int_list(path: Path) -> np.ndarray:
    fmt, label_bytes, scalar_bytes = _detect_format(path)
    if fmt == "binary":
        logger.info(
            "Reading binary labels from %s (label=%d B)", path.name, label_bytes
        )
        return _read_binary_labels(path, label_bytes)
    return _parse_int_list_ascii(_read_foam_lines(path))


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _triple(a, b, c) -> float:
    """Scalar triple product a · (b × c)."""
    return float(np.dot(a, np.cross(b, c)))


def _node_adjacency(faces) -> dict:
    """Build a node-to-node adjacency dict from a list of faces."""
    adj: dict[int, set] = {}
    for f in faces:
        m = len(f)
        for i in range(m):
            a, b = f[i], f[(i + 1) % m]
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
    return adj


def _cell_faces_csr(n_cells, owner, neighbour) -> _RaggedArray:
    """
    Vectorised cell -> faces topology as a CSR :class:`_RaggedArray`.

    Row ``c`` holds the ids of every face touching cell ``c``. Handles both
    neighbour conventions: length nInternalFaces (standard OpenFOAM), or
    length nFaces with -1 on boundary faces.
    """
    if n_cells == 0:
        return _RaggedArray(np.empty(0, dtype=np.int64), np.zeros(1, dtype=np.int64))

    internal = neighbour >= 0
    cell_of = np.concatenate([owner, neighbour[internal]])
    face_of = np.concatenate(
        [np.arange(len(owner)), np.arange(len(neighbour))[internal]]
    )
    order = np.argsort(cell_of, kind="stable")  # group faces by cell id
    cf_flat = face_of[order]

    fpc = np.bincount(cell_of, minlength=n_cells)
    cf_off = np.empty(n_cells + 1, dtype=np.int64)
    cf_off[0] = 0
    np.cumsum(fpc, out=cf_off[1:])
    return _RaggedArray(cf_flat, cf_off)


def _outward_faces(cell_faces, faces, owner, cell_id):
    """
    Returns the faces of the cell oriented outward.

    The stored normal points from owner to neighbour (outward from owner).
    For the neighbour cell, the winding is reversed.
    """
    oriented = []
    for fid in cell_faces:
        f = faces[fid]
        oriented.append(list(f) if int(owner[fid]) == cell_id else list(reversed(f)))
    return oriented


def _match_top(bottom, oriented):
    """
    For each base node, find its unique vertical neighbour.
    Returns the ordered top ring, or None if the topology is ambiguous.
    """
    adj = _node_adjacency(oriented)
    base = set(bottom)
    top = []
    for b in bottom:
        cand = [x for x in adj[b] if x not in base]
        if len(cand) != 1:
            return None
        top.append(cand[0])
    return top


def _build_tetra(oriented, P):
    """Build a tetrahedron connectivity with positive volume orientation."""
    base = oriented[0]
    apex = (set().union(*oriented) - set(base)).pop()
    n = [base[0], base[1], base[2], apex]
    p = [P[i] for i in n]
    if _triple(p[1] - p[0], p[2] - p[0], p[3] - p[0]) < 0:
        n = [base[0], base[2], base[1], apex]
    return n


def _build_pyramid(oriented, P):
    """Build a pyramid connectivity with positive volume orientation."""
    quad = next(f for f in oriented if len(f) == 4)
    apex = (set().union(*oriented) - set(quad)).pop()
    n = list(quad) + [apex]
    p = [P[i] for i in n]
    if _triple(p[1] - p[0], p[3] - p[0], p[4] - p[0]) < 0:
        n = [quad[0], quad[3], quad[2], quad[1], apex]
    return n


def _build_wedge(oriented, P):
    """Build a wedge connectivity with positive volume orientation."""
    bottom = next(f for f in oriented if len(f) == 3)
    top = _match_top(bottom, oriented)
    if top is None:
        return None
    n = list(bottom) + top
    p = [P[i] for i in n]
    if _triple(p[1] - p[0], p[2] - p[0], p[3] - p[0]) < 0:
        n = [bottom[0], bottom[2], bottom[1], top[0], top[2], top[1]]
    return n


def _build_hexahedron(oriented, P):
    """Build a hexahedron connectivity with positive volume orientation."""
    bottom = next(f for f in oriented if len(f) == 4)
    top = _match_top(bottom, oriented)
    if top is None:
        return None
    n = list(bottom) + top
    p = [P[i] for i in n]
    if _triple(p[1] - p[0], p[3] - p[0], p[4] - p[0]) < 0:
        n = [bottom[0], bottom[3], bottom[2], bottom[1], top[0], top[3], top[2], top[1]]
    return n


def _build_boundary_polygons(poly_faces, poly_tags):
    """Split boundary polygons by vertex count -> polygonN CellBlocks."""
    by_n = defaultdict(list)
    tag_n = defaultdict(list)
    for f, t in zip(poly_faces, poly_tags):
        by_n[len(f)].append(list(f))
        tag_n[len(f)].append(t)
    cells, tags = [], []
    for n, faces in by_n.items():
        cells.append(CellBlock(f"polygon{n}", np.array(faces, dtype=int)))
        tags.append(np.array(tag_n[n], dtype=int))
    return cells, tags


def _build_polyhedra(poly_cells):
    """Split general polyhedra by unique node count -> polyhedronN CellBlocks."""
    by_n = defaultdict(list)
    for oriented in poly_cells:
        n_nodes = len(set().union(*oriented))
        by_n[n_nodes].append([list(f) for f in oriented])
    cells = []
    for n_nodes, polys in by_n.items():
        data = np.empty(len(polys), dtype=object)
        for i, p in enumerate(polys):
            data[i] = [np.array(f, dtype=int) for f in p]
        cells.append(CellBlock(f"polyhedron{n_nodes}", data))
    return cells


def _reconstruct_cell(oriented, P):
    """
    Classify a cell by (n_faces, n_points).

    Returns (meshio_type, connectivity) where:
      - for standard types : connectivity is a flat list of point ids
      - for 'polyhedron'   : connectivity is the list of outward-oriented faces
    """
    n_faces = len(oriented)
    n_pts = len(set().union(*oriented))

    if n_faces == 4 and n_pts == 4:
        return "tetra", _build_tetra(oriented, P)
    if n_faces == 5 and n_pts == 5:
        return "pyramid", _build_pyramid(oriented, P)
    if n_faces == 5 and n_pts == 6:
        return "wedge", _build_wedge(oriented, P)
    if n_faces == 6 and n_pts == 8:
        return "hexahedron", _build_hexahedron(oriented, P)

    # General polyhedron: keep outward-oriented faces
    return "polyhedron", oriented


def _build_volume_cells(n_cells, faces, owner, neighbour, P):
    """
    Build volume CellBlocks from raw polyMesh data.

    Uses a vectorised CSR cell -> faces topology and reads each face from the
    CSR ``_RaggedArray`` (or a plain list of faces), so peak memory stays bounded
    even for meshes with millions of cells. The per-cell classification reuses
    the orientation-aware reconstruction helpers unchanged.
    """
    cell_faces = _cell_faces_csr(n_cells, owner, neighbour)
    owner = np.asarray(owner)

    buckets: dict[str, list] = {}
    poly_cells: list = []
    n_skipped = 0

    for cell_id in range(n_cells):
        oriented = [
            list(faces[f]) if int(owner[f]) == cell_id else list(faces[f])[::-1]
            for f in cell_faces[cell_id]
        ]
        mtype, conn = _reconstruct_cell(oriented, P)

        if mtype == "polyhedron":
            poly_cells.append(conn)
        elif conn is None:
            n_skipped += 1
        else:
            buckets.setdefault(mtype, []).append(conn)

    if n_skipped:
        logger.warning("%d cell(s) skipped (degenerate topology).", n_skipped)
    if poly_cells:
        logger.info("%d general polyhedron cell(s) found.", len(poly_cells))

    cells = [CellBlock(t, np.array(c, dtype=int)) for t, c in buckets.items()]

    if poly_cells:
        cells.extend(_build_polyhedra(poly_cells))

    return cells


def _build_boundary_cells(boundary, faces):
    """
    Group boundary faces by geometric type.

    Triangles and quads  -> regular 2-D CellBlocks.
    Polygons (n > 4)     -> grouped by vertex count via _build_boundary_polygons.
    """
    by_size: dict[int, list] = {3: [], 4: []}
    tags_by_size: dict[int, list] = {3: [], 4: []}
    poly_faces: list = []
    poly_tags: list = []
    patch_tags: dict = {}

    for patch_id, (name, info) in enumerate(boundary.items()):
        fam = -(patch_id + 1)  # MED family id (negative)
        patch_tags[fam] = [name]
        for fid in range(info["startFace"], info["startFace"] + info["nFaces"]):
            if fid >= len(faces):
                continue
            f = faces[fid]
            if len(f) == 3:
                by_size[3].append(f)
                tags_by_size[3].append(fam)
            elif len(f) == 4:
                by_size[4].append(f)
                tags_by_size[4].append(fam)
            else:
                poly_faces.append(f)
                poly_tags.append(fam)

    cells: list = []
    tags: list = []

    for size, mtype in ((3, "triangle"), (4, "quad")):
        if by_size[size]:
            cells.append(CellBlock(mtype, np.array(by_size[size], dtype=int)))
            tags.append(np.array(tags_by_size[size], dtype=int))

    if poly_faces:
        logger.info("%d boundary polygon(s) with n>4 nodes found.", len(poly_faces))
        poly_cells, poly_tag_arrays = _build_boundary_polygons(poly_faces, poly_tags)
        cells.extend(poly_cells)
        tags.extend(poly_tag_arrays)

    return cells, tags, patch_tags


# ---------------------------------------------------------------------------
# polyMesh path resolution
# ---------------------------------------------------------------------------


def _resolve_polymesh(path: Path) -> Path:
    """Locate the polyMesh directory from a .foam file, case dir, or polyMesh dir."""
    if path.suffix == ".foam":
        c = path.parent / "constant" / "polyMesh"
        if c.exists():
            return c
    if path.name == "polyMesh" and path.is_dir():
        return path
    for c in (path / "constant" / "polyMesh", path / "polyMesh"):
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Could not locate polyMesh from '{path}'. "
        "Expected <case>/constant/polyMesh/."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def read(filename) -> Mesh:
    """Read an OpenFOAM polyMesh case and return a :class:`meshlane.Mesh`."""
    poly = _resolve_polymesh(Path(filename))
    logger.info("Reading polyMesh from %s", poly)

    points = _read_points(poly / "points")
    faces = _read_faces(poly / "faces")
    owner = _read_int_list(poly / "owner")
    neighbour = (
        _read_int_list(poly / "neighbour")
        if (poly / "neighbour").exists()
        else np.array([], dtype=int)
    )
    boundary = (
        _parse_boundary(_read_foam_lines(poly / "boundary"))
        if (poly / "boundary").exists()
        else {}
    )

    n_cells = (
        int(max(owner.max(initial=-1), neighbour.max(initial=-1))) + 1
        if len(owner)
        else 0
    )
    logger.info(
        "%d points, %d faces, %d cells, %d patches",
        len(points),
        len(faces),
        n_cells,
        len(boundary),
    )

    vol_cells = _build_volume_cells(n_cells, faces, owner, neighbour, points)
    patch_cells, patch_tags_data, patch_tags = _build_boundary_cells(boundary, faces)

    cells = vol_cells + patch_cells

    cell_data_tags = [np.zeros(len(cb.data), dtype=int) for cb in vol_cells]
    cell_data_tags.extend(patch_tags_data)

    mesh = Mesh(
        points=points,
        cells=cells,
        cell_data={"cell_tags": cell_data_tags} if cell_data_tags else {},
    )
    mesh.cell_tags = patch_tags
    mesh.point_tags = {}
    return mesh


register_format("openfoam", [".foam"], read, {})

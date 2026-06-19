"""
I/O for OpenFOAM polyMesh format (reader).

Supports both ASCII and binary (LSB, label=32, scalar=64) formats.
Handles general polyhedra in addition to tetra/pyramid/wedge/hexahedron.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from collections import defaultdict

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
    fmt          = "ascii"
    label_bytes  = 8
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
                    label_bytes  = int(ml.group(1)) // 8
                if ms:
                    scalar_bytes = int(ms.group(1)) // 8

            # End of header
            if line == "}":
                break

    return fmt, label_bytes, scalar_bytes


def _find_n_and_data_start(
    raw: bytes,
    skip_paren: bool = False,
) -> tuple[int, int]:
    """
    Iterates through `raw` line by line (ASCII) to find:
        1. The end of the FoamFile header (line containing '}')
        2. The number N (first purely numeric line after the header)
        3. Optionally the line '(' and the empty line that may follow it
           (if skip_paren=True)
    Returns (N, offset_start_of_binary_data).
    """
    pos         = 0
    in_foam     = False
    depth       = 0
    header_done = False
    n           = None

    while pos < len(raw):
        nl = raw.find(b"\n", pos)
        if nl == -1:
            nl = len(raw) - 1

        line = raw[pos:nl].decode("ascii", errors="replace").strip()

        # Phase 1: consume the header
        if not header_done:
            if "FoamFile" in line:
                in_foam = True
            if in_foam:
                depth += line.count("{") - line.count("}")
                if depth <= 0 and "}" in line:
                    header_done = True
            pos = nl + 1
            continue

        # Phase 2: after the header
        # Skip empty lines and comments
        if not line or line.startswith("//") or line.startswith("/*"):
            pos = nl + 1
            continue

        # Find N
        if n is None and line.isdigit():
            n   = int(line)
            pos = nl + 1
            continue

        # After N
        if n is not None:
            if skip_paren and line == "(":
                pos = nl + 1
                # Skip optional empty line following "("
                nl2  = raw.find(b"\n", pos)
                if nl2 == -1:
                    nl2 = len(raw) - 1
                line2 = raw[pos:nl2].decode("ascii", errors="replace").strip()
                if not line2:
                    pos = nl2 + 1
                return n, pos
            if not skip_paren:
                # Data starts at this position
                return n, pos

        pos = nl + 1

    raise ValueError(
        f"Could not find data block (n={n}, skip_paren={skip_paren})"
    )


def _read_binary_points(path: Path, scalar_bytes: int = 8) -> np.ndarray:
    """
    Binary points structure:
        <ASCII header>
        N\\n
        <N * 3 * scalar bytes binary>   (no parentheses)
    """
    raw = path.read_bytes()
    n, data_start = _find_n_and_data_start(raw, skip_paren=False)

    dtype    = "<f4" if scalar_bytes == 4 else "<f8"
    expected = n * 3 * scalar_bytes
    pts_raw  = raw[data_start: data_start + expected]

    if len(pts_raw) != expected:
        raise ValueError(
            f"points: expected {expected} bytes, got {len(pts_raw)} "
            f"(n={n}, data_start={data_start}, file_size={len(raw)})"
        )

    return np.frombuffer(pts_raw, dtype=dtype).reshape(n, 3).astype(float).copy()


def _read_binary_labels(path: Path, label_bytes: int = 4) -> np.ndarray:
    """
    Binary owner/neighbour structure:
        <ASCII header>
        N\\n
        <N * label bytes binary>   (no parentheses)
    """
    raw = path.read_bytes()
    n, data_start = _find_n_and_data_start(raw, skip_paren=False)

    dtype    = "<i4" if label_bytes == 4 else "<i8"
    expected = n * label_bytes
    arr_raw  = raw[data_start: data_start + expected]

    if len(arr_raw) != expected:
        raise ValueError(
            f"labels: expected {expected} bytes, got {len(arr_raw)} "
            f"(n={n}, data_start={data_start})"
        )

    return np.frombuffer(arr_raw, dtype=dtype).astype(np.int64).copy()


def _read_binary_faces(path: Path, label_bytes: int = 4) -> list[list[int]]:
    """
    Binary OpenFOAM faces structure:
        <ASCII header>
        N\\n
        (\\n
        \\n                           <- optional empty line
        <n_pts_face_0>\\n             <- ASCII
        [n_pts * label bytes binary] <- points of face 0
        \\n                           <- separator
        <n_pts_face_1>\\n
        [n_pts * label bytes binary]
        ...
        )\\n
    """
    raw   = path.read_bytes()
    dtype = "<i4" if label_bytes == 4 else "<i8"

    n, pos = _find_n_and_data_start(raw, skip_paren=True)

    faces = []
    for _ in range(n):
        # Skip empty lines / separators
        while pos < len(raw):
            nl   = raw.find(b"\n", pos)
            if nl == -1:
                nl = len(raw) - 1
            line = raw[pos:nl].decode("ascii", errors="replace").strip()
            if line and line.isdigit():
                npts = int(line)
                pos  = nl + 1
                break
            if line == ")":
                return faces
            pos = nl + 1

        # Read npts binary labels
        nbytes = npts * label_bytes
        pts    = np.frombuffer(raw[pos: pos + nbytes], dtype=dtype).tolist()
        faces.append([int(p) for p in pts])
        pos   += nbytes

    return faces


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//.*",      "",  text)
    return text


def _skip_header(lines: list[str]) -> list[str]:
    """Remove the FoamFile { ... } block."""
    in_header = False
    depth     = 0
    result    = []
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
    coords   = []
    in_block = False
    n        = None
    num      = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
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
    faces    = []
    in_block = False
    n        = None
    face_re  = re.compile(r"(\d+)\s*\(([^)]+)\)")
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
    tokens   = []
    in_block = False
    n        = None
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
    flat    = "\n".join(lines)
    pattern = re.compile(r"(\w+)\s*\{([^}]*)\}", re.DOTALL)
    for match in pattern.finditer(flat):
        name = match.group(1)
        body = match.group(2)

        def _get(key, body=body):
            m = re.search(rf"{key}\s+([^\s;]+)\s*;", body)
            return m.group(1) if m else None

        n_faces    = _get("nFaces")
        start_face = _get("startFace")
        if n_faces is not None and start_face is not None:
            patches[name] = {
                "type":      _get("type"),
                "nFaces":    int(n_faces),
                "startFace": int(start_face),
            }
    return patches


def _read_points(path: Path) -> np.ndarray:
    fmt, label_bytes, scalar_bytes = _detect_format(path)
    if fmt == "binary":
        logger.info("Reading binary points from %s", path.name)
        return _read_binary_points(path, scalar_bytes)
    return _parse_points_ascii(_read_foam_lines(path))


def _read_faces(path: Path) -> list[list[int]]:
    fmt, label_bytes, scalar_bytes = _detect_format(path)
    if fmt == "binary":
        logger.info(
            "Reading binary faces from %s (label=%d B)", path.name, label_bytes
        )
        return _read_binary_faces(path, label_bytes)
    return _parse_faces_ascii(_read_foam_lines(path))


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


def _build_cell_to_faces(n_cells, owner, neighbour):
    """
    Returns cell_faces[i] = list of face ids touching cell i.

    Handles two conventions:
      - neighbour of length nInternalFaces (standard OpenFOAM)
      - neighbour of length nFaces with -1 for boundary faces
    """
    cell_faces = [[] for _ in range(n_cells)]
    n_nb = len(neighbour)
    for fid in range(len(owner)):
        cell_faces[int(owner[fid])].append(fid)
        if fid < n_nb and neighbour[fid] >= 0:
            cell_faces[int(neighbour[fid])].append(fid)
    return cell_faces


def _outward_faces(cell_faces, faces, owner, cell_id):
    """
    Returns the faces of the cell oriented outward.

    The stored normal points from owner to neighbour (outward from owner).
    For the neighbour cell, the winding is reversed.
    """
    oriented = []
    for fid in cell_faces:
        f = faces[fid]
        oriented.append(
            list(f) if int(owner[fid]) == cell_id else list(reversed(f))
        )
    return oriented


def _match_top(bottom, oriented):
    """
    For each base node, find its unique vertical neighbour.
    Returns the ordered top ring, or None if the topology is ambiguous.
    """
    adj  = _node_adjacency(oriented)
    base = set(bottom)
    top  = []
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
    n    = [base[0], base[1], base[2], apex]
    p    = [P[i] for i in n]
    if _triple(p[1] - p[0], p[2] - p[0], p[3] - p[0]) < 0:
        n = [base[0], base[2], base[1], apex]
    return n


def _build_pyramid(oriented, P):
    """Build a pyramid connectivity with positive volume orientation."""
    quad = next(f for f in oriented if len(f) == 4)
    apex = (set().union(*oriented) - set(quad)).pop()
    n    = list(quad) + [apex]
    p    = [P[i] for i in n]
    if _triple(p[1] - p[0], p[3] - p[0], p[4] - p[0]) < 0:
        n = [quad[0], quad[3], quad[2], quad[1], apex]
    return n


def _build_wedge(oriented, P):
    """Build a wedge connectivity with positive volume orientation."""
    bottom = next(f for f in oriented if len(f) == 3)
    top    = _match_top(bottom, oriented)
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
    top    = _match_top(bottom, oriented)
    if top is None:
        return None
    n = list(bottom) + top
    p = [P[i] for i in n]
    if _triple(p[1] - p[0], p[3] - p[0], p[4] - p[0]) < 0:
        n = [bottom[0], bottom[3], bottom[2], bottom[1],
             top[0],    top[3],    top[2],    top[1]]
    return n


def _build_boundary_polygons(poly_faces, poly_tags):
    """Split boundary polygons by vertex count -> polygonN CellBlocks."""
    by_n   = defaultdict(list)
    tag_n  = defaultdict(list)
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
    n_pts   = len(set().union(*oriented))

    if n_faces == 4 and n_pts == 4:
        return "tetra",      _build_tetra(oriented, P)
    if n_faces == 5 and n_pts == 5:
        return "pyramid",    _build_pyramid(oriented, P)
    if n_faces == 5 and n_pts == 6:
        return "wedge",      _build_wedge(oriented, P)
    if n_faces == 6 and n_pts == 8:
        return "hexahedron", _build_hexahedron(oriented, P)

    # General polyhedron: keep outward-oriented faces
    return "polyhedron", oriented


def _build_volume_cells(n_cells, faces, owner, neighbour, P):
    """Build volume CellBlocks from raw polyMesh data."""
    cell_faces_map = _build_cell_to_faces(n_cells, owner, neighbour)

    buckets:    dict[str, list] = {}
    poly_cells: list            = []
    n_skipped = 0

    for cell_id, cf in enumerate(cell_faces_map):
        oriented    = _outward_faces(cf, faces, owner, cell_id)
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

    cells = [
        CellBlock(t, np.array(c, dtype=int))
        for t, c in buckets.items()
    ]

    if poly_cells:
        cells.extend(_build_polyhedra(poly_cells))

    return cells


def _build_boundary_cells(boundary, faces):
    """
    Group boundary faces by geometric type.

    Triangles and quads  -> regular 2-D CellBlocks.
    Polygons (n > 4)     -> grouped by vertex count via _build_boundary_polygons.
    """
    by_size:      dict[int, list] = {3: [], 4: []}
    tags_by_size: dict[int, list] = {3: [], 4: []}
    poly_faces:   list            = []
    poly_tags:    list            = []
    patch_tags:   dict            = {}

    for patch_id, (name, info) in enumerate(boundary.items()):
        fam = -(patch_id + 1)   # MED family id (negative)
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
    tags:  list = []

    for size, mtype in ((3, "triangle"), (4, "quad")):
        if by_size[size]:
            cells.append(CellBlock(mtype, np.array(by_size[size], dtype=int)))
            tags.append(np.array(tags_by_size[size], dtype=int))

    if poly_faces:
        logger.info(
            "%d boundary polygon(s) with n>4 nodes found.", len(poly_faces)
        )
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

    points    = _read_points(poly / "points")
    faces     = _read_faces(poly / "faces")
    owner     = _read_int_list(poly / "owner")
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
        len(points), len(faces), n_cells, len(boundary),
    )

    vol_cells = _build_volume_cells(n_cells, faces, owner, neighbour, points)
    patch_cells, patch_tags_data, patch_tags = _build_boundary_cells(
        boundary, faces
    )

    cells = vol_cells + patch_cells

    cell_data_tags = [np.zeros(len(cb.data), dtype=int) for cb in vol_cells]
    cell_data_tags.extend(patch_tags_data)

    mesh = Mesh(
        points=points,
        cells=cells,
        cell_data={"cell_tags": cell_data_tags} if cell_data_tags else {},
    )
    mesh.cell_tags  = patch_tags
    mesh.point_tags = {}
    return mesh


register_format("openfoam", [".foam"], read, {})
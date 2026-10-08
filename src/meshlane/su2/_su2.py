"""
I/O SU2 mesh format
<https://su2code.github.io/docs_v7/Mesh-File/>

Boundary markers are read into ``cell_sets`` (one set per ``MARKER_TAG`` name)
and into the integer ``cell_data["su2:tag"]``. On writing, markers come from
``cell_sets``, else from the ``cell_tags`` families of the MED/OpenFOAM readers,
else from the first integer cell data (e.g. ``gmsh:physical``).

Multi-zone files (``NZONE``/``IZONE``) hold one independent mesh per zone: they
are read and written with ``read_multi`` / ``write_multi`` as a list of meshes,
like the multi-mesh MED functions. ``read`` rejects them.
"""

import re

import numpy as np

from .._common import warn
from .._exceptions import ReadError, WriteError
from .._files import open_file
from .._helpers import register_format
from .._mesh import CellBlock, Mesh

# follows VTK conventions
su2_type_to_numnodes = {
    3: 2,  # line
    5: 3,  # triangle
    9: 4,  # quad
    10: 4,  # tetra
    12: 8,  # hexahedron
    13: 6,  # wedge
    14: 5,  # pyramid
}
su2_to_meshio_type = {
    3: "line",
    5: "triangle",
    9: "quad",
    10: "tetra",
    12: "hexahedron",
    13: "wedge",
    14: "pyramid",
}
meshio_to_su2_type = {v: k for k, v in su2_to_meshio_type.items()}
# linear type of each (possibly higher-order) meshio type
_linear_type = re.compile(r"(line|triangle|quad|tetra|pyramid|wedge|hexahedron)\d*$")

# element types of the domain (NELEM) and of the markers, per dimension
_volume_types = {
    2: ("triangle", "quad"),
    3: ("tetra", "hexahedron", "wedge", "pyramid"),
}
_boundary_types = {2: ("line",), 3: ("triangle", "quad")}

_keyword = re.compile(r"^([A-Za-z_]+)\s*=\s*(.*?)\s*$")
_default_marker = "boundary"
_mesh_keywords = {
    "NZONE",
    "IZONE",
    "NDIME",
    "NPOIN",
    "NELEM",
    "NMARK",
    "MARKER_TAG",
    "MARKER_ELEMS",
}


_zone_name = re.compile(r"^%\s*meshlane zone name:\s*(.*?)\s*$")


def read(filename):
    with open_file(filename, "r") as f:
        return read_buffer(f)


def read_buffer(f):
    zones = _split_zones(_lines(f))
    if len(zones) > 1:
        raise ReadError(
            f"SU2: multi-zone file ({len(zones)} zones). Use "
            "meshlane.su2.read_multi(), or convert it to MED "
            "(meshlane convert file.su2 file.med) to keep every zone."
        )
    return _read_lines(zones[0][1])


def read_multi(filename):
    """Read every zone of a (multi-zone) SU2 file.

    Returns ``(meshes, names)``: one :class:`Mesh` per ``IZONE`` section, and the
    zone names (``zone_<i>``, or the names ``write_multi`` stored).
    """
    with open_file(filename, "r") as f:
        zones = _split_zones(_lines(f))
    return [_read_lines(lines) for _, lines in zones], [name for name, _ in zones]


def _lines(f):
    text = f.read()
    if isinstance(text, bytes):
        text = text.decode()
    # splitlines() also handles CRLF (Windows) files
    return text.splitlines()


def _split_zones(lines):
    """[(zone name, lines of the zone)]; a single zone when there is no IZONE."""
    starts = []
    n_zones = None
    for k, line in enumerate(lines):
        match = _keyword.match(line.strip())
        if match is None:
            continue
        key = match.group(1).upper()
        if key == "IZONE":
            starts.append((k, match.group(2)))
        elif key == "NZONE" and n_zones is None:
            n_zones = int(match.group(2))
    if not starts:
        return [("zone_1", lines)]
    if n_zones is not None and n_zones != len(starts):
        warn(f"SU2: NZONE= {n_zones} but {len(starts)} IZONE sections found.")

    zones = []
    ends = [k for k, _ in starts[1:]] + [len(lines)]
    for (start, number), end in zip(starts, ends):
        body = lines[start + 1 : end]
        name = f"zone_{number}"
        for line in body[:3]:
            match = _zone_name.match(line.strip())
            if match:
                name = match.group(1)
        zones.append((name, body))
    return zones


def _read_lines(lines):
    dim = 0
    points = None
    volume = []  # {cell type: connectivity} of each NELEM section
    markers = []  # (name, tag, [{cell type: connectivity}, ...])
    expected_markers = 0
    tag = 0

    # Other sections (NPERIODIC, FFD_* boxes...) are kept verbatim: they are
    # rewritten as they are, as long as the points do not change.
    extra = []
    in_extra = False

    k = 0
    while k < len(lines):
        raw = lines[k]
        line = raw.strip()
        k += 1
        if not line or (line[0] == "%" and not in_extra):
            continue
        match = _keyword.match(line)
        key = match.group(1).upper() if match else None
        if key not in _mesh_keywords:
            if match or in_extra:
                in_extra = True
                extra.append(raw)
            else:
                warn(f"SU2: could not parse line '{line}', skipping it.")
            continue
        in_extra = False
        value = match.group(2)

        if key in ("NZONE", "IZONE"):
            continue  # zones are split before (_split_zones)
        elif key == "NDIME":
            dim = int(value)
            if dim not in (2, 3):
                raise ReadError(f"SU2: invalid dimension NDIME= {value}.")
        elif key == "NPOIN":
            if dim == 0:
                raise ReadError("SU2: NPOIN found before NDIME.")
            count = int(value.split()[0])
            points = _parse_points(lines[k : k + count], count, dim)
            k += count
        elif key in ("NELEM", "MARKER_ELEMS"):
            count = int(value)
            cells = _parse_elements(lines[k : k + count], count, key)
            k += count
            if key == "NELEM":
                volume.append(cells)
            elif not markers:
                raise ReadError("SU2: MARKER_ELEMS found before any MARKER_TAG.")
            else:
                markers[-1][2].append(cells)
        elif key == "NMARK":
            expected_markers = int(value)
        elif key == "MARKER_TAG":
            # numeric tags keep their value, names are numbered from the last tag
            try:
                tag = int(value)
            except ValueError:
                tag += 1
            markers.append((value, tag, []))

    if points is None:
        raise ReadError("SU2: no NPOIN section found.")
    if len(markers) != expected_markers:
        warn(
            f"SU2: expected {expected_markers} markers according to NMARK, "
            f"found {len(markers)}."
        )

    cells = []
    tags = []
    for section in volume:
        for cell_type, data in section.items():
            cells.append(CellBlock(cell_type, data))
            tags.append(np.zeros(len(data), dtype=np.int32))

    # one block per boundary cell type, all markers together
    n_volume = len(cells)
    boundary = {}  # cell type -> [(marker index, connectivity)]
    for m, (_, _, sections) in enumerate(markers):
        for section in sections:
            for cell_type, data in section.items():
                boundary.setdefault(cell_type, []).append((m, data))
    marker_of_cell = []
    for cell_type, parts in boundary.items():
        cells.append(CellBlock(cell_type, np.concatenate([d for _, d in parts])))
        which = np.concatenate([np.full(len(d), m) for m, d in parts])
        marker_of_cell.append(which)
        tags.append(np.array([markers[m][1] for m in which], dtype=np.int32))

    cell_sets = {}
    for m, (name, _, _) in enumerate(markers):
        members = [np.array([], dtype=int)] * n_volume
        members += [np.flatnonzero(which == m) for which in marker_of_cell]
        if name in cell_sets:
            members = [np.union1d(a, b) for a, b in zip(cell_sets[name], members)]
        cell_sets[name] = members

    mesh = Mesh(points, cells, cell_data={"su2:tag": tags}, cell_sets=cell_sets)
    if extra:
        mesh.su2_extra = "\n".join(extra) + "\n"
        mesh.su2_extra_npoin = len(points)
    return mesh


def _parse_points(lines, count, dim):
    if len(lines) < count:
        raise ReadError(f"SU2: expected {count} points, found {len(lines)}.")
    if count == 0:
        return np.empty((0, dim))
    # some files add one or two index columns after the coordinates
    n_columns = len(lines[0].split())
    values = np.array(" ".join(lines).split(), dtype=float)
    if n_columns < dim or values.size != count * n_columns:
        raise ReadError("SU2: inconsistent number of columns in the NPOIN section.")
    return values.reshape(count, n_columns)[:, :dim]


def _parse_elements(lines, count, key):
    """{cell type: connectivity} of ``count`` element lines ``type n0 n1 ...``,
    each optionally followed by an element index."""
    if len(lines) < count:
        raise ReadError(f"SU2: expected {count} elements in {key}, found {len(lines)}.")
    if count == 0:
        return {}
    lengths = np.fromiter((len(line.split()) for line in lines), dtype=int, count=count)
    flat = np.array(" ".join(lines).split(), dtype=np.int64)
    starts = np.cumsum(lengths) - lengths
    types = flat[starts]

    n_nodes = np.full(max(su2_type_to_numnodes) + 1, -1)
    for su2_type, n in su2_type_to_numnodes.items():
        n_nodes[su2_type] = n
    if types.min() < 0 or types.max() >= len(n_nodes) or np.any(n_nodes[types] < 0):
        unknown = sorted(set(types.tolist()) - set(su2_type_to_numnodes))
        raise ReadError(f"SU2: unknown element type(s) {unknown} in {key}.")
    extra = lengths - 1 - n_nodes[types]
    if np.any((extra < 0) | (extra > 1)):
        raise ReadError(f"SU2: invalid number of columns in {key}.")

    cells = {}
    for su2_type in np.unique(types):
        rows = starts[types == su2_type]
        n = su2_type_to_numnodes[int(su2_type)]
        cells[su2_to_meshio_type[int(su2_type)]] = flat[
            rows[:, None] + 1 + np.arange(n)
        ]
    return cells


def write(filename, mesh, float_fmt=".17g"):
    with open_file(filename, "w") as f:
        _write_zone(f, mesh, float_fmt)


def write_multi(filename, meshes, mesh_names=None, float_fmt=".17g", **kwargs):
    """Write several meshes as the zones of a multi-zone SU2 file.

    The zone names (``mesh_names``) are stored in comments, which SU2 ignores,
    so that ``read_multi`` gives them back.
    """
    if mesh_names is not None and len(mesh_names) != len(meshes):
        raise WriteError("SU2: one name per mesh expected.")
    with open_file(filename, "w") as f:
        f.write(f"NZONE= {len(meshes)}\n")
        for i, mesh in enumerate(meshes):
            f.write(f"IZONE= {i + 1}\n")
            if mesh_names is not None:
                f.write(f"% meshlane zone name: {mesh_names[i]}\n")
            _write_zone(f, mesh, float_fmt)


def _write_zone(f, mesh, float_fmt):
    points = np.asarray(mesh.points, dtype=float)

    # SU2 only has linear cells: quadratic ones are written with their corners
    written = []  # (block index, linear cell type, connectivity)
    unsupported, quadratic = set(), set()
    for b, cell_block in enumerate(mesh.cells):
        match = _linear_type.match(cell_block.type)
        if not match or cell_block.type == "vertex":
            unsupported.add(cell_block.type)
            continue
        linear = match.group(1)
        data = np.asarray(cell_block.data)
        if cell_block.type != linear:
            quadratic.add(cell_block.type)
            data = data[:, : su2_type_to_numnodes[meshio_to_su2_type[linear]]]
        written.append((b, linear, data))
    if unsupported:
        warn(
            f"SU2 does not support cells of type {', '.join(sorted(unsupported))}; skipped."
        )
    if quadratic:
        warn(
            f"SU2: only the corner nodes of {', '.join(sorted(quadratic))} cells "
            "are used."
        )

    dim = 3 if any(t in _volume_types[3] for _, t, _ in written) else 2
    volume = [(t, d) for _, t, d in written if t in _volume_types[dim]]
    boundary = [(b, t, d) for b, t, d in written if t in _boundary_types[dim]]
    _report_skipped(
        mesh,
        [
            (b, t)
            for b, t, _ in written
            if t not in (_volume_types[dim] + _boundary_types[dim])
        ],
        dim,
    )

    if dim == 2 and points.shape[1] == 3:
        # 2D meshes from Gmsh, Salome... come with z = 0
        if len(points) and np.ptp(points[:, 2]) > 0:
            raise WriteError(
                "SU2: a mesh without volume cells must be planar (constant z); "
                "surface meshes in 3D cannot be written."
            )
        points = points[:, :2]
    if points.shape[1] != dim:
        raise WriteError(f"SU2: {dim}D cells need {dim}D points.")

    markers = _markers(mesh, boundary)

    # drop the points no written cell uses (mid-edge nodes, skipped cells...)
    used = np.zeros(len(points), dtype=bool)
    for _, data in volume:
        used[data.ravel()] = True
    for _, parts in markers:
        for _, data in parts:
            used[data.ravel()] = True
    renumbered = not used.all()
    if renumbered:
        warn(f"SU2: {int(np.sum(~used))} point(s) not used by any cell dropped.")
        new_index = np.cumsum(used) - 1
        points = points[used]
        volume = [(t, new_index[d]) for t, d in volume]
        markers = [(n, [(t, new_index[d]) for t, d in parts]) for n, parts in markers]

    extra = getattr(mesh, "su2_extra", None)
    if extra and (renumbered or getattr(mesh, "su2_extra_npoin", None) != len(points)):
        warn(
            "SU2: the points changed, so the extra sections of the original file "
            "(periodicity, FFD boxes) that refer to them are not written."
        )
        extra = None

    f.write(f"NDIME= {dim}\n")
    f.write(f"NELEM= {sum(len(d) for _, d in volume)}\n")
    for cell_type, data in volume:
        f.write(_element_lines(cell_type, data))
    f.write(f"NPOIN= {len(points)}\n")
    fmt = " ".join([f"%{float_fmt}"] * dim) + "\n"
    f.write("".join(map(fmt.__mod__, map(tuple, points.tolist()))))
    f.write(f"NMARK= {len(markers)}\n")
    for name, parts in markers:
        f.write(f"MARKER_TAG= {name}\n")
        f.write(f"MARKER_ELEMS= {sum(len(data) for _, data in parts)}\n")
        for cell_type, data in parts:
            f.write(_element_lines(cell_type, data))
    if extra:
        f.write(extra)


def _report_skipped(mesh, skipped, dim):
    """Warn about cells neither in the domain nor on its boundary (lines in 3D)."""
    if not skipped:
        return
    types = sorted({t for _, t in skipped})
    blocks = {b for b, _ in skipped}
    groups = sorted(
        name
        for name, members in mesh.cell_sets.items()
        if any(
            b < len(members) and members[b] is not None and len(members[b])
            for b in blocks
        )
    )
    warn(
        f"SU2: {', '.join(types)} cells are neither domain nor boundary elements "
        f"of a {dim}D mesh; skipped"
        + (f" (groups: {', '.join(groups)})." if groups else ".")
    )


def _element_lines(cell_type, data):
    data = np.asarray(data)
    rows = np.column_stack([np.full(len(data), meshio_to_su2_type[cell_type]), data])
    fmt = " ".join(["%d"] * rows.shape[1]) + "\n"
    return "".join(map(fmt.__mod__, map(tuple, rows.tolist())))


def _marker_name(name):
    name = re.sub(r"\s+", "_", str(name).strip())
    return name or _default_marker


def _markers(mesh, boundary):
    """[(marker name, [(cell type, connectivity), ...]), ...] for boundary cells.

    ``boundary`` lists the ``(block index, cell type, connectivity)`` written.
    """
    blocks = [b for b, _, _ in boundary]
    names = []  # marker names, in order of first appearance
    origin = {}  # marker name -> first group name that gave it
    collisions = set()
    labels = {b: np.full(len(d), -1) for b, _, d in boundary}  # name index

    def index_of(name):
        marker = _marker_name(name)
        if marker not in names:
            names.append(marker)
            origin[marker] = str(name)
        elif origin[marker] != str(name):
            collisions.add(f"{origin[marker]!r} and {str(name)!r} -> {marker}")
        return names.index(marker)

    sets = {k: v for k, v in mesh.cell_sets.items() if not k.startswith("gmsh:")}
    tags = mesh.cell_data.get("cell_tags")
    families = getattr(mesh, "cell_tags", None) or {}
    if sets:
        n_overlaps = 0
        for name, members in sets.items():
            for b in blocks:
                if b >= len(members) or members[b] is None or len(members[b]) == 0:
                    continue
                idx = np.asarray(members[b], dtype=int)
                free = labels[b][idx] < 0
                n_overlaps += int(np.sum(~free))
                labels[b][idx[free]] = index_of(name)
        if n_overlaps:
            warn(
                f"SU2: {n_overlaps} boundary cell(s) belong to several groups; "
                "each is put in the first of its groups."
            )
    elif tags is not None and families:
        # markers in the order of the family table
        for tag, groups in families.items():
            if not groups:
                continue
            for b in blocks:
                selected = tags[b] == tag
                if np.any(selected):
                    labels[b][selected] = index_of(groups[0])
    else:
        # the first integer cell data defined on the boundary blocks
        candidates = [
            key
            for key, values in mesh.cell_data.items()
            if any(values[b] is not None for b in blocks)
            and all(
                values[b] is None or np.asarray(values[b]).dtype.kind in "iu"
                for b in blocks
            )
        ]
        if candidates:
            key = candidates[0]
            other = [k for k in mesh.cell_data if k != key]
            if other:
                warn(f"SU2: markers taken from {key}, ignoring {', '.join(other)}.")
            for b in blocks:
                values = mesh.cell_data[key][b]
                if values is None:
                    continue
                values = np.asarray(values).ravel()
                for tag in np.unique(values):
                    labels[b][values == tag] = index_of(tag)

    if any(np.any(labels[b] < 0) for b in blocks):
        default = index_of(_default_marker)
        for b in blocks:
            labels[b][labels[b] < 0] = default

    if collisions:
        warn(f"SU2: groups merged into one marker: {'; '.join(sorted(collisions))}.")

    data_of = {b: d for b, _, d in boundary}
    type_of = {b: t for b, t, _ in boundary}
    markers = []
    for k, name in enumerate(names):
        parts = []
        for b in blocks:
            selected = labels[b] == k
            if np.any(selected):
                parts.append((type_of[b], data_of[b][selected]))
        markers.append((name, parts))
    return markers


register_format("su2", [".su2"], read, {"su2": write})

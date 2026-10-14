"""
CGNS <https://cgns.org/>, HDF5 files (files in the older ADF format must first be
converted with ``cgnsconvert -h in.cgns out.cgns``).

The reader handles unstructured zones: standard elements (linear and quadratic),
MIXED sections, and polyhedra (NGON_n faces + NFACE_n cells, CGNS 3 and CGNS 4
layouts). The zones of the first base are merged into one mesh; when there are
several, each zone also becomes a cell set.
"""

import numpy as np

from .._common import warn
from .._exceptions import ReadError
from .._helpers import register_format
from .._mesh import CellBlock, Mesh
from ._elements import (
    MIXED,
    NFACE_n,
    NGON_n,
    nodes_per_element,
    ragged,
    renumber,
    split_mixed,
    to_meshlane,
)
from ._polyhedra import build_cells
from ._tree import child, children, data, file_version, name, resolve, text

_HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"


def read(filename):
    import h5py

    with open(filename, "rb") as fh:
        if not fh.read(8).startswith(_HDF5_SIGNATURE):
            raise ReadError(
                f"{filename} is not an HDF5 CGNS file (ADF files are not supported; "
                "convert them with: cgnsconvert -h in.cgns out.cgns)."
            )
    with h5py.File(filename, "r") as f:
        version = file_version(f)
        bases = children(f, "CGNSBase_t")
        if not bases:
            raise ReadError(f"{filename}: no CGNSBase_t node.")
        if len(bases) > 1:
            warn(f"{filename}: {len(bases)} CGNS bases, only the first is read.")
        try:
            return _read_base(bases[0], version)
        except (KeyError, ValueError, IndexError) as e:
            raise ReadError(f"{filename}: invalid CGNS file: {e}") from e


def _read_base(base, version):
    phys_dim = int(np.ravel(data(base))[1])
    zones = children(base, "Zone_t")
    if not zones:
        raise ReadError(f"Base {name(base)} has no zone.")
    steps = _time_step_zones(base)
    first_step = [z for z in zones if steps is not None and name(z) in steps]
    if first_step:
        # time series: each step has its own zones, read those of the first one
        zones = first_step
        warn(f"Base {name(base)} is a time series, only its first step is read.")

    points, cells, zone_blocks = [], [], {}
    offset = 0
    for zone in zones:
        zone_type = child(zone, "ZoneType_t")
        zone_type = text(zone_type).strip() if zone_type is not None else "Unstructured"
        if zone_type != "Unstructured":
            raise ReadError(f"Zone {name(zone)}: {zone_type} zones are not supported.")
        zone_points = _coordinates(zone, phys_dim)
        first_block = len(cells)
        for cell_type, conn, _ in _zone_cells(zone, zone_points, version):
            cells.append(CellBlock(cell_type, _shift(cell_type, conn, offset)))
        zone_blocks[name(zone)] = (first_block, len(cells))
        points.append(zone_points)
        offset += len(zone_points)

    cell_sets = {}
    if len(zones) > 1:
        for zone_name, (a, b) in zone_blocks.items():
            cell_sets[zone_name] = [
                np.arange(len(block.data)) if a <= i < b else np.array([], dtype=int)
                for i, block in enumerate(cells)
            ]
    return Mesh(np.concatenate(points), cells, cell_sets=cell_sets)


def _time_step_zones(base):
    """Names of the zones of the first time step, if the base is a time series
    whose zones change with time (BaseIterativeData/ZonePointers)."""
    iterative = child(base, "BaseIterativeData_t")
    if iterative is None or "ZonePointers" not in iterative:
        return None
    pointers = np.asarray(data(resolve(iterative["ZonePointers"])))
    if pointers.ndim != 3 or len(pointers) < 2:
        return None
    count = iterative.get("NumberOfZones")
    count = int(np.ravel(data(count))[0]) if count is not None else pointers.shape[1]
    return {
        bytes(row.astype(np.int8)).decode(errors="replace").strip("\x00 ")
        for row in pointers[0][:count]
    }


def _shift(cell_type, conn, offset):
    if offset == 0:
        return conn
    if cell_type.startswith("polyhedron"):
        out = np.empty(len(conn), dtype=object)
        for i, faces in enumerate(conn):
            out[i] = [face + offset for face in faces]
        return out
    return conn + offset


def _coordinates(zone, phys_dim):
    grids = children(zone, "GridCoordinates_t")
    if not grids:
        raise ReadError(f"Zone {name(zone)} has no GridCoordinates.")
    grid = next((g for g in grids if name(g) == "GridCoordinates"), grids[0])
    arrays = {name(a): data(a) for a in children(grid, "DataArray_t")}
    axes = [f"Coordinate{c}" for c in "XYZ"[:phys_dim]]
    if "CoordinateX" not in arrays:
        found = ", ".join(sorted(arrays))
        raise ReadError(
            f"Zone {name(zone)}: only Cartesian coordinates are supported ({found})."
        )
    num_points = np.asarray(arrays["CoordinateX"]).size
    return np.column_stack(
        [
            (
                np.asarray(arrays[a], dtype=float).ravel()
                if a in arrays
                else np.zeros(num_points)
            )
            for a in axes
        ]
    )


def _element_range(section):
    first, last = (int(x) for x in np.ravel(data(resolve(section["ElementRange"]))))
    return first, last


def _section_arrays(section):
    first, last = _element_range(section)
    conn = np.ravel(data(resolve(section["ElementConnectivity"])))
    offsets = section.get("ElementStartOffset")
    offsets = None if offsets is None else np.ravel(data(resolve(offsets)))
    return first, last, conn, offsets


def _zone_cells(zone, points, version):
    """Cell blocks of an unstructured zone, as (cell type, 0-based connectivity,
    CGNS element numbers)."""
    num_points = len(points)
    sections = sorted(children(zone, "Elements_t"), key=lambda s: _element_range(s)[0])
    blocks, ngon, nface = [], [], []
    for section in sections:
        code = renumber(np.ravel(data(section))[0], version)
        first, last, conn, offsets = _section_arrays(section)
        n = last - first + 1
        numbers = np.arange(first, last + 1)
        if code == NGON_n:
            ngon.append((numbers, *ragged(conn, n, offsets)))
        elif code == NFACE_n:
            nface.append((numbers, *ragged(conn, n, offsets)))
        elif code == MIXED:
            for sub, (local, nodes) in split_mixed(conn, n, offsets, version).items():
                cell_type, nodes = to_meshlane(sub, nodes)
                blocks.append((cell_type, nodes - 1, numbers[local]))
        else:
            k = nodes_per_element(code)
            if conn.size != n * k:
                raise ReadError(f"Section {name(section)} has the wrong size.")
            cell_type, nodes = to_meshlane(code, conn.reshape(n, k).astype(np.int64))
            blocks.append((cell_type, nodes - 1, numbers))

    if ngon:
        blocks.extend(_polygonal(points, ngon, nface))

    for cell_type, conn, _ in blocks:
        if cell_type.startswith("polyhedron"):
            continue
        if len(conn) and (conn.min() < 0 or conn.max() >= num_points):
            raise ReadError(f"Zone {name(zone)}: node index out of range.")
    return blocks


def _polygonal(points, ngon, nface):
    """Blocks of the NGON_n faces and NFACE_n cells of a zone."""
    ngon.sort(key=lambda s: s[0][0])
    face_numbers = np.concatenate([s[0] for s in ngon])
    face_nodes = np.concatenate([s[2] for s in ngon]) - 1
    sizes = np.concatenate([np.diff(s[1]) for s in ngon])
    face_ptr = np.concatenate([[0], np.cumsum(sizes)])
    if len(face_nodes) and (face_nodes.min() < 0 or face_nodes.max() >= len(points)):
        raise ReadError("NGON_n node index out of range.")

    if not nface:
        # no cells: the faces are the cells of a 2D zone, or a surface mesh
        return _face_blocks(
            face_ptr, face_nodes, np.arange(len(sizes)), None, face_numbers
        )

    nface.sort(key=lambda s: s[0][0])
    cell_numbers = np.concatenate([s[0] for s in nface])
    cell_ptr = np.concatenate(
        [[0], np.cumsum(np.concatenate([np.diff(s[1]) for s in nface]))]
    )
    signed = np.concatenate([s[2] for s in nface])
    refs = np.abs(signed)
    cell_faces = np.searchsorted(face_numbers, refs)
    if len(refs) and (
        cell_faces.max() >= len(face_numbers)
        or (face_numbers[cell_faces] != refs).any()
    ):
        raise ReadError("NFACE_n refers to faces that are not NGON_n elements.")

    points3 = (
        points
        if points.shape[1] == 3
        else np.column_stack([points, np.zeros((len(points), 3 - points.shape[1]))])
    )
    cells, boundary, reverse = build_cells(
        points3, face_ptr, face_nodes, cell_ptr, cell_faces, signed > 0
    )
    blocks = [(t, d, cell_numbers[ids]) for t, d, ids in cells]
    blocks.extend(
        _face_blocks(
            face_ptr, face_nodes, np.flatnonzero(boundary), reverse, face_numbers
        )
    )
    return blocks


def _face_blocks(face_ptr, face_nodes, faces, reverse, face_numbers):
    """triangle/quad/polygon<n> blocks of the given NGON_n faces, flipped where
    reverse is set."""
    sizes = np.diff(face_ptr)[faces]
    blocks = []
    for k in np.unique(sizes).tolist():
        sel = faces[sizes == k]
        nodes = face_nodes[face_ptr[sel][:, None] + np.arange(k)]
        if reverse is not None:
            nodes = np.where(reverse[sel][:, None], nodes[:, ::-1], nodes)
        cell_type = {2: "line", 3: "triangle", 4: "quad"}.get(k, f"polygon{k}")
        blocks.append((cell_type, nodes, face_numbers[sel]))
    return blocks


def write(filename, mesh, compression="gzip", compression_opts=4):
    import h5py

    f = h5py.File(filename, "w")

    base = f.create_group("Base")

    # TODO something is missing here

    zone1 = base.create_group("Zone1")
    coords = zone1.create_group("GridCoordinates")

    # write points
    coord_x = coords.create_group("CoordinateX")
    coord_x.create_dataset(
        " data",
        data=mesh.points[:, 0],
        compression=compression,
        compression_opts=compression_opts,
    )
    coord_y = coords.create_group("CoordinateY")
    coord_y.create_dataset(
        " data",
        data=mesh.points[:, 1],
        compression=compression,
        compression_opts=compression_opts,
    )
    coord_z = coords.create_group("CoordinateZ")
    coord_z.create_dataset(
        " data",
        data=mesh.points[:, 2],
        compression=compression,
        compression_opts=compression_opts,
    )

    # write cells
    # TODO write cells other than tetra
    elems = zone1.create_group("GridElements")
    rnge = elems.create_group("ElementRange")
    for cell_block in mesh.cells:
        if cell_block.type == "tetra":
            rnge.create_dataset(
                " data",
                data=[1, cell_block.data.shape[0]],
                compression=compression,
                compression_opts=compression_opts,
            )
    conn = elems.create_group("ElementConnectivity")
    for cell_block in mesh.cells:
        if cell_block.type == "tetra":
            conn.create_dataset(
                " data",
                data=cell_block.data.reshape(-1) + 1,
                compression=compression,
                compression_opts=compression_opts,
            )


register_format("cgns", [".cgns"], read, {"cgns": write})

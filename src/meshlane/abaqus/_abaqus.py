"""
I/O for Abaqus inp files.
"""

import pathlib

import numpy as np

from ..__about__ import __version__
from .._common import num_nodes_per_cell
from .._exceptions import ReadError
from .._files import open_file
from .._helpers import register_format
from .._mesh import CellBlock, Mesh

abaqus_to_meshio_type = {
    # trusses
    "T2D2": "line",
    "T2D2H": "line",
    "T2D3": "line3",
    "T2D3H": "line3",
    "T3D2": "line",
    "T3D2H": "line",
    "T3D3": "line3",
    "T3D3H": "line3",
    # beams
    "B21": "line",
    "B21H": "line",
    "B22": "line3",
    "B22H": "line3",
    "B31": "line",
    "B31H": "line",
    "B32": "line3",
    "B32H": "line3",
    "B33": "line3",
    "B33H": "line3",
    # surfaces
    "M3D3": "triangle",
    "SFM3D3": "triangle",
    "M3D4": "quad",
    "SFM3D4": "quad",
    "M3D6": "triangle6",
    "SFM3D6": "triangle6",
    "M3D8": "quad8",
    "SFM3D8": "quad8",
    "CPS4": "quad",
    "CPS4R": "quad",
    "S4": "quad",
    "S4R": "quad",
    "S4RS": "quad",
    "S4RSW": "quad",
    "S4R5": "quad",
    "S8R": "quad8",
    "S8R5": "quad8",
    "S9R5": "quad9",
    #
    "CPS3": "triangle",
    "STRI3": "triangle",
    "S3": "triangle",
    "S3R": "triangle",
    "S3RS": "triangle",
    "R3D3": "triangle",
    #
    "STRI65": "triangle6",
    # volumes
    "C3D8": "hexahedron",
    "C3D8H": "hexahedron",
    "C3D8I": "hexahedron",
    "C3D8IH": "hexahedron",
    "C3D8R": "hexahedron",
    "C3D8RH": "hexahedron",
    "C3D20": "hexahedron20",
    "C3D20H": "hexahedron20",
    "C3D20R": "hexahedron20",
    "C3D20RH": "hexahedron20",
    #
    "C3D4": "tetra",
    "C3D4H": "tetra4",
    "C3D10": "tetra10",
    "C3D10H": "tetra10",
    "C3D10I": "tetra10",
    "C3D10M": "tetra10",
    "C3D10MH": "tetra10",
    #
    "C3D6": "wedge",
    "C3D15": "wedge15",
    #
    # 4-node bilinear displacement and pore pressure
    "CAX4P": "quad",
    # 6-node quadratic
    "CPE6": "triangle6",
    "DCOUP3D": "vertex",
}
meshio_to_abaqus_type = {v: k for k, v in abaqus_to_meshio_type.items()}


def read(filename):
    """Reads a Abaqus inp file."""
    with open_file(filename, "r") as f:
        out = read_buffer(f)
    return out


def _build_id_resolver(id_map):
    """Turn a ``{abaqus_id: index}`` dict into sorted (keys, values) arrays
    usable with :func:`_resolve` for vectorized lookups."""
    n = len(id_map)
    keys = np.fromiter(id_map.keys(), dtype=np.int64, count=n)
    vals = np.fromiter(id_map.values(), dtype=np.int64, count=n)
    order = np.argsort(keys, kind="stable")
    return keys[order], vals[order]


def _resolve(keys_sorted, vals_sorted, data):
    """Map an array of abaqus ids to their internal indices, vectorized."""
    data = np.asarray(data)
    if data.size == 0:
        return data.astype(np.int64)
    if keys_sorted.size == 0:
        raise ReadError("reference to an id while no ids were defined")
    pos = np.clip(np.searchsorted(keys_sorted, data), 0, keys_sorted.size - 1)
    if not np.array_equal(keys_sorted[pos], data):
        raise ReadError("reference to an undefined id")
    return vals_sorted[pos]


def read_buffer(f):
    # nodes
    points = []
    point_ids = {}
    counter = 0

    # cells, grouped by meshio type 
    cell_types = []            # block order = order of first appearance
    cell_type_index = {}       # meshio type -> block index
    cell_rows = []             # per block: list of node-id rows (raw abaqus ids)
    elem_id_to_block = {}      # global element id -> block index
    elem_id_to_local = {}      # global element id -> local index within block

    # sets
    point_sets_raw = {}        # name -> array of node ids
    elset_order = []           # definition order (for by-name resolution)
    elset_from_element = {}    # name -> [element ids]  (ELSET= on *ELEMENT)
    elset_numeric = {}         # name -> array of element ids (explicit *ELSET)
    elset_byname = {}          # name -> [referenced set names]

    field_data = {}
    cell_data = {}
    point_data = {}
    point_sets = {}
    cell_sets = {}

    included = []              # meshes pulled in via *INCLUDE

    def add_elements(cell_type, rows, elset_name):
        if not rows:
            return
        b = cell_type_index.get(cell_type)
        if b is None:
            b = len(cell_types)
            cell_type_index[cell_type] = b
            cell_types.append(cell_type)
            cell_rows.append([])
        block = cell_rows[b]
        for r in rows:
            gid = r[0]
            elem_id_to_block[gid] = b
            elem_id_to_local[gid] = len(block)
            block.append(r[1:])
        if elset_name is not None:
            elset_from_element.setdefault(elset_name, []).extend(r[0] for r in rows)
            if elset_name not in elset_order:
                elset_order.append(elset_name)

    line = f.readline()
    while True:
        if not line:  # EOF
            break

        # Comments
        if line.startswith("**"):
            line = f.readline()
            continue

        keyword = line.partition(",")[0].strip().replace("*", "").upper()
        if keyword == "NODE":
            points, point_ids, counter, line = _read_nodes(
                f, points, point_ids, counter
            )
        elif keyword == "ELEMENT":
            params_map = get_param_map(line, required_keys=["TYPE"])
            cell_type, rows, line = _read_cells(f, params_map)
            add_elements(cell_type, rows, params_map.get("ELSET"))
        elif keyword == "NSET":
            params_map = get_param_map(line, required_keys=["NSET"])
            set_ids, _, line = _read_set(f, params_map)
            point_sets_raw[params_map["NSET"]] = set_ids
        elif keyword == "ELSET":
            params_map = get_param_map(line, required_keys=["ELSET"])
            set_ids, set_names, line = _read_set(f, params_map)
            name = params_map["ELSET"]
            if name not in elset_order:
                elset_order.append(name)
            if set_ids.size:
                elset_numeric[name] = set_ids
            else:
                elset_byname[name] = set_names
        elif keyword == "INCLUDE":
            # e.g. *INCLUDE,INPUT=wInclude_bulk.inp
            ext_input_file = pathlib.Path(line.split("=")[-1].strip())
            if not ext_input_file.exists():
                cd = pathlib.Path(f.name).parent
                ext_input_file = cd / ext_input_file
            out = read(ext_input_file)
            if len(out.points) > 0:
                included.append(out)
            line = f.readline()
        else:
            # There are just too many Abaqus keywords to explicitly skip them.
            line = f.readline()

    # finalize points & cells 
    points = np.asarray(points, dtype=float)
    node_keys, node_vals = _build_id_resolver(point_ids)

    cells = []
    for b, ctype in enumerate(cell_types):
        raw = np.array(cell_rows[b], dtype=np.int64)
        cells.append(CellBlock(ctype, _resolve(node_keys, node_vals, raw)))
    n_blocks = len(cells)

    #  helper: list of element ids -> per-block local-index arrays 
    def distribute(gids):
        per_block = [[] for _ in range(n_blocks)]
        for gid in gids:
            b = elem_id_to_block.get(gid)
            if b is not None:  # ids not belonging to any block are ignored
                per_block[b].append(elem_id_to_local[gid])
        return [np.array(x, dtype="int32") for x in per_block]

    # combine element-implied and explicit numeric elsets
    elset_gids = {}
    for name, gids in elset_from_element.items():
        elset_gids.setdefault(name, []).extend(gids)
    for name, gids in elset_numeric.items():
        elset_gids.setdefault(name, []).extend(int(g) for g in gids)

    # resolve cell sets in definition order (for by-name references) 
    ci_resolved = {}  # UPPER name -> stored name
    for name in elset_order:
        if name in elset_byname and name not in elset_gids:
            merged = [[] for _ in range(n_blocks)]
            for ref in elset_byname[name]:
                target = ci_resolved.get(ref.upper())
                if target is None:
                    raise ReadError(f"Unknown cell set '{ref}'")
                for b, arr in enumerate(cell_sets[target]):
                    merged[b].append(arr)
            cell_sets[name] = [
                np.concatenate(parts) if parts else np.array([], dtype="int32")
                for parts in merged
            ]
        else:
            cell_sets[name] = distribute(elset_gids.get(name, []))
        ci_resolved[name.upper()] = name

    #  node sets 
    for name, set_ids in point_sets_raw.items():
        point_sets[name] = _resolve(node_keys, node_vals, set_ids).astype("int32")

    # merge any *INCLUDE meshes 
    for out in included:
        points, cells = merge(out, points, cells, point_sets)
    if len(cells) > n_blocks:
        # pad existing cell sets with empty arrays for the appended blocks
        pad = len(cells) - n_blocks
        for v in cell_sets.values():
            v.extend(np.array([], dtype="int32") for _ in range(pad))

    return Mesh(
        points,
        cells,
        point_data=point_data,
        cell_data=cell_data,
        field_data=field_data,
        point_sets=point_sets,
        cell_sets=cell_sets,
    )


def _ends_block(line):
    """An Abaqus data block ends at a real keyword line (`*...`).

    `**` comment lines may appear *inside* a block (e.g. splitting a long
    *NODE list); they are not keywords and must not end the block.
    """
    return line.startswith("*") and not line.startswith("**")


def _read_nodes(f, points=None, point_ids=None, counter=0):
    if points is None:
        points = []
    else:
        points = list(points)
    if point_ids is None:
        point_ids = {}

    while True:
        line = f.readline()
        if not line:
            break
        if line.startswith("**"):
            continue
        if _ends_block(line):
            break
        if line.strip() == "":
            continue

        line = line.strip().split(",")
        point_id, coords = line[0], line[1:]
        point_ids[int(point_id)] = counter
        points.append([float(x) for x in coords])
        counter += 1

    return points, point_ids, counter, line


def _read_cells(f, params_map):
    # Abaqus element types are case-insensitive; normalise to match the map keys.
    etype = params_map["TYPE"].upper()
    if etype not in abaqus_to_meshio_type:
        raise ReadError(f"Element type not available: {etype}")

    cell_type = abaqus_to_meshio_type[etype]
    num_data = num_nodes_per_cell[cell_type] + 1  # ElementID + NodeIDs

    rows = []
    row = []
    while True:
        line = f.readline()
        if not line:
            break
        if line.startswith("**"):
            continue
        if _ends_block(line):
            break
        stripped = line.strip()
        if stripped == "":
            continue

        # Abaqus continues a data line when it ends with a comma.
        continues = stripped.endswith(",")
        row += [int(k) for k in filter(None, stripped.split(","))]
        if continues:
            continue

        # Complete line: id + nodes (+ optional extra columns, e.g. the beam
        # orientation node). Keep only what the connectivity needs.
        if len(row) < num_data:
            raise ReadError(
                f"{etype} ({cell_type}): element with {len(row)} fields, "
                f"expected at least {num_data}"
            )
        rows.append(row[:num_data])
        row = []

    return cell_type, rows, line


def merge(mesh, points, cells, point_sets):
    """Append an external :class:`Mesh` (from *INCLUDE) into the current
    ``points`` array and ``cells`` list, offsetting node indices."""
    ext_points = np.asarray(mesh.points, dtype=float)

    if len(points) > 0:
        offset = points.shape[0]
        points = np.concatenate([points, ext_points])
    else:
        offset = 0
        points = ext_points

    for c in mesh.cells:
        cells.append(CellBlock(c.type, np.asarray(c.data) + offset))

    for key, val in mesh.point_sets.items():
        point_sets[key] = np.asarray(val) + offset

    # Note: merging the external mesh's *cell* sets is not supported.
    return points, cells


def get_param_map(word, required_keys=None):
    """
    get the optional arguments on a line

    Example
    -------
    >>> word = 'elset,instance=dummy2,generate'
    >>> params = get_param_map(word, required_keys=['instance'])
    params = {
        'elset' : None,
        'instance' : 'dummy2,
        'generate' : None,
    }
    """
    if required_keys is None:
        required_keys = []
    words = word.split(",")
    param_map = {}
    for wordi in words:
        if "=" not in wordi:
            key = wordi.strip().upper()
            value = None
        else:
            sword = wordi.split("=")
            if len(sword) != 2:
                raise ReadError(sword)
            key = sword[0].strip().upper()
            value = sword[1].strip()
        param_map[key] = value

    msg = ""
    for key in required_keys:
        if key not in param_map:
            msg += f"{key} not found in {word}\n"
    if msg:
        raise RuntimeError(msg)
    return param_map


def _read_set(f, params_map):
    set_ids = []
    set_names = []
    while True:
        line = f.readline()
        if not line:
            break
        if line.startswith("**"):
            continue
        if _ends_block(line):
            break
        if line.strip() == "":
            continue

        line = line.strip().strip(",").split(",")
        if line[0].isnumeric():
            set_ids += [int(k) for k in line]
        else:
            # set defined from other sets, listed by name; a single line may
            # list several (case-insensitive resolution happens at the caller)
            set_names += [k for k in line if k]

    set_ids = np.array(set_ids, dtype="int32")
    if "GENERATE" in params_map:
        if len(set_ids) != 3:
            raise ReadError(set_ids)
        set_ids = np.arange(set_ids[0], set_ids[1] + 1, set_ids[2], dtype="int32")
    return set_ids, set_names, line


def write(
    filename, mesh: Mesh, float_fmt: str = ".16e", translate_cell_names: bool = True
) -> None:
    with open_file(filename, "wt") as f:
        f.write("*HEADING\n")
        f.write("Abaqus DataFile Version 6.14\n")
        f.write(f"written by meshlane v{__version__}\n")
        f.write("*NODE\n")
        fmt = ", ".join(["{}"] + ["{:" + float_fmt + "}"] * mesh.points.shape[1]) + "\n"
        for k, x in enumerate(mesh.points):
            f.write(fmt.format(k + 1, *x))
        eid = 0
        for cell_block in mesh.cells:
            cell_type = cell_block.type
            node_idcs = cell_block.data
            name = (
                meshio_to_abaqus_type[cell_type] if translate_cell_names else cell_type
            )
            f.write(f"*ELEMENT, TYPE={name}\n")
            for row in node_idcs:
                eid += 1
                nids_strs = (str(nid + 1) for nid in row.tolist())
                f.write(str(eid) + "," + ",".join(nids_strs) + "\n")

        nnl = 8
        offset = 0
        for ic in range(len(mesh.cells)):
            for k, v in mesh.cell_sets.items():
                if ic < len(v) and len(v[ic]) > 0:
                    els = [str(i + 1 + offset) for i in v[ic]]
                    f.write(f"*ELSET, ELSET={k}\n")
                    f.write(
                        ",\n".join(
                            ",".join(els[i : i + nnl]) for i in range(0, len(els), nnl)
                        )
                        + "\n"
                    )
            offset += len(mesh.cells[ic].data)

        for k, v in mesh.point_sets.items():
            nds = [str(i + 1) for i in v]
            f.write(f"*NSET, NSET={k}\n")
            f.write(
                ",\n".join(",".join(nds[i : i + nnl]) for i in range(0, len(nds), nnl))
                + "\n"
            )


register_format("abaqus", [".inp"], read, {"abaqus": write})
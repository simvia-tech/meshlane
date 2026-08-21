"""
I/O for Abaqus inp files.
"""

import io
import pathlib

import numpy as np

from ..__about__ import __version__
from .._common import num_nodes_per_cell, warn
from .._exceptions import ReadError
from .._files import is_buffer, open_file
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
}
meshio_to_abaqus_type = {v: k for k, v in abaqus_to_meshio_type.items()}

# Read-only aliases: thermal (DC*), acoustic (AC*) and gasket (GK*) elements
# share geometry with the displacement families. Added *after* the reverse map
# so the writer still emits the canonical C3D*/S*/... names.
abaqus_to_meshio_type.update(
    {
        "DC1D2": "line",
        "DC1D3": "line3",
        "DC2D3": "triangle",
        "DC2D6": "triangle6",
        "DC2D4": "quad",
        "DC2D8": "quad8",
        "DC3D4": "tetra",
        "DC3D10": "tetra10",
        "DC3D6": "wedge",
        "DC3D15": "wedge15",
        "DC3D8": "hexahedron",
        "DC3D20": "hexahedron20",
        "AC1D2": "line",
        "AC2D3": "triangle",
        "AC2D4": "quad",
        "AC2D6": "triangle6",
        "AC2D8": "quad8",
        "AC3D4": "tetra",
        "AC3D10": "tetra10",
        "AC3D6": "wedge",
        "AC3D8": "hexahedron",
        "AC3D20": "hexahedron20",
        "GK3D8": "hexahedron",
        "GK3D6": "wedge",
    }
)

# Non-mesh elements (connectors, springs, dashpots, masses, couplings, joints,
# gaps): not cells, skipped on read.
_NON_MESH_ELEMENTS = {"MASS", "ROTARYI"}
_NON_MESH_PREFIXES = ("CONN", "SPRING", "DASHPOT", "DCOUP", "JOINT", "GAP")


def _is_non_mesh(etype):
    return etype in _NON_MESH_ELEMENTS or etype.startswith(_NON_MESH_PREFIXES)


# Template for an instance that references no part (pure orphan mesh).
_EMPTY_PART = {
    "node_ids": np.array([], dtype=np.int64),
    "coords": np.zeros((0, 3)),
    "sections": [],
    "nsets": {},
    "elsets": {},
}


def read(filename):
    """Reads a Abaqus inp file."""
    # A passed-in buffer is read as-is.
    if is_buffer(filename, "r"):
        return read_buffer(filename)
    # For a path, decode with an encoding fallback so industrial .inp files
    # written in cp1252 (e.g. accented set names) are handled. This is done
    # here, not in the shared open_file, so binary readers that rely on a real
    # file descriptor (np.fromfile) keep working.
    with open(filename, "rb") as fb:
        raw = fb.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("cp1252", errors="replace")
    buf = io.StringIO(text)
    buf.name = str(filename)  # needed to resolve relative *INCLUDE paths
    return read_buffer(buf)


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

    # cells, one block for each Abaqus *ELEMENT section
    cell_types = []            # block order = order of first appearance
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

    def _add_block(cell_type, id_array, conn_array):
        # register one cell block from numpy id + connectivity arrays.
        # conn_array holds raw Abaqus node ids; they're converted to
        # points-array indices in one pass at the end (see finalize / _resolve).
        b = len(cell_types)
        cell_types.append(cell_type)
        cell_rows.append(conn_array)
        for i, gid in enumerate(id_array.tolist()):
            elem_id_to_block[gid] = b
            elem_id_to_local[gid] = i
        return b

    def add_elements(cell_type, rows, elset_name):
        if not rows:
            return
        conn = np.array([r[1:] for r in rows], dtype=np.int64)
        ids = np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))
        _add_block(cell_type, ids, conn)
        if elset_name is not None:
            elset_from_element.setdefault(elset_name, []).extend(ids.tolist())
            if elset_name not in elset_order:
                elset_order.append(elset_name)

    # ---- Abaqus assembly (parts + instances) support ----
    # A *Part numbers its nodes and elements locally (each part starts at 1), and
    # a *Instance drops a positioned copy of that part into the model. So the same
    # local id (eg. node 1) exists once per instance and would collide if stored
    # as-is. On each instantiation we therefore renumber every node and element to
    # a new, globally-unique id. Once ids are unique, an instance is just another
    # batch of nodes and cells: the ordinary finalize step (id resolver and sets)
    # handles it with no assembly-specific code.
    parts = {}            # part name -> template dict
    instantiated_parts = set()
    inst_maps = {}        # instance name -> (local_node_id->syn, local_elem_id->syn)
    asm_node_syn = {}     # assembly-level (orphan) node id -> syn
    syn = [1_000_000_000]

    def _add_nset(name, syn_ids):
        if not syn_ids:
            return
        arr = np.array(syn_ids, dtype=np.int64)
        prev = point_sets_raw.get(name)
        point_sets_raw[name] = np.concatenate([prev, arr]) if prev is not None else arr

    def _add_elset(name, syn_ids):
        if not syn_ids:
            return
        if name not in elset_order:
            elset_order.append(name)
        arr = np.array(syn_ids, dtype=np.int64)
        prev = elset_numeric.get(name)
        elset_numeric[name] = np.concatenate([prev, arr]) if prev is not None else arr

    def _instantiate(part, data_lines, iname):
        node_ids = np.asarray(part["node_ids"], dtype=np.int64)
        coords = np.asarray(part["coords"], dtype=float)
        n = len(node_ids)
        if coords.size:
            coords = _instance_transform(coords, data_lines)
        base = len(points)
        points.extend(coords.tolist())
        node_syn = np.arange(syn[0], syn[0] + n, dtype=np.int64)
        syn[0] += n
        point_ids.update(zip(node_syn.tolist(), range(base, base + n)))
        lid2syn = dict(zip(node_ids.tolist(), node_syn.tolist()))
        if n:
            order = np.argsort(node_ids)
            sorted_ids = node_ids[order]
            syn_sorted = node_syn[order]
        eid2syn = {}
        for ctype, ids_arr, conn_arr in part["sections"]:
            ids_arr = np.asarray(ids_arr, dtype=np.int64)
            conn_arr = np.asarray(conn_arr, dtype=np.int64)
            m = len(ids_arr)
            elem_syn = np.arange(syn[0], syn[0] + m, dtype=np.int64)
            syn[0] += m
            pos = np.searchsorted(sorted_ids, conn_arr.ravel())
            syn_conn = syn_sorted[pos].reshape(conn_arr.shape)
            _add_block(ctype, elem_syn, syn_conn)
            eid2syn.update(zip(ids_arr.tolist(), elem_syn.tolist()))
        for sname, ids in part["nsets"].items():
            _add_nset(sname, [lid2syn[i] for i in ids if i in lid2syn])
        for sname, eids in part["elsets"].items():
            _add_elset(sname, [eid2syn[e] for e in eids if e in eid2syn])
        inst_maps[iname] = (lid2syn, eid2syn)

    def _map_nset(iname, sname, ids):
        m = inst_maps[iname][0] if iname in inst_maps else asm_node_syn
        _add_nset(sname, [m[i] for i in ids if i in m])

    def _map_elset(iname, sname, ids):
        if iname in inst_maps:
            m = inst_maps[iname][1]
            _add_elset(sname, [m[i] for i in ids if i in m])

    def _read_assembly(line):
        line = f.readline()
        while line:
            if line.startswith("**"):
                line = f.readline()
                continue
            kw = line.partition(",")[0].strip().replace("*", "").upper()
            if kw == "END ASSEMBLY":
                line = f.readline()
                break
            if kw == "INSTANCE":
                pm = get_param_map(line, required_keys=["NAME"])
                iname, pname = pm.get("NAME"), pm.get("PART")
                data_lines = []
                line = f.readline()
                while line and not _ends_block(line):
                    if not line.startswith("**") and line.strip():
                        data_lines.append(
                            [float(x) for x in line.replace(",", " ").split()]
                        )
                    line = f.readline()
                # instance-level additions (Abaqus orphan-mesh style)
                ins_nid, ins_xyz, ins_sec = [], [], []
                ins_nsets, ins_elsets = {}, {}
                while line:
                    nkw = line.partition(",")[0].strip().replace("*", "").upper()
                    if nkw == "END INSTANCE":
                        line = f.readline()
                        break
                    if nkw == "NODE":
                        pts, pid, _, line = _read_nodes(f)
                        ins_nid.extend(pid.keys())
                        ins_xyz.extend(pts)
                    elif nkw == "ELEMENT":
                        pm2 = get_param_map(line, required_keys=["TYPE"])
                        ctype, rows, line = _read_cells(f, pm2)
                        if ctype is not None:
                            ins_sec.append(
                                (
                                    ctype,
                                    np.fromiter(
                                        (r[0] for r in rows),
                                        dtype=np.int64,
                                        count=len(rows),
                                    ),
                                    np.array([r[1:] for r in rows], dtype=np.int64),
                                )
                            )
                            els = pm2.get("ELSET")
                            if els:
                                ins_elsets.setdefault(els, []).extend(
                                    r[0] for r in rows
                                )
                    elif nkw == "NSET":
                        pm2 = get_param_map(line, required_keys=["NSET"])
                        ids, _, line = _read_set(f, pm2)
                        ins_nsets.setdefault(pm2["NSET"], []).extend(
                            int(i) for i in ids
                        )
                    elif nkw == "ELSET":
                        pm2 = get_param_map(line, required_keys=["ELSET"])
                        ids, _, line = _read_set(f, pm2)
                        ins_elsets.setdefault(pm2["ELSET"], []).extend(
                            int(i) for i in ids
                        )
                    else:
                        line = _skip_block(f)
                part = parts.get(pname)
                if not (ins_nid or ins_sec or ins_nsets or ins_elsets):
                    # mesh lives entirely in the part: instantiate it directly
                    # (no per-instance copy).
                    _instantiate(part or _EMPTY_PART, data_lines, iname)
                else:
                    p = part or _EMPTY_PART
                    if len(p["node_ids"]) == 0:
                        node_ids = np.array(ins_nid, dtype=np.int64)
                        coords = (
                            np.array(ins_xyz, dtype=float)
                            if ins_xyz
                            else np.zeros((0, 3))
                        )
                    elif ins_nid:
                        node_ids = np.concatenate(
                            [p["node_ids"], np.array(ins_nid, dtype=np.int64)]
                        )
                        coords = np.concatenate(
                            [p["coords"], np.array(ins_xyz, dtype=float)]
                        )
                    else:
                        node_ids, coords = p["node_ids"], p["coords"]
                    tmpl = {
                        "node_ids": node_ids,
                        "coords": coords,
                        "sections": list(p["sections"]) + ins_sec,
                        "nsets": {**p["nsets"], **ins_nsets},
                        "elsets": {**p["elsets"], **ins_elsets},
                    }
                    _instantiate(tmpl, data_lines, iname)
                if pname:
                    instantiated_parts.add(pname)
            elif kw == "NODE":
                pts, pid, _, line = _read_nodes(f)
                for lid in pid.keys():
                    sid = syn[0]
                    syn[0] += 1
                    point_ids[sid] = len(points)
                    points.append(pts[pid[lid]])
                    asm_node_syn[lid] = sid
            elif kw == "NSET":
                pm = get_param_map(line, required_keys=["NSET"])
                ids, _, line = _read_set(f, pm)
                _map_nset(pm.get("INSTANCE"), pm["NSET"], ids)
            elif kw == "ELSET":
                pm = get_param_map(line, required_keys=["ELSET"])
                ids, _, line = _read_set(f, pm)
                _map_elset(pm.get("INSTANCE"), pm["ELSET"], ids)
            else:
                line = f.readline()
        return line

    line = f.readline()
    while True:
        if not line:  # EOF
            break

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
        elif keyword == "PART":
            pname, pdata, line = _read_part(f, line)
            parts[pname] = pdata
        elif keyword == "ASSEMBLY":
            line = _read_assembly(line)
            counter = len(points)
        else:
            # There are just too many Abaqus keywords to explicitly skip them.
            line = f.readline()

    # a *Part defined but never placed by an *Instance (e.g. a part-only .inp
    # with no assembly) is still emitted, at identity.
    for pname, pdata in parts.items():
        if pname not in instantiated_parts:
            _instantiate(pdata, [], pname)

    # finalize points & cells
    points = np.asarray(points, dtype=float)
    node_keys, node_vals = _build_id_resolver(point_ids)

    cells = []
    for b, ctype in enumerate(cell_types):
        cells.append(CellBlock(ctype, _resolve(node_keys, node_vals, cell_rows[b])))
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
                    # A referenced set may belong to elements we skipped
                    # (e.g. connectors); keep what resolves instead of aborting.
                    warn(
                        f"Abaqus: cell set {name!r} references unknown set "
                        f"{ref!r}; skipping that reference."
                    )
                    continue
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


def _skip_block(f):
    # Read and discard the current data block; return the keyword line that ends
    # it (or an empty string at EOF).
    while True:
        line = f.readline()
        if not line or _ends_block(line):
            return line


def _read_cells(f, params_map):
    # Abaqus element types are case-insensitive; normalise to match the map keys.
    etype = params_map["TYPE"].upper()
    cell_type = abaqus_to_meshio_type.get(etype)
    if cell_type is None:
        # Not a mesh cell: known non-mesh elements (connectors, springs, masses,
        # couplings, ...) are skipped silently; unknown types are skipped with a
        # warning. Either way, consume the block so reading can continue.
        if not _is_non_mesh(etype):
            warn(f"Abaqus: skipping unsupported element type {etype!r}.")
        return None, [], _skip_block(f)

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
            set_names += [k.strip() for k in line if k.strip()]

    set_ids = np.array(set_ids, dtype="int32")
    if "GENERATE" in params_map:
        if len(set_ids) != 3:
            raise ReadError(set_ids)
        set_ids = np.arange(set_ids[0], set_ids[1] + 1, set_ids[2], dtype="int32")
    return set_ids, set_names, line


def _rotate_about_axis(coords, a, b, angle_deg):
    # Rotate points by angle_deg (degrees) about the axis from point a to point b.
    a = np.asarray(a, dtype=float)
    axis = np.asarray(b, dtype=float) - a
    norm = np.linalg.norm(axis)
    if norm == 0.0:
        return coords
    axis = axis / norm
    th = np.radians(angle_deg)
    p = coords - a
    c = np.cos(th)
    s = np.sin(th)
    return a + p * c + np.cross(axis, p) * s + np.outer(p @ axis, axis) * (1.0 - c)


def _instance_transform(coords, data_lines):
    # Abaqus *INSTANCE: an optional translation line, then an optional rotation
    # line (point a, point b, angle). Translation is applied first, then rotation.
    coords = np.asarray(coords, dtype=float)
    if coords.size == 0:
        return coords
    dim = coords.shape[1]
    if len(data_lines) >= 1 and len(data_lines[0]) >= dim:
        coords = coords + np.asarray(data_lines[0][:dim], dtype=float)
    if dim == 3 and len(data_lines) >= 2 and len(data_lines[1]) >= 7:
        d = data_lines[1]
        coords = _rotate_about_axis(coords, d[0:3], d[3:6], d[6])
    return coords


def _read_part(f, line):
    """Read a `*Part` block (local node/element numbering) into a template dict."""
    name = get_param_map(line, required_keys=["NAME"]).get("NAME")
    node_ids, coords = [], []
    sections = []          # [(cell_type, [(elem_id, [node_ids]), ...]), ...]
    nsets, elsets = {}, {}
    line = f.readline()
    while line:
        if line.startswith("**"):
            line = f.readline()
            continue
        kw = line.partition(",")[0].strip().replace("*", "").upper()
        if kw == "END PART":
            line = f.readline()
            break
        if kw == "NODE":
            pts, pid, _, line = _read_nodes(f)
            node_ids.extend(pid.keys())
            coords.extend(pts)
        elif kw == "ELEMENT":
            pm = get_param_map(line, required_keys=["TYPE"])
            ctype, rows, line = _read_cells(f, pm)
            if ctype is not None:
                sections.append((ctype, [(r[0], r[1:]) for r in rows]))
                els = pm.get("ELSET")
                if els:
                    elsets.setdefault(els, []).extend(r[0] for r in rows)
        elif kw == "NSET":
            pm = get_param_map(line, required_keys=["NSET"])
            ids, _, line = _read_set(f, pm)
            if ids.size:
                nsets.setdefault(pm["NSET"], []).extend(int(i) for i in ids)
        elif kw == "ELSET":
            pm = get_param_map(line, required_keys=["ELSET"])
            ids, _, line = _read_set(f, pm)
            if ids.size:
                elsets.setdefault(pm["ELSET"], []).extend(int(i) for i in ids)
        elif kw == "INCLUDE":
            # a part's mesh may live in an included file; pull in its
            # nodes/elements (sets from includes are not carried over).
            inc = pathlib.Path(line.split("=")[-1].strip())
            if not inc.exists() and hasattr(f, "name"):
                inc = pathlib.Path(f.name).parent / inc
            sub = read(inc)
            start = (max(node_ids) + 1) if node_ids else 1
            for i, p in enumerate(sub.points):
                node_ids.append(start + i)
                coords.append([float(x) for x in p])
            for cb in sub.cells:
                elems = [
                    (k + 1, [start + int(x) for x in row])
                    for k, row in enumerate(cb.data)
                ]
                sections.append((cb.type, elems))
            line = f.readline()
        else:
            line = f.readline()
    # store the template compactly (numpy) so many part templates can coexist
    # with the instantiated mesh without blowing up memory.
    np_sections = [
        (
            ct,
            np.fromiter((e[0] for e in elems), dtype=np.int64, count=len(elems)),
            np.array([e[1] for e in elems], dtype=np.int64),
        )
        for ct, elems in sections
    ]
    return name, {
        "node_ids": np.array(node_ids, dtype=np.int64),
        "coords": np.array(coords, dtype=float) if coords else np.zeros((0, 3)),
        "sections": np_sections,
        "nsets": nsets,
        "elsets": elsets,
    }, line


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

"""
Autonomous I/O for the Ansys MAPDL "coded database" format (.cdb / .inp).

This module reads AND writes the format by directly parsing MAPDL blocks
(ET/ETBLOCK, NBLOCK, EBLOCK, CMBLOCK) and converting them to/from
the neutral pivot object meshlane.Mesh. NO external dependencies, NO passing
through another format.

"""
import re

import numpy as np

from .._exceptions import ReadError, WriteError
from .._files import open_file
from .._helpers import register_format
from .._mesh import Mesh

# ----- Ansys type <-> meshlane type mappings -----
_FAMILY = {}
for _n in (5, 45, 70, 87, 90, 92, 95, 162, 185, 186, 187, 226, 227, 285):
    _FAMILY[_n] = "solid"
for _n in (28, 43, 63, 93, 131, 132, 181, 281):
    _FAMILY[_n] = "shell"
for _n in (25, 42, 77, 82, 182, 183, 223):
    _FAMILY[_n] = "plane"
for _n in (1, 3, 4, 21, 180, 188, 189, 288, 289):
    _FAMILY[_n] = "line"

_TO_MESHIO = {
    ("solid", 4): "tetra", ("solid", 10): "tetra10",
    ("solid", 8): "hexahedron", ("solid", 20): "hexahedron20",
    ("solid", 6): "wedge", ("solid", 15): "wedge15",
    ("solid", 5): "pyramid", ("solid", 13): "pyramid13",
    ("shell", 3): "triangle", ("shell", 6): "triangle6",
    ("shell", 4): "quad", ("shell", 8): "quad8",
    ("plane", 3): "triangle", ("plane", 6): "triangle6",
    ("plane", 4): "quad", ("plane", 8): "quad8",
    ("line", 2): "line", ("line", 3): "line3",
}

_FROM_MESHIO = {
    "tetra": 285, "tetra10": 187, "hexahedron": 185, "hexahedron20": 186,
    "wedge": 185, "wedge15": 186, "pyramid": 185, "pyramid13": 186,
    "triangle": 181, "triangle6": 281, "quad": 181, "quad8": 281,
    "line": 188, "line3": 189,
}


def _int_width(fmt):
    m = re.search(r"(\d+)i(\d+)", fmt, re.IGNORECASE)
    return int(m.group(2)) if m else 0


def _real_width(fmt):
    m = re.search(r"(\d+)[eg](\d+)\.", fmt, re.IGNORECASE)
    return int(m.group(2)) if m else 0


def _slice_ints(line, width):
    out = []
    line = line.rstrip("\n")
    for i in range(0, len(line), width):
        chunk = line[i:i + width].strip()
        if chunk:
            try:
                out.append(int(chunk))
            except ValueError:
                # Non numerical chunk, ignore (e.g. "R5.3" in "N,R5.3,LOC,...")
                break
    return out


def _slice_reals(s, width):
    out = []
    for i in range(0, len(s), width):
        chunk = s[i:i + width].strip()
        if chunk:
            out.append(float(chunk))
    return out

def _is_data_line(line):
    s = line.strip()
    if not s:
        return False
    
    up = s.upper()

    _KEYWORDS = (
        "FINISH", "NBLOCK", "EBLOCK", "CMBLOCK", "ETBLOCK",
        "/PREP7", "/SOLU", "/POST1", "/EOF",
        "KEYOPT", "MPDATA", "MPTEMP", "LOCAL", "SECBLOCK",
        "RLBLOCK", "DBLOCK", "FBLOCK", "SFEBLOCK",
    )
    for kw in _KEYWORDS:
        if up.startswith(kw):
            return False
        
    if re.match(r"^[A-Z]{1,8},", up):
        return False

    if s.startswith("!") or s.startswith("/"):
        return False

    return True
    

# READ
def read(filename):
    with open_file(filename, "r") as f:
        lines = f.read().splitlines()
    return _read_lines(lines)


def _read_lines(lines):
    etype_lib, node_id, coords, elements = {}, [], [], []
    node_comps, elem_comps = {}, {}
    saw_block = False
    i, n = 0, len(lines)
    while i < n:
        line = lines[i].strip()
        up = line.upper()

        if up.startswith("ET,"):
            p = line.split(",")
            if len(p) >= 3:
                try:
                    etype_lib[int(p[1])] = int(float(p[2]))
                except ValueError:
                    pass
            i += 1
        elif up.startswith("ETBLOCK"):
            saw_block = True
            ntypes = int(line.split(",")[1].split("!")[0].strip())
            iw = _int_width(lines[i + 1]) or 9
            i += 2
            got = 0
            while i < n and got < ntypes:
                if not _is_data_line(lines[i]):
                    break
                v = _slice_ints(lines[i], iw)
                if v and v[0] == -1:
                    i += 1
                    break
                if len(v) >= 2:
                    etype_lib[v[0]] = v[1]
                    got += 1
                i += 1
        elif up.startswith("NBLOCK"):
            saw_block = True
            iw = _int_width(lines[i + 1]) or 9
            rw = _real_width(lines[i + 1]) or 20
            i += 2
            while i < n:
                l = lines[i]
                s = l.strip().upper()
                if s.startswith("N,") or s.startswith("-1") or s == "":
                    i += 1
                    break
                if not _is_data_line(l):
                    break
                try:
                    nid = int(l[0:iw])
                except ValueError:
                    i += 1
                    continue
                if nid < 0:
                    i += 1
                    break

                rs = (_slice_reals(l[3 * iw:], rw) + [0.0, 0.0, 0.0])[:3]
                node_id.append(nid)
                coords.append(rs)
                i += 1
        elif up.startswith("EBLOCK"):
            saw_block = True
            iw = _int_width(lines[i + 1]) or 9
            i += 2
            while i < n:
                l = lines[i]
                if l.strip().startswith("-1"):
                    i += 1
                    break
                if not _is_data_line(l):
                    break

                fields = _slice_ints(l, iw)
                if not fields:
                    i += 1
                    continue
                etype_local, nnodes, elem_id = fields[1], fields[8], fields[10]
                nodes = fields[11:]
                i += 1
                while len(nodes) < nnodes and i < n:
                    next_1 = lines[i]
                    if not _is_data_line(next_1):
                        break
                    if next_1.strip().startswith("-1"):
                        break
                    nodes += _slice_ints(lines[i], iw)
                    i += 1
                elements.append((etype_local, elem_id, nodes[:nnodes]))
        elif up.startswith("CMBLOCK"):
            saw_block = True
            p = line.split(",")
            cname = p[1].strip()
            entity = p[2].strip().upper()
            numitems = int(p[3].split("!")[0].strip())
            iw = _int_width(lines[i + 1]) or 10
            i += 2
            items = []
            while i < n and len(items) < numitems:
                if not _is_data_line(lines[i]):
                    break
                items += _slice_ints(lines[i], iw)
                i += 1
            items = items[:numitems]
            expanded, prev = [], None
            for it in items:
                if it < 0:
                    if prev is None:
                        raise ReadError(
                            f"Invalid CMBLOCK '{cname}': range marker "
                            "(negative value) before any base value."
                        )
                    expanded += list(range(prev + 1, -it + 1))
                    prev = -it
                else:
                    expanded.append(it)
                    prev = it
            dest = node_comps if entity.startswith("NODE") else elem_comps
            dest[cname] = expanded
        else:
            i += 1

    if not saw_block:
        raise ReadError("No MAPDL block (NBLOCK/EBLOCK/CMBLOCK) found.")
    return _build_mesh(etype_lib, node_id, coords, elements, node_comps, elem_comps)


def _meshio_type(etype_lib, etype_local, nnodes):
    family = _FAMILY.get(etype_lib.get(etype_local), "solid")
    key = (family, nnodes)
    if key not in _TO_MESHIO:
        raise ReadError(f"Unsupported type: etype {etype_local} with {nnodes} nodes.")
    return _TO_MESHIO[key]


def _build_mesh(etype_lib, node_id, coords, elements, node_comps, elem_comps):
    points = np.array(coords, dtype=float)
    nid_to_index = {nid: k for k, nid in enumerate(node_id)}
    blocks, eid_to_loc = {}, {}
    for etype_local, elem_id, nodes in elements:
        mtype = _meshio_type(etype_lib, etype_local, len(nodes))
        blocks.setdefault(mtype, [])
        eid_to_loc[elem_id] = (mtype, len(blocks[mtype]))
        blocks[mtype].append([nid_to_index[x] for x in nodes])
    cells = [(t, np.array(c, dtype=int)) for t, c in blocks.items()]
    order = [t for t, _ in cells]

    point_sets = {
        name: np.array([nid_to_index[x] for x in ids if x in nid_to_index], dtype=int)
        for name, ids in node_comps.items()
    }
    cell_sets = {}
    for name, ids in elem_comps.items():
        per = [[] for _ in order]
        for eid in ids:
            if eid in eid_to_loc:
                t, loc = eid_to_loc[eid]
                per[order.index(t)].append(loc)
        cell_sets[name] = [np.array(p, dtype=int) for p in per]
    return Mesh(points, cells, point_sets=point_sets, cell_sets=cell_sets)


# Write: ET/ETBLOCK and NBLOCK blocks are written first, then EBLOCK, then CMBLOCK.
def write(filename, mesh):
    pts = mesh.points
    if pts.shape[1] == 2:
        pts = np.column_stack([pts, np.zeros(len(pts))])

    type_slot = {}
    for b in mesh.cells:
        if b.type not in _FROM_MESHIO:
            raise WriteError(f"Unhandled meshlane type: {b.type}")
        type_slot.setdefault(b.type, len(type_slot) + 1)

    with open_file(filename, "w") as f:
        f.write("/PREP7\n")
        for t, slot in type_slot.items():
            f.write(f"ET,{slot},{_FROM_MESHIO[t]}\n")
        nn = len(pts)
        f.write(f"NBLOCK,6,SOLID,{nn},{nn}\n(3i9,6e20.13)\n")
        for k, (x, y, z) in enumerate(pts):
            f.write(f"{k+1:9d}{0:9d}{0:9d}" + "% .13E% .13E% .13E" % (x, y, z) + "\n")
        f.write("N,R5.3,LOC,      -1,\n")

        ntot = sum(len(b.data) for b in mesh.cells)
        f.write(f"EBLOCK,19,SOLID,{ntot},{ntot}\n(19i9)\n")
        eid = 0
        loc_to_eid = {}
        for bi, b in enumerate(mesh.cells):
            slot = type_slot[b.type]
            for li, conn in enumerate(b.data):
                eid += 1
                loc_to_eid[(bi, li)] = eid
                nodes = [int(x) + 1 for x in conn]
                first = [1, slot, 1, 1, 0, 0, 0, 0, len(nodes), 0, eid] + nodes[:8]
                f.write("".join(f"{v:9d}" for v in first) + "\n")
                if len(nodes) > 8:
                    f.write("".join(f"{v:9d}" for v in nodes[8:]) + "\n")
        f.write(f"{-1:9d}\n")

        for name, ids in mesh.point_sets.items():
            vals = [int(x) + 1 for x in ids]
            f.write(f"CMBLOCK,{name},NODE,{len(vals):9d}\n(8i10)\n")
            _write_items(f, vals)
        for name, blocks in mesh.cell_sets.items():
            vals = sorted(loc_to_eid[(bi, int(li))]
                          for bi, arr in enumerate(blocks)
                          for li in np.asarray(arr).tolist())
            f.write(f"CMBLOCK,{name},ELEM,{len(vals):9d}\n(8i10)\n")
            _write_items(f, vals)
        f.write("FINISH\n")


def _write_items(f, vals):
    for i in range(0, len(vals), 8):
        f.write("".join(f"{v:10d}" for v in vals[i:i + 8]) + "\n")


register_format("ansysInp", [".cdb", ".inp"], read, {"ansysInp": write})

"""CGNS element types (ElementType_t) and the decoding of element sections.

Codes are those of CGNS 3.1 and later (cgnslib.h). Files written by older versions
are renumbered on the fly, as the CGNS library does (cgns_internals.c).
"""

import numpy as np

from .._exceptions import ReadError

NODE = 2
MIXED = 20
PYRA_13 = 21
NGON_n = 22
NFACE_n = 23

# code -> (meshlane cell type, nodes per element)
TYPES = {
    2: ("vertex", 1),
    3: ("line", 2),
    4: ("line3", 3),
    5: ("triangle", 3),
    6: ("triangle6", 6),
    7: ("quad", 4),
    8: ("quad8", 8),
    9: ("quad9", 9),
    10: ("tetra", 4),
    11: ("tetra10", 10),
    12: ("pyramid", 5),
    14: ("wedge", 6),
    15: ("wedge15", 15),
    17: ("hexahedron", 8),
    18: ("hexahedron20", 20),
    19: ("hexahedron27", 27),
    21: ("pyramid13", 13),
}

NAMES = {
    2: "NODE", 3: "BAR_2", 4: "BAR_3", 5: "TRI_3", 6: "TRI_6", 7: "QUAD_4",
    8: "QUAD_8", 9: "QUAD_9", 10: "TETRA_4", 11: "TETRA_10", 12: "PYRA_5",
    13: "PYRA_14", 14: "PENTA_6", 15: "PENTA_15", 16: "PENTA_18", 17: "HEXA_8",
    18: "HEXA_20", 19: "HEXA_27", 20: "MIXED", 21: "PYRA_13", 22: "NGON_n",
    23: "NFACE_n",
}  # fmt: skip

# Node order: CGNS -> meshlane (VTK). The corners and the edge midpoints of the
# first faces agree; CGNS lists the vertical edges of hexahedra and wedges before
# the top ones, VTK after, and the face centers of HEXA_27 differ.
_HEXA_20 = list(range(12)) + [16, 17, 18, 19, 12, 13, 14, 15]
PERMUTATIONS = {
    "wedge15": list(range(9)) + [12, 13, 14, 9, 10, 11],
    "hexahedron20": _HEXA_20,
    "hexahedron27": _HEXA_20 + [24, 22, 21, 23, 20, 25, 26],
}


def type_name(code):
    return NAMES.get(int(code), str(int(code)))


def renumber(code, version):
    """Element type code in the current numbering, for a file of this version."""
    code = int(code)
    if 3000 <= version < 3100 and 12 < code < NGON_n:
        # 3.0 numbering: PYRA_14 was 13, the following types were shifted by one
        return PYRA_13 if code == 13 else code - 1
    return code


def _renumber_mixed(code, version):
    code = int(code)
    if version < 3000:
        # before 3.0, NGON_n was 21 and polygons in MIXED were NGON_n + n
        return code + 1 if code > MIXED else code
    return renumber(code, version)


def nodes_per_element(code):
    if code in TYPES:
        return TYPES[code][1]
    if code > NGON_n:
        # polygon with code - NGON_n nodes, possible in old MIXED sections
        return code - NGON_n
    raise ReadError(f"Unsupported CGNS element type {type_name(code)}.")


def ragged(connectivity, num_elements, offsets):
    """CSR (ptr, values) of an NGON_n or NFACE_n section: CGNS 4 stores the
    offsets in ElementStartOffset, older files put each element's length in
    front of its values."""
    connectivity = np.asarray(connectivity, dtype=np.int64)
    if offsets is not None:
        return np.asarray(offsets, dtype=np.int64), connectivity
    values = connectivity.tolist()
    starts = np.empty(num_elements, dtype=np.int64)
    sizes = np.empty(num_elements, dtype=np.int64)
    i = 0
    for k in range(num_elements):
        sizes[k] = values[i]
        starts[k] = i + 1
        i += values[i] + 1
    if i != len(values):
        raise ReadError("NGON_n/NFACE_n connectivity has the wrong size.")
    ptr = np.concatenate([[0], np.cumsum(sizes)])
    index = np.repeat(starts - ptr[:-1], sizes) + np.arange(ptr[-1])
    return ptr, connectivity[index]


def split_mixed(connectivity, num_elements, offsets, version):
    """Split a MIXED section by element type.

    Returns ``{code: (local element indices, nodes (n, k))}``, where code is the
    element type in the current numbering, or NGON_n + k for polygons."""
    connectivity = np.asarray(connectivity, dtype=np.int64)
    if offsets is not None:
        offsets = np.asarray(offsets, dtype=np.int64)
        starts = offsets[:-1]
        codes = connectivity[starts]
        if version < 3100:
            codes = np.array([_renumber_mixed(c, version) for c in codes.tolist()])
    else:
        values = connectivity.tolist()
        starts = np.empty(num_elements, dtype=np.int64)
        codes = np.empty(num_elements, dtype=np.int64)
        i = 0
        for k in range(num_elements):
            code = _renumber_mixed(values[i], version)
            starts[k] = i
            codes[k] = code
            i += nodes_per_element(code) + 1
        if i != len(values):
            raise ReadError("MIXED connectivity has the wrong size.")
    out = {}
    for code in np.unique(codes).tolist():
        local = np.flatnonzero(codes == code)
        k = nodes_per_element(code)
        nodes = connectivity[starts[local][:, None] + 1 + np.arange(k)]
        out[code] = (local, nodes)
    return out


def to_meshlane(code, nodes):
    """meshlane (cell type, nodes) of elements of a fixed CGNS type; nodes are
    still 1-based."""
    if code > NGON_n:
        n = code - NGON_n
        return {3: "triangle", 4: "quad"}.get(n, f"polygon{n}"), nodes
    if code not in TYPES:
        raise ReadError(f"Unsupported CGNS element type {type_name(code)}.")
    cell_type = TYPES[code][0]
    if cell_type in PERMUTATIONS:
        nodes = nodes[:, PERMUTATIONS[cell_type]]
    return cell_type, nodes

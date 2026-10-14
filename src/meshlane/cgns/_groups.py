"""Groups of a CGNS zone: boundary conditions (ZoneBC_t/BC_t) and families.

A BC lists either elements (ElementList/ElementRange, or PointList/PointRange
with a GridLocation such as FaceCenter or CellCenter) or nodes (GridLocation
Vertex, the default and the usual choice of older writers). Element groups become
cell sets; node groups become point sets, and the boundary faces whose nodes are
all in the group also become a cell set.

Element sections and zones can also belong to families (FamilyName_t and
AdditionalFamilyName_t), which become cell sets named after the family.
"""

import numpy as np

from ._tree import child, children, data, name, resolve, text


def _string(node):
    return text(node).strip("\x00 ")


def location(node, version):
    """GridLocation of a BC or FlowSolution node ("Vertex" by default)."""
    loc = child(node, "GridLocation_t")
    if loc is None and version <= 1200:
        # until CGNS 1.2, the location of a BC was under its BCDataSet
        for dataset in children(node, "BCDataSet_t"):
            loc = child(dataset, "GridLocation_t")
            if loc is not None:
                break
    return _string(loc) if loc is not None else "Vertex"


def family_names(node, version):
    """Families of a node (FamilyName_t, then AdditionalFamilyName_t)."""
    names = []
    family = child(node, "FamilyName_t")
    if family is not None:
        # until CGNS 1.2, the family name was the name of the FamilyName_t node
        names.append(name(family) if version <= 1200 else _string(family))
    names += [_string(extra) for extra in children(node, "AdditionalFamilyName_t")]
    return [n for n in names if n]


def _point_set(node):
    """(kind of list, 1-based values) of a BC, or (None, None)."""
    for key, is_range in (
        ("PointList", False),
        ("PointRange", True),
        ("ElementList", False),
        ("ElementRange", True),
    ):
        if key in node:
            values = np.ravel(data(resolve(node[key]))).astype(np.int64)
            if is_range and len(values):
                values = np.arange(values.min(), values.max() + 1)
            return key, values
    return None, None


def boundary_conditions(zone, version):
    """``[(name, kind, values)]`` for the BCs of a zone, kind being "elements"
    (1-based element numbers) or "points" (1-based node numbers)."""
    zone_bc = child(zone, "ZoneBC_t")
    if zone_bc is None:
        return []
    out = []
    for bc in children(zone_bc, "BC_t"):
        key, values = _point_set(bc)
        if values is None:
            continue
        is_elements = key.startswith("Element") or location(bc, version) != "Vertex"
        if is_elements and len(values) and values.min() == 0:
            # element numbers start at 1: some CGNS 1.x writers used 0-based lists
            values = values + 1
        group = name(bc) or next(iter(family_names(bc, version)), "")
        if group:
            out.append((group, "elements" if is_elements else "points", values))
    return out

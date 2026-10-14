"""Fields of a CGNS zone (FlowSolution_t), as point or cell data."""

from collections import Counter

import numpy as np

from .._common import warn
from ._groups import location
from ._structured import _strip_rind
from ._tree import child, children, data, name, resolve


def _first_pointer(zone):
    """Name of the FlowSolution of the first time step, when the zone lists one
    per step (ZoneIterativeData/FlowSolutionPointers)."""
    iterative = child(zone, "ZoneIterativeData_t")
    if iterative is None or "FlowSolutionPointers" not in iterative:
        return None
    pointers = np.asarray(data(resolve(iterative["FlowSolutionPointers"])))
    if pointers.ndim != 2 or len(pointers) == 0:
        return None
    return bytes(pointers[0].astype(np.int8)).decode(errors="replace").strip("\x00 ")


def zone_fields(zone, version):
    """``[(field name, location, values)]`` of the numeric arrays of the
    FlowSolution nodes of a zone. A field name found in several solutions gets the
    solution name as suffix."""
    solutions = children(zone, "FlowSolution_t")
    first = _first_pointer(zone)
    if first is not None and any(name(s) == first for s in solutions):
        if len(solutions) > 1:
            warn(f"Zone {name(zone)}: one solution per time step, the first is read.")
        solutions = [s for s in solutions if name(s) == first]
    entries = []
    for solution in solutions:
        loc = location(solution, version)
        rind = solution.get("Rind")
        rind = None if rind is None else np.ravel(data(resolve(rind))).astype(int)
        for array in children(solution, "DataArray_t"):
            values = data(array)
            if values is None or values.dtype.kind not in "fiu":
                continue
            values = np.asarray(values, dtype=float)
            if rind is not None and values.ndim > 1 and len(rind) == 2 * values.ndim:
                values = _strip_rind(values, rind)  # structured: per direction
            values = values.ravel()
            if rind is not None and len(rind) == 2 and values.ndim == 1:
                values = values[rind[0] : len(values) - rind[1]]
            entries.append((name(solution), name(array), loc, values))
    count = Counter(field for _, field, _, _ in entries)
    return [
        (field if count[field] == 1 else f"{field}_{sol}", loc, values)
        for sol, field, loc, values in entries
    ]

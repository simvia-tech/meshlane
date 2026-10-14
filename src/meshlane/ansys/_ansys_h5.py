"""
I/O for Ansys Fluent's HDF5 mesh format (.msh.h5), written by Fluent 2020 R1 and
later. It holds the same data as the .msh format (node zones, faces with their
nodes and two cells, named zones) in HDF5 datasets:

    /meshes/1/nodes/coords/<section>          coordinates, minId/maxId attributes
    /meshes/1/faces/nodes/<section>/nnodes    number of nodes of each face
    /meshes/1/faces/nodes/<section>/nodes     face nodes, concatenated
    /meshes/1/faces/c0/<section>, c1/<section>  cells on either side (0: none)
    /meshes/1/{cells,faces}/zoneTopology      zone id, minId, maxId, name, ...

so the cells are rebuilt from the faces as for .msh files.
"""

import numpy as np

from .._common import warn
from .._exceptions import ReadError
from .._helpers import register_format
from ._ansys import _FluentMesh


def _sections(group):
    """The sections of a group, sorted by their first index."""
    return sorted(group.values(), key=lambda s: int(s.attrs["minId"][0]))


def _zone_topology(group):
    """Zone rows (id, minId, maxId, name, zoneType or 0) of a zoneTopology
    group. Names are stored as one ';'-separated string."""
    ids = group["id"][()]
    names = group["name"][0].decode().split(";") if "name" in group else []
    zone_type = group["zoneType"][()] if "zoneType" in group else None
    return [
        (
            int(ids[i]),
            int(group["minId"][i]),
            int(group["maxId"][i]),
            names[i] if i < len(names) else f"zone{ids[i]}",
            0 if zone_type is None else int(zone_type[i]),
        )
        for i in range(len(ids))
    ]


def read(filename):
    import h5py

    try:
        f = h5py.File(filename, "r")
    except OSError as e:
        raise ReadError(f"Not an HDF5 file: {e}") from e
    with f:
        if "meshes/1" not in f:
            raise ReadError("Not a Fluent HDF5 mesh file (no /meshes/1).")
        if len(f["meshes"]) > 1:
            warn(f"{filename}: {len(f['meshes'])} meshes, only the first is read.")
        m = f["meshes/1"]
        out = _FluentMesh()
        out.dim = int(m.attrs["dimension"][0])
        out.num_cells = int(m.attrs["cellCount"][0])

        for s in _sections(m["nodes/coords"]):
            out.nodes.append((int(s.attrs["minId"][0]), s[()].astype(float)))

        if "cells/zoneTopology" in m:
            for zone, first, last, name, _ in _zone_topology(m["cells/zoneTopology"]):
                out.cell_zones.append((zone, first, last))
                out.zones[zone] = ("", name)

        # all faces, by global index
        nnodes, nodes, c0, c1 = [], [], [], []
        for s in _sections(m["faces/nodes"]):
            nnodes.append(s["nnodes"][()].astype(np.int64))
            nodes.append(s["nodes"][()].astype(np.int64))
        for cells, key in ((c0, "faces/c0"), (c1, "faces/c1")):
            for s in _sections(m[key]):
                cells.append(s[()].astype(np.int64))
        nnodes = np.concatenate(nnodes) if nnodes else np.zeros(0, dtype=np.int64)
        nodes = np.concatenate(nodes) if nodes else np.zeros(0, dtype=np.int64)
        c0 = np.concatenate(c0) if c0 else np.zeros(0, dtype=np.int64)
        c1 = np.concatenate(c1) if c1 else np.zeros(0, dtype=np.int64)
        if not len(nnodes) == len(c0) == len(c1):
            raise ReadError("Face sections do not match.")
        ptr = np.concatenate([[0], np.cumsum(nnodes)])
        offset = int(m.attrs["faceOffset"][0]) + 1 if "faceOffset" in m.attrs else 1

        for zone, first, last, name, bc in _zone_topology(m["faces/zoneTopology"]):
            a, b = first - offset, last - offset + 1
            out.face_zones.append(
                (
                    zone,
                    bc,
                    ptr[a : b + 1] - ptr[a],
                    nodes[ptr[a] : ptr[b]],
                    c0[a:b],
                    c1[a:b],
                )
            )
            out.zones[zone] = ("", name)
        # the zones hold views of these arrays; drop the other references so
        # that mesh() can free them once it has gathered the faces
        del nnodes, nodes, c0, c1, ptr

    try:
        return out.mesh()
    except (ValueError, IndexError, KeyError) as e:
        raise ReadError(f"Invalid Fluent mesh file: {e}") from e


register_format("ansys-h5", [".msh.h5"], read, {})

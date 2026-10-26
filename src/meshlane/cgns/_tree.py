"""Read access to a CGNS tree stored in HDF5 (the CGNS SIDS-to-HDF5 mapping).

Each CGNS node is an HDF5 group with the attributes ``name``, ``label`` (its SIDS
type, e.g. ``Zone_t``) and ``type`` (data type: ``MT``, ``I4``, ``I8``, ``R4``,
``R8``, ``C1``, ...). Its value, if any, is the dataset ``" data"``. Node names are
free, so nodes are found by label. Arrays are stored in Fortran order: their HDF5
shape is the reverse of the SIDS one.
"""

import os

import numpy as np


def label(node):
    value = node.attrs.get("label", b"")
    return value.decode() if isinstance(value, bytes) else str(value)


def name(node):
    value = node.attrs.get("name", b"")
    value = value.decode() if isinstance(value, bytes) else str(value)
    return value or node.name.rsplit("/", 1)[-1]


def data(node):
    """Value of a node as a numpy array, or None."""
    if " data" not in node:
        return None
    return node[" data"][()]


def text(node):
    """Value of a character (C1) node."""
    value = data(node)
    if value is None:
        return ""
    return bytes(np.asarray(value, dtype=np.int8).ravel()).decode(errors="replace")


def resolve(node):
    """Target of a CGNS link node (data type ``LK``), or the node itself.

    A link node holds the target file and path, and an HDF5 link ``" link"`` to
    it; when HDF5 cannot follow that one, the file is looked up next to the file
    holding the link."""
    import h5py

    if node.attrs.get("type", b"") not in (b"LK", "LK"):
        return node
    try:
        return node[" link"]
    except (KeyError, OSError):
        pass
    path = text(node[" path"]).strip("\x00 ") if " path" in node else ""
    target = text(node[" file"]).strip("\x00 ") if " file" in node else ""
    if not path:
        return node
    if not target:
        return node.file[path]
    target = os.path.join(os.path.dirname(node.file.filename), target)
    return h5py.File(target, "r")[path]


def children(node, node_label):
    """Child nodes with the given label (links resolved), in the file's order."""
    import h5py

    out = []
    for value in node.values():
        if isinstance(value, h5py.Group):
            value = resolve(value)
            if isinstance(value, h5py.Group) and label(value) == node_label:
                out.append(value)
    return out


def child(node, node_label):
    """First child node with the given label, or None."""
    found = children(node, node_label)
    return found[0] if found else None


def file_version(f):
    """CGNS version that wrote the file, as an integer (3.1 -> 3100)."""
    node = f.get("CGNSLibraryVersion")
    if node is None or data(node) is None:
        return 4000
    return int(round(float(np.ravel(data(node))[0]) * 1000))

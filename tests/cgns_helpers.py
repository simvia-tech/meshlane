"""Write small CGNS (HDF5) files by hand, to test the reader on layouts that the
meshlane writer does not produce (old versions, MIXED, NGON_n/NFACE_n, links...).

The node layout follows files written by the CGNS library: attributes ``name``,
``label`` (33-byte strings), ``type`` (3-byte string) and ``flags``, value in the
dataset ``" data"``, children kept in creation order.
"""

import numpy as np


def new_file(path, version=4.4):
    import h5py

    f = h5py.File(path, "w")
    f.attrs.create("name", np.bytes_("HDF5 MotherNode"), dtype="S33")
    f.attrs.create("label", np.bytes_("Root Node of HDF5 File"), dtype="S33")
    f.attrs.create("type", np.bytes_("MT"), dtype="S3")
    f.create_dataset(" format", data=np.frombuffer(b"IEEE_LITTLE_32\0", dtype=np.int8))
    hdf5 = b"HDF5 Version 1.14.0".ljust(33, b"\0")
    f.create_dataset(" hdf5version", data=np.frombuffer(hdf5, dtype=np.int8))
    node(f, "CGNSLibraryVersion", "CGNSLibraryVersion_t", np.float32([version]))
    return f


def node(parent, name, label, value=None, dtype=None):
    """A CGNS node; value is an array, a string (C1) or None (MT)."""
    group = parent.create_group(name, track_order=True)
    if value is None:
        kind = "MT"
    elif isinstance(value, str):
        value = np.frombuffer(value.encode(), dtype=np.int8)
        kind = "C1"
    else:
        value = np.asarray(value, dtype=dtype)
        kind = {"int32": "I4", "int64": "I8", "float32": "R4", "float64": "R8"}[
            value.dtype.name
        ]
    group.attrs.create("name", np.bytes_(name), dtype="S33")
    group.attrs.create("label", np.bytes_(label), dtype="S33")
    group.attrs.create("type", np.bytes_(kind), dtype="S3")
    group.attrs.create("flags", np.int32([1]))
    if value is not None:
        # CGNS arrays are Fortran ordered: the HDF5 shape is reversed
        group.create_dataset(" data", data=value.T if value.ndim > 1 else value)
    return group


def base(f, name="Base", cell_dim=3, phys_dim=3):
    return node(f, name, "CGNSBase_t", np.int32([cell_dim, phys_dim]))


def unstructured_zone(parent, points, num_cells, name="Zone"):
    zone = node(parent, name, "Zone_t", np.int64([[len(points), num_cells, 0]]))
    node(zone, "ZoneType", "ZoneType_t", "Unstructured")
    grid = node(zone, "GridCoordinates", "GridCoordinates_t")
    points = np.asarray(points, dtype=float)
    for k, axis in enumerate("XYZ"[: points.shape[1]]):
        node(grid, f"Coordinate{axis}", "DataArray_t", points[:, k])
    return zone


def section(zone, name, code, first, connectivity, num, offsets=None, dtype=np.int64):
    """An Elements_t node of num elements numbered from first."""
    elements = node(zone, name, "Elements_t", np.int32([code, 0]))
    node(elements, "ElementRange", "IndexRange_t", np.int64([first, first + num - 1]))
    if offsets is not None:
        node(elements, "ElementStartOffset", "DataArray_t", np.asarray(offsets, dtype))
    node(
        elements,
        "ElementConnectivity",
        "DataArray_t",
        np.asarray(connectivity, dtype=dtype).ravel(),
    )
    return elements

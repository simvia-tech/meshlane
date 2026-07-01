"""
I/O for multi-mesh MED/Salome files.

This module builds on ._med (single-mesh implementation) and adds support for
files containing several meshes under ENS_MAA, with fields that may belong to
different meshes (disambiguated with an ``@<mesh_name>`` suffix when a field
name collides across meshes).

It preserves, on a read -> write round-trip:
  * coordinate / time units (UNI, UNT),
  * field component names (NOM),
  * field time-step metadata (NDT, NOR, PDT),
  * families and their groups (GRO/NOM as H5T_ARRAY[80] of char).

Two things are essential for medfile / Salome / mdump to read the result:
  * h5py must track HDF5 link creation order (medfile enumerates objects with
    H5_INDEX_CRT_ORDER) -> we set track_order=True before creating any group;
  * GRO/NOM must be a H5T_ARRAY{[80] H5T_NATIVE_CHAR} dataset, which is handled
    inside ._med._write_families.
"""

import numpy as np
from collections import Counter, defaultdict

from .._common import num_nodes_per_cell
from .._exceptions import WriteError
from .._mesh import Mesh
from ._med41 import FieldBitmaskWriter
from ._med import (
    meshio_to_med_type,
    med_to_geo_type,
    med_to_meshio_type,
    med_type_to_entity,
    numpy_to_med_type,
    numpy_void_str,
    MED_FLOAT64,
    _med_cells_for_write,
    _reorder_med_cells,
    _warn_unconverted_3d,
    _write_families,
    _read_families,
    _read_data,
    _parse_med_field_name,
    _write_field_step,
)



def _resolve_mesh_names(meshes, mesh_names=None):
    """Return a list of unique mesh names, one per mesh.

    Missing names are filled with mesh_<i>; duplicates are de-duplicated by
    appending a counter suffix.
    """
    if mesh_names is None:
        mesh_names = [f"mesh_{i}" for i in range(len(meshes))]

    if len(mesh_names) > len(meshes):
        raise WriteError(
            f"More mesh names ({len(mesh_names)}) than meshes ({len(meshes)})."
        )

    # de-duplicate
    seen = {}
    resolved = []
    for name in mesh_names:
        if name in seen:
            seen[name] += 1
            resolved.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 0
            resolved.append(name)
    mesh_names = resolved

    # pad the rest with defaults
    if len(mesh_names) < len(meshes):
        mesh_names = mesh_names + [
            f"mesh_{i}" for i in range(len(mesh_names), len(meshes))
        ]
    return mesh_names


def _find_field_collisions(meshes):
    """Base field names that appear in more than one *distinct* mesh.

    A base name is counted at most once per mesh, so the many time-steps of a
    single field on a single mesh (Boundary temperature[0] - 0.1, [1] - 0.2 ...)
    do NOT look like a collision and the field keeps its plain name.
    """
    counts = Counter()
    for mesh in meshes:
        bases = set()
        for name in mesh.point_data.keys():
            if name != "point_tags":
                base, _, _ = _parse_med_field_name(name)
                bases.add(base)
        for name in mesh.cell_data.keys():
            if name not in {"cell_tags", "gmsh:physical"}:
                base, _, _ = _parse_med_field_name(name)
                bases.add(base)
        for base in bases:
            counts[base] += 1
    return {name for name, count in counts.items() if count > 1}


def _bytes_attr(value, fallback=numpy_void_str):
    """Encode a python str / bytes into a MED-friendly fixed string attr.

    MED stores strings as 8-bit char arrays, so non-ASCII text is Latin-1 encoded.
    """
    if value is None or value == "":
        return fallback
    if isinstance(value, (bytes, np.bytes_)):
        return np.bytes_(value)
    return np.bytes_(str(value).encode("latin-1"))


def _create_field_group(fields, hdf5_name, mesh_name, first_data,
                        n_components, units, comp_name):
    """Create (or fetch) the CHA/<field> group with its MED attributes."""
    if hdf5_name in fields:
        return fields[hdf5_name]

    field = fields.create_group(hdf5_name)
    field.attrs.create("MAI", np.bytes_(mesh_name))
    field.attrs.create("TYP", numpy_to_med_type.get(first_data.dtype, MED_FLOAT64))
    field.attrs.create("NCO", n_components)
    field.attrs.create("UNI", units[0] if units[0] is not None else numpy_void_str)
    field.attrs.create("UNT", units[1] if units[1] is not None else numpy_void_str)
    if comp_name:
        nom = np.bytes_("".join(f"{n:<16}" for n in comp_name))
    else:
        nom = np.bytes_(f"{'':<16}")
    field.attrs.create("NOM", nom)
    return field


def _write_mesh_fields(fields, mesh, mesh_name, collisions):
    """Write every (nodal + cell) field of one mesh into the shared CHA group,
    preserving NDT/NOR/PDT, units and component names."""
    field_comp_names = mesh.field_data.get("med:nom", [])
    step_meta = mesh.field_data.get("med:step_meta", {})
    field_units = mesh.field_data.get("med:field_units", {})
    name_idx = 0

    #  Nodal fields, grouped by base name (multi-timestep)
    nodal_groups = defaultdict(list)
    for name, data in mesh.point_data.items():
        if name == "point_tags":
            continue
        base, idx, pdt = _parse_med_field_name(name)
        nodal_groups[base].append((idx, pdt, data))

    for base_name, entries in nodal_groups.items():
        entries.sort(key=lambda x: x[0] if x[0] is not None else 0)
        comp_name = (
            field_comp_names[name_idx] if name_idx < len(field_comp_names) else None
        )
        name_idx += 1

        first_data = entries[0][2]
        n_components = 1 if first_data.ndim == 1 else first_data.shape[-1]
        units = field_units.get(base_name, (numpy_void_str, numpy_void_str))
        hdf5_name = f"{base_name}@{mesh_name}" if base_name in collisions else base_name

        field = _create_field_group(
            fields, hdf5_name, mesh_name, first_data, n_components, units, comp_name
        )

        tracker = FieldBitmaskWriter()
        meta_list = step_meta.get(base_name, [])
        for i, (idx, pdt_orig, data) in enumerate(entries):
            meta = meta_list[i] if i < len(meta_list) else {}
            ndt = meta.get("ndt", i + 1)
            nor = meta.get("nor", -1)
            pdt = meta.get("pdt", pdt_orig if pdt_orig is not None else 0.0)
            step_name = f"{ndt:020d}{nor:020d}"
            _write_field_step(field, step_name, ndt, nor, pdt, "NOEU", data)
            tracker.notify("MED_NODE", "MED_NO_GEOTYPE", step_name)
        tracker.flush(field)

    #  Cell fields, grouped by base name (multi-timestep)
    cell_groups = defaultdict(list)
    for name, d in mesh.cell_data.items():
        if name in ("cell_tags", "gmsh:physical"):
            continue
        base, idx, pdt_orig = _parse_med_field_name(name)
        for cell, data in zip(mesh.cells, d):
            if data is None:
                continue
            cell_groups[base].append((idx, pdt_orig, cell.type, data))

    for base_name, entries in cell_groups.items():
        entries.sort(key=lambda x: x[0] if x[0] is not None else 0)
        comp_name = (
            field_comp_names[name_idx] if name_idx < len(field_comp_names) else None
        )
        name_idx += 1

        first_data = entries[0][3]
        n_components = 1 if first_data.ndim == 1 else first_data.shape[-1]
        units = field_units.get(base_name, (numpy_void_str, numpy_void_str))
        hdf5_name = f"{base_name}@{mesh_name}" if base_name in collisions else base_name

        field = _create_field_group(
            fields, hdf5_name, mesh_name, first_data, n_components, units, comp_name
        )

        tracker = FieldBitmaskWriter()
        meta_list = step_meta.get(base_name, [])
        for i, (idx, pdt_orig, cell_type, data) in enumerate(entries):
            if data.dtype == object:
                continue
            med_type = meshio_to_med_type[cell_type]

            if data.ndim > 2:
                if data.shape[1] == num_nodes_per_cell[cell_type]:
                    supp = "ELNO"
                else:
                    continue  # skip ELGA
            else:
                supp = "ELEM"

            meta = meta_list[i] if i < len(meta_list) else {}
            ndt = meta.get("ndt", i + 1)
            nor = meta.get("nor", -1)
            pdt = meta.get("pdt", pdt_orig if pdt_orig is not None else 0.0)
            step_name = f"{ndt:020d}{nor:020d}"

            _write_field_step(
                field, step_name, ndt, nor, pdt, supp, data, med_type=med_type
            )
            if med_type in med_to_geo_type and med_type in med_type_to_entity:
                tracker.notify(
                    med_type_to_entity[med_type], med_to_geo_type[med_type], step_name
                )
        tracker.flush(field)


def _write_med_multi(filename, meshes, mesh_names=None, med_version="4.1.0", **kwargs):
    import h5py

    if meshes is None or len(meshes) == 0:
        raise WriteError("No mesh to write.")
    if not isinstance(meshes, list):
        raise WriteError("Meshes must be provided as a list.")

    try:
        version_parts = [int(x) for x in med_version.split(".")]
        maj = version_parts[0]
        minor = version_parts[1] if len(version_parts) > 1 else 0
        rel = version_parts[2] if len(version_parts) > 2 else 0
    except ValueError:
        maj, minor, rel = 4, 1, 0

    mesh_names = _resolve_mesh_names(meshes, mesh_names)
    collisions = _find_field_collisions(meshes)

    f = h5py.File(filename, "w")

    info = f.create_group("INFOS_GENERALES")
    info.attrs.create("MAJ", maj)
    info.attrs.create("MIN", minor)
    info.attrs.create("REL", rel)

    # Meshes
    mesh_ensemble = f.create_group("ENS_MAA")
    for mesh, name in zip(meshes, mesh_names):
        med_mesh = mesh_ensemble.create_group(name)
        med_mesh.attrs.create("DIM", mesh.points.shape[1])
        med_mesh.attrs.create("ESP", mesh.points.shape[1])
        med_mesh.attrs.create("REP", 0)

        # preserve original metadata (DESCRIPTION kept verbatim)
        med_mesh.attrs.create("UNT", _bytes_attr(getattr(mesh, "unit_time", "")))
        med_mesh.attrs.create("UNI", _bytes_attr(getattr(mesh, "unit_coords", "")))
        med_mesh.attrs.create("SRT", 1)
        axis_names = ["X", "Y", "Z"][: mesh.points.shape[1]]
        med_mesh.attrs.create(
            "NOM", np.bytes_("".join(f"{n:<16}" for n in axis_names))
        )
        med_mesh.attrs.create(
            "DES",
            _bytes_attr(
                getattr(mesh, "description", ""),
                fallback=np.bytes_("Mesh created with meshlane"),
            ),
        )
        med_mesh.attrs.create("TYP", 0)

        step = "-0000000000000000001-0000000000000000001"  # NDT NOR
        time_step = med_mesh.create_group(step)
        time_step.attrs.create("CGT", 1)
        time_step.attrs.create("NDT", -1)
        time_step.attrs.create("NOR", -1)
        time_step.attrs.create("PDT", -1.0)

        # Points
        nodes_group = time_step.create_group("NOE")
        nodes_group.attrs.create("CGT", 1)
        nodes_group.attrs.create("CGS", 1)
        profile = "MED_NO_PROFILE_INTERNAL"
        nodes_group.attrs.create("PFL", np.bytes_(profile))
        coo = nodes_group.create_dataset("COO", data=mesh.points.flatten(order="F"))
        coo.attrs.create("CGT", 1)
        coo.attrs.create("NBR", len(mesh.points))

        if "point_tags" in mesh.point_data:
            fam = nodes_group.create_dataset(
                "FAM", data=mesh.point_data["point_tags"]
            )
            fam.attrs.create("CGT", 1)
            fam.attrs.create("NBR", len(mesh.points))

        # Cells (merge several blocks of the same type)
        cells_by_type = {}
        cell_tags_by_type = {}
        for k, cell_block in enumerate(mesh.cells):
            ct = cell_block.type
            cells_by_type.setdefault(ct, []).append(cell_block.data)
            if "cell_tags" in mesh.cell_data:
                cell_tags_by_type.setdefault(ct, []).append(
                    mesh.cell_data["cell_tags"][k]
                )

        cells_group = time_step.create_group("MAI")
        cells_group.attrs.create("CGT", 1)
        for cell_type, cells_list in cells_by_type.items():
            med_type = meshio_to_med_type[cell_type]
            med_cells = cells_group.create_group(med_type)
            med_cells.attrs.create("CGT", 1)
            med_cells.attrs.create("CGS", 1)
            med_cells.attrs.create("PFL", np.bytes_(profile))

            if cell_type in ("polygon", "polygon2"):
                all_polygons = sum(cells_list, [])
                all_nodes = np.concatenate([c + 1 for c in all_polygons])
                lengths = [len(c) for c in all_polygons]
                inn = np.concatenate([[1], np.cumsum(lengths) + 1])
                nod = med_cells.create_dataset("NOD", data=all_nodes)
                nod.attrs.create("CGT", 1)
                nod.attrs.create("NBR", len(all_polygons))
                inn_ds = med_cells.create_dataset("INN", data=inn)
                inn_ds.attrs.create("CGT", 1)
                n_merged = len(all_polygons)
            else:
                merged_cells = np.concatenate(cells_list, axis=0)
                merged_cells = _med_cells_for_write(cell_type, merged_cells)
                nod = med_cells.create_dataset(
                    "NOD", data=merged_cells.flatten(order="F") + 1
                )
                nod.attrs.create("CGT", 1)
                nod.attrs.create("NBR", len(merged_cells))
                n_merged = len(merged_cells)

            if cell_tags_by_type.get(cell_type):
                merged_tags = np.concatenate(cell_tags_by_type[cell_type])
                fam = med_cells.create_dataset("FAM", data=merged_tags)
                fam.attrs.create("CGT", 1)
                fam.attrs.create("NBR", n_merged)

    # Families (FAS)
    fas = f.create_group("FAS")
    for mesh, name in zip(meshes, mesh_names):
        families = fas.create_group(name)
        family_zero = families.create_group("FAMILLE_ZERO")
        family_zero.attrs.create("NUM", 0)

        try:
            if len(mesh.point_tags) > 0:
                node = families.create_group("NOEUD")
                _write_families(
                    node, mesh.point_tags, getattr(mesh, "point_tag_groups", {})
                )
        except AttributeError:
            pass

        try:
            if len(mesh.cell_tags) > 0:
                element = families.create_group("ELEME")
                _write_families(
                    element, mesh.cell_tags, getattr(mesh, "cell_tag_groups", {})
                )
        except AttributeError:
            pass

    # Fields (CHA)
    any_fields = any(
        any(k != "point_tags" for k in mesh.point_data)
        or any(k not in ("cell_tags", "gmsh:physical") for k in mesh.cell_data)
        for mesh in meshes
    )
    if any_fields:
        fields = f.create_group("CHA")
        for mesh, name in zip(meshes, mesh_names):
            _write_mesh_fields(fields, mesh, name, collisions)

    f.close()


def write_med_multi(filename, meshes, mesh_names=None, med_version="4.1.0", **kwargs):
    """Write several meshes to one MED file.

    Wraps the real writer so that HDF5 link-creation-order tracking is enabled
    for the whole file (required by medfile / Salome) and restored afterwards.
    """
    import h5py

    cfg = h5py.get_config()
    prev = cfg.track_order
    cfg.track_order = True
    try:
        _write_med_multi(
            filename, meshes, mesh_names=mesh_names, med_version=med_version, **kwargs
        )
    finally:
        cfg.track_order = prev


def read_med_multi(filename, **kwargs):
    """Read a multi-mesh MED file. Returns (meshes, mesh_names)."""
    import h5py

    with h5py.File(filename, "r") as f:
        mesh_names = list(f["ENS_MAA"].keys())
        meshes = [_read_single_mesh(f, name) for name in mesh_names]
    return meshes, mesh_names


def _read_single_mesh(f, name):
    mesh_grp = f["ENS_MAA"][name]
    dim = mesh_grp.attrs["ESP"]

    # metadata read from the top mesh group (before descending into a step)
    description = mesh_grp.attrs.get("DES", b"").decode("latin-1").strip().rstrip("\x00")
    unit_time = mesh_grp.attrs.get("UNT", b"").decode("latin-1").strip().rstrip("\x00")
    unit_coords = mesh_grp.attrs.get("UNI", b"").decode("latin-1").strip().rstrip("\x00")

    if "NOE" not in mesh_grp:
        time_step = list(mesh_grp.keys())
        mesh_grp = mesh_grp[time_step[0]]

    point_data = {}
    cell_data = {}
    field_data = {}

    # Points
    pts_dataset = mesh_grp["NOE"]["COO"]
    n_points = pts_dataset.attrs["NBR"]
    points = pts_dataset[()].reshape((n_points, dim), order="F")

    if "FAM" in mesh_grp["NOE"]:
        point_data["point_tags"] = mesh_grp["NOE"]["FAM"][()]

    # FAS: inside the mesh, or at root level
    if "FAS" in mesh_grp:
        fas = mesh_grp["FAS"]
    elif "FAS" in f and name in f["FAS"]:
        fas = f["FAS"][name]
    else:
        fas = None

    point_tags, point_tag_groups = {}, {}
    if fas is not None and "NOEUD" in fas:
        point_tags, point_tag_groups = _read_families(fas["NOEUD"])

    # Cells
    cells = []
    cell_types = []
    med_cells = mesh_grp["MAI"]
    for med_cell_type, med_cell_type_group in med_cells.items():
        cell_type = med_to_meshio_type[med_cell_type]
        cell_types.append(cell_type)
        if med_cell_type in ("POG", "POG2"):
            nod = med_cell_type_group["NOD"][()] - 1
            inn = med_cell_type_group["INN"][()]
            polygons = [nod[inn[i] - 1: inn[i + 1] - 1] for i in range(len(inn) - 1)]
            cells.append((cell_type, polygons))
        else:
            nod = med_cell_type_group["NOD"]
            n_cells = nod.attrs["NBR"]
            data = nod[()].reshape(n_cells, -1, order="F") - 1
            _warn_unconverted_3d(cell_type)
            data = _reorder_med_cells(cell_type, data)  # MED -> meshio order
            cells += [(cell_type, data)]

        if "FAM" in med_cell_type_group:
            cell_data.setdefault("cell_tags", []).append(
                med_cell_type_group["FAM"][()]
            )

    cell_tags, cell_tag_groups = {}, {}
    if fas is not None and "ELEME" in fas:
        cell_tags, cell_tag_groups = _read_families(fas["ELEME"])

    # Fields (filtered to this mesh) - _read_data fills med:step_meta / units / nom
    if "CHA" in f:
        profiles = f["PROFILS"] if "PROFILS" in f else None
        for field_name, field_grp in f["CHA"].items():
            if "@" in field_name:
                logical_name, owner = field_name.rsplit("@", 1)
                if owner != name:
                    continue
            else:
                mai = field_grp.attrs.get("MAI", b"").decode().strip("\x00")
                if mai and mai != name:
                    continue
                logical_name = field_name

            _read_data(
                {logical_name: field_grp},
                profiles,
                cell_types,
                point_data,
                cell_data,
                field_data,
            )

    result = Mesh(
        points, cells,
        point_data=point_data,
        cell_data=cell_data,
        field_data=field_data,
    )
    result.point_tags = point_tags
    result.cell_tags = cell_tags
    result.point_tag_groups = point_tag_groups
    result.cell_tag_groups = cell_tag_groups
    result.mesh_name = name
    result.description = description
    result.unit_time = unit_time
    result.unit_coords = unit_coords
    return result

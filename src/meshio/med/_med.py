"""
I/O for MED/Salome, cf.
<https://docs.salome-platform.org/latest/dev/MEDCoupling/developer/med-file.html>.
"""

import numpy as np
import re


from ._med41 import FieldBitmaskWriter

from .._common import num_nodes_per_cell
from .._exceptions import ReadError, WriteError
from .._helpers import register_format
from collections import defaultdict
from .._mesh import Mesh

# https://docs.salome-platform.org/5/med/dev/med__outils_8hxx.html
meshio_to_med_type = {
    "vertex": "PO1",
    "line": "SE2",
    "line3": "SE3",
    "triangle": "TR3",
    "triangle6": "TR6",
    "triangle7": "TR7",
    "quad": "QU4",
    "quad8": "QU8",
    "quad9": "QU9",
    "tetra": "TE4",
    "tetra10": "T10",
    "hexahedron": "HE8",
    "hexahedron20": "H20",
    "pyramid": "PY5",
    "pyramid13": "P13",
    "wedge": "PE6",
    "wedge15": "P15",
    "polygon": "POG",
    "polygon2": "POG2",
}
med_to_meshio_type = {v: k for k, v in meshio_to_med_type.items()}
numpy_void_str = np.bytes_("")

MED_FLOAT32 = 4
MED_FLOAT64 = 6
MED_INT32 = 24
MED_INT64 = 26

numpy_to_med_type = {
    np.dtype("float32"): MED_FLOAT32,
    np.dtype("float64"): MED_FLOAT64,
    np.dtype("int32"): MED_INT32,
    np.dtype("int64"): MED_INT64,
}

# Dictionnaire de traduction pour le tracker MED 4.1
med_to_geo_type = {
    "PO1": "MED_POINT1",
    "SE2": "MED_SEG2", "SE3": "MED_SEG3", "SE4": "MED_SEG4",
    "TR3": "MED_TRIA3", "TR6": "MED_TRIA6", "TR7": "MED_TRIA7",
    "QU4": "MED_QUAD4", "QU8": "MED_QUAD8", "QU9": "MED_QUAD9",
    "TE4": "MED_TETRA4", "T10": "MED_TETRA10",
    "HE8": "MED_HEXA8", "H20": "MED_HEXA20", "H27": "MED_HEXA27",
    "PY5": "MED_PYRA5", "P13": "MED_PYRA13",
    "PE6": "MED_PENTA6", "P15": "MED_PENTA15", "PE18": "MED_PENTA18",
    "POG": "MED_POLYGON", "POG2": "MED_POLYGON2"
}
med_type_to_entity = {
    "PO1": "MED_NODE_ELEMENT",
    "SE2": "MED_CELL", "SE3": "MED_CELL", "SE4": "MED_CELL",
    "TR3": "MED_CELL", "TR6": "MED_CELL", "TR7": "MED_CELL",
    "QU4": "MED_CELL", "QU8": "MED_CELL", "QU9": "MED_CELL",
    "TE4": "MED_CELL", "T10": "MED_CELL",
    "HE8": "MED_CELL", "H20": "MED_CELL", "H27": "MED_CELL",
    "PY5": "MED_CELL", "P13": "MED_CELL",
    "PE6": "MED_CELL", "P15": "MED_CELL", "PE18": "MED_CELL",
    "POG": "MED_CELL", "POG2": "MED_CELL",
}

def _parse_med_field_name(name):
    """Parse 'Temperature[2] - 0.5' into ('Temperature', 2, 0.5)."""
    m = re.match(r"(.+)\[(\d+)\]\s*-\s*([0-9.eE+-]+)$", name)
    if m:
        try:
            return m.group(1), int(m.group(2)), float(m.group(3))
        except ValueError:
            pass
    return name, None, None

def read(filename):
    import h5py

    f = h5py.File(filename, "r")

    # Mesh ensemble
    mesh_ensemble = f["ENS_MAA"]
    meshes = mesh_ensemble.keys()
    if len(meshes) != 1:
        raise ReadError(f"Must only contain exactly 1 mesh, found {len(meshes)}.")
    mesh_name = list(meshes)[0]
    mesh = mesh_ensemble[mesh_name]
    mesh_description = mesh.attrs.get("DES", b"").decode().strip().rstrip("\x00")
    mesh_unit_time   = mesh.attrs.get("UNT", b"").decode().strip().rstrip("\x00")
    mesh_unit_coords = mesh.attrs.get("UNI", b"").decode().strip().rstrip("\x00")

    dim = mesh.attrs["ESP"]

    # Possible time-stepping
    if "NOE" not in mesh:
        # One needs NOE (node) and MAI (French maillage, meshing) data. If they
        # are not available in the mesh, check for time-steppings.
        time_step = mesh.keys()
        if len(time_step) != 1:
            raise ReadError(
                f"Must only contain exactly 1 time-step, found {len(time_step)}."
            )
        mesh = mesh[list(time_step)[0]]

    # Initialize data
    point_data = {}
    cell_data = {}
    field_data = {}

    # Points
    pts_dataset = mesh["NOE"]["COO"]
    n_points = pts_dataset.attrs["NBR"]
    points = pts_dataset[()].reshape((n_points, dim), order="F")

    # Point tags
    if "FAM" in mesh["NOE"]:
        tags = mesh["NOE"]["FAM"][()]
        point_data["point_tags"] = tags  # replacing previous "point_tags"

    # Information for point tags
    point_tags = {}
    point_tag_groups = {}
    if "FAS" in mesh:        # first check for FAS in the mesh, then in the root group, since some MED files have FAS only in the root group
        fas = mesh["FAS"]
    elif "FAS" in f and mesh_name in f["FAS"]:
        fas = f["FAS"][mesh_name]
    else:
        fas = None           # if FAS is not found, point_tags will be empty and the mesh.point_tags attribute will be an empty dict
    if fas is not None and "NOEUD" in fas:
        point_tags, point_tag_groups = _read_families(fas["NOEUD"])

    # CellBlock
    cells = []
    cell_types = []
    med_cells = mesh["MAI"]
    for med_cell_type, med_cell_type_group in med_cells.items():
        cell_type = med_to_meshio_type[med_cell_type]
        cell_types.append(cell_type)
        if med_cell_type in ("POG", "POG2"):  # polygonal cells with variable node count
            nod = med_cell_type_group["NOD"][()] - 1
            inn = med_cell_type_group["INN"][()]
            polygons = [
                nod[inn[i] - 1 : inn[i + 1] - 1] for i in range(len(inn) - 1)
            ]
            cells.append((cell_type, polygons))
        else:
            nod = med_cell_type_group["NOD"]
            n_cells = nod.attrs["NBR"]
            cells += [(cell_type, nod[()].reshape(n_cells, -1, order="F") - 1)]

        # Cell tags
        if "FAM" in med_cell_type_group:
            tags = med_cell_type_group["FAM"][()]
            if "cell_tags" not in cell_data:
                cell_data["cell_tags"] = []
            cell_data["cell_tags"].append(tags)

    # Information for cell tags
    cell_tags = {}
    cell_tag_groups = {}
    if fas is not None and "ELEME" in fas:
        cell_tags, cell_tag_groups = _read_families(fas["ELEME"])

    # Read nodal and cell data if they exist
    try:
        fields = f["CHA"]  # champs (fields) in French
    except KeyError:
        pass
    else:
        profiles = f["PROFILS"] if "PROFILS" in f else None
        _read_data(fields, profiles, cell_types, point_data, cell_data, field_data)

    # Construct the mesh object
    mesh = Mesh(
        points, cells, point_data=point_data, cell_data=cell_data, field_data=field_data
    )
    mesh.point_tags = point_tags
    mesh.cell_tags = cell_tags
    mesh.mesh_name   = mesh_name
    mesh.description = mesh_description
    mesh.unit_time   = mesh_unit_time
    mesh.unit_coords = mesh_unit_coords
    mesh.point_tag_groups = point_tag_groups
    mesh.cell_tag_groups = cell_tag_groups
    return mesh


def _read_data(fields, profiles, cell_types, point_data, cell_data, field_data):
    if "med:field_units" not in field_data:
        field_data["med:field_units"] = {}
    if "med:step_meta" not in field_data:
        field_data["med:step_meta"] = {}

    for name, data in fields.items():
        # Preserve field units
        field_data["med:field_units"][name] = (
            data.attrs.get("UNI", numpy_void_str),
            data.attrs.get("UNT", numpy_void_str),
        )
        field_data["med:step_meta"][name] = []

        if "NOM" in data.attrs:
            if "med:nom" not in field_data:
                field_data["med:nom"] = []
            field_data["med:nom"].append(data.attrs["NOM"].decode().split())

        time_step = sorted(data.keys())
        if len(time_step) == 1:
            names = [name]
            key = time_step[0]
            med_data = data[key]
            field_data["med:step_meta"][name].append({
                "ndt": med_data.attrs.get("NDT", 0),
                "nor": med_data.attrs.get("NOR", -1),
                "pdt": med_data.attrs["PDT"],
                "key": key,
            })
        else:
            names = []
            for i, key in enumerate(time_step):
                med_data = data[key]
                t = med_data.attrs["PDT"]
                field_data["med:step_meta"][name].append({
                    "ndt": med_data.attrs.get("NDT", i),
                    "nor": med_data.attrs.get("NOR", -1),
                    "pdt": t,
                    "key": key,
                })
                names.append(name + f"[{i:d}] - {t:g}")

        for i, key in enumerate(time_step):
            med_data = data[key]
            name_i = names[i]
            for supp in med_data:
                if supp == "NOE":
                    point_data[name_i] = _read_nodal_data(med_data, profiles)
                else:
                    cell_type = med_to_meshio_type[supp.partition(".")[2]]
                    assert cell_type in cell_types
                    cell_index = cell_types.index(cell_type)
                    if name_i not in cell_data:
                        cell_data[name_i] = [None] * len(cell_types)
                    cell_data[name_i][cell_index] = _read_cell_data(
                        med_data[supp], profiles
                    )

def _read_nodal_data(med_data, profiles):
    profile = med_data["NOE"].attrs["PFL"]
    data_profile = med_data["NOE"][profile]
    n_points = data_profile.attrs["NBR"]
    if profile.decode() == "MED_NO_PROFILE_INTERNAL":  # default profile with everything
        values = data_profile["CO"][()].reshape(n_points, -1, order="F")
    else:
        n_data = profiles[profile].attrs["NBR"]
        index_profile = profiles[profile]["PFL"][()] - 1
        values_profile = data_profile["CO"][()].reshape(n_data, -1, order="F")
        values = np.full((n_points, values_profile.shape[1]), np.nan)
        values[index_profile] = values_profile
    if values.shape[-1] == 1:  # cut off for scalars
        values = values[:, 0]
    return values


def _read_cell_data(med_data, profiles):
    profile = med_data.attrs["PFL"]
    data_profile = med_data[profile]
    n_cells = data_profile.attrs["NBR"]
    n_gauss_points = data_profile.attrs["NGA"]
    if profile.decode() == "MED_NO_PROFILE_INTERNAL":  # default profile with everything
        values = data_profile["CO"][()].reshape(n_cells, n_gauss_points, -1, order="F")
    else:
        n_data = profiles[profile].attrs["NBR"]
        index_profile = profiles[profile]["PFL"][()] - 1
        values_profile = data_profile["CO"][()].reshape(
            n_data, n_gauss_points, -1, order="F"
        )
        values = np.full(
            (n_cells, values_profile.shape[1], values_profile.shape[2]), np.nan
        )
        values[index_profile] = values_profile

    # Only 1 data point per cell, shape -> (n_cells, n_components)
    if n_gauss_points == 1:
        values = values[:, 0, :]
        if values.shape[-1] == 1:  # cut off for scalars
            values = values[:, 0]
    return values


def _read_families(fas_data):
    families = {}
    group_names = {}
    for _, node_set in fas_data.items():
        set_id = node_set.attrs["NUM"]
        group_name = node_set.name.split("/")[-1]
        if "GRO" not in node_set:
            families[set_id] = []
            group_names[set_id] = group_name
            continue
        n_subsets = node_set["GRO"].attrs["NBR"]
        nom_dataset = node_set["GRO"]["NOM"][()]
        name = [None] * n_subsets
        for i in range(n_subsets):
            name[i] = "".join([chr(x) for x in nom_dataset[i]]).strip().rstrip("\x00")
        families[set_id] = name
        group_names[set_id] = group_name
    return families, group_names


def write(filename, mesh, med_version="4.1.0", **kwargs):
    import h5py

    # MED doesn't support compression,
    # <https://github.com/nschloe/meshio/issues/781#issuecomment-616438066>
    # compression = None

    # Use the specified MED version, default 4.1.0
    try:
        version_parts = [int(x) for x in med_version.split(".")]
        major = version_parts[0]
        minor = version_parts[1] if len(version_parts) > 1 else 0
        release = version_parts[2] if len(version_parts) > 2 else 0
    except ValueError:
        major, minor, release = 4, 1, 0
    f = h5py.File(filename, "w", track_order=True)

    # MED file format version
    info = f.create_group("INFOS_GENERALES")
    info.attrs.create("MAJ", major)
    info.attrs.create("MIN", minor)
    info.attrs.create("REL", release)

    # Meshes
    mesh_ensemble = f.create_group("ENS_MAA")
    mesh_name = getattr(mesh, "mesh_name", "mesh")
    med_mesh = mesh_ensemble.create_group(mesh_name)
    med_mesh.attrs.create("DIM", mesh.points.shape[1])  # mesh dimension
    med_mesh.attrs.create("ESP", mesh.points.shape[1])  # spatial dimension
    med_mesh.attrs.create("REP", 0)  # cartesian coordinate system (repère in French)
    unt  = getattr(mesh, "unit_time", "")
    uni  = getattr(mesh, "unit_coords", "")
    desc = getattr(mesh, "description", "Mesh created with meshio")
    med_mesh.attrs.create("UNT", np.bytes_(unt) if unt else numpy_void_str)
    med_mesh.attrs.create("UNI", np.bytes_(uni) if uni else numpy_void_str)
    med_mesh.attrs.create("SRT", 1)  # sorting type MED_SORT_ITDT
    # component names:
    names = ["X", "Y", "Z"][: mesh.points.shape[1]]
    med_mesh.attrs.create("NOM", np.bytes_("".join(f"{name:<16}" for name in names)))
    med_mesh.attrs.create("DES", np.bytes_(desc))
    med_mesh.attrs.create("TYP", 0)  # mesh type (MED_NON_STRUCTURE)

    # Time-step
    step = "-0000000000000000001-0000000000000000001"  # NDT NOR
    time_step = med_mesh.create_group(step)
    time_step.attrs.create("CGT", 1)
    time_step.attrs.create("NDT", -1)  # no time step (-1)
    time_step.attrs.create("NOR", -1)  # no iteration step (-1)
    time_step.attrs.create("PDT", -1.0)  # current time

    # Points
    nodes_group = time_step.create_group("NOE")
    nodes_group.attrs.create("CGT", 1)
    nodes_group.attrs.create("CGS", 1)
    profile = "MED_NO_PROFILE_INTERNAL"
    nodes_group.attrs.create("PFL", np.bytes_(profile))
    coo = nodes_group.create_dataset("COO", data=mesh.points.flatten(order="F"))
    coo.attrs.create("CGT", 1)
    coo.attrs.create("NBR", len(mesh.points))

    # Point tags
    if "point_tags" in mesh.point_data:  # only works for med -> med
        family = nodes_group.create_dataset("FAM", data=mesh.point_data["point_tags"])
        family.attrs.create("CGT", 1)
        family.attrs.create("NBR", len(mesh.points))

    # Cells (mailles in French)
    cells_by_type = {}
    cell_tags_by_type = {}

    for k, cell_block in enumerate(mesh.cells):
        cell_type = cell_block.type
        if cell_type not in cells_by_type:
            cells_by_type[cell_type] = []
            cell_tags_by_type[cell_type] = []
        cells_by_type[cell_type].append(cell_block.data)
        if "cell_tags" in mesh.cell_data:
            cell_tags_by_type[cell_type].append(mesh.cell_data["cell_tags"][k])
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
            # Merge cells of the same type
            merged_cells = np.concatenate(cells_list, axis=0)
            nod = med_cells.create_dataset("NOD", data=merged_cells.flatten(order="F") + 1)
            nod.attrs.create("CGT", 1)
            nod.attrs.create("NBR", len(merged_cells))
            n_merged = len(merged_cells)

        # Cell tags
        if cell_tags_by_type.get(cell_type):
            merged_tags = np.concatenate(cell_tags_by_type[cell_type])
            family = med_cells.create_dataset("FAM", data=merged_tags)
            family.attrs.create("CGT", 1)
            family.attrs.create("NBR", n_merged)

    # Information about point and cell sets (familles in French)
    fas = f.create_group("FAS")
    families = fas.create_group(mesh_name)
    family_zero = families.create_group("FAMILLE_ZERO")  # must be defined in any case
    family_zero.attrs.create("NUM", 0)

    # For point tags
    try:
        if len(mesh.point_tags) > 0:
            node = families.create_group("NOEUD")
            _write_families(node, mesh.point_tags, getattr(mesh, "point_tag_groups", {}))
    except AttributeError:
        pass

    # For cell tags
    try:
        if len(mesh.cell_tags) > 0:
            element = families.create_group("ELEME")
            _write_families(element, mesh.cell_tags, getattr(mesh, "cell_tag_groups", {}))
    except AttributeError:
        pass

    # Write nodal/cell data
        # Fields
    has_point_data = any(k != "point_tags" for k in mesh.point_data)
    has_cell_data = any(k not in ("cell_tags", "gmsh:physical") for k in mesh.cell_data)

    if not has_point_data and not has_cell_data:
        f.close()
        return

    fields = f.create_group("CHA")
    field_comp_names = mesh.field_data.get("med:nom", [])
    step_meta = mesh.field_data.get("med:step_meta", {})      
    field_units = mesh.field_data.get("med:field_units", {})  
    name_idx = 0

    # Nodal data grouped by base field name for multi-timestep support
    nodal_groups = defaultdict(list)
    for name, data in mesh.point_data.items():
        if name == "point_tags":
            continue
        base, idx, pdt = _parse_med_field_name(name)
        nodal_groups[base].append((idx, pdt, data))

    for base_name, entries in nodal_groups.items():
        entries.sort(key=lambda x: x[0] if x[0] is not None else 0)
        comp_name = field_comp_names[name_idx] if name_idx < len(field_comp_names) else None
        name_idx += 1

        first_data = entries[0][2]
        n_components = 1 if first_data.ndim == 1 else first_data.shape[-1]
        units = field_units.get(base_name, (numpy_void_str, numpy_void_str)) 
        try:
            field = fields.create_group(base_name)
            field.attrs.create("MAI", np.bytes_(mesh_name))
            field.attrs.create("TYP", numpy_to_med_type.get(first_data.dtype, MED_FLOAT64))
            field.attrs.create("NCO", n_components)
            field.attrs.create("UNI", units[0] if units[0] is not None else numpy_void_str)  
            field.attrs.create("UNT", units[1] if units[1] is not None else numpy_void_str)  
            nom = (
                np.bytes_("".join(f"{n:<16}" for n in comp_name))
                if comp_name
                else np.bytes_(f"{'':<16}")
            )
            field.attrs.create("NOM", nom)
        except ValueError:
            field = fields[base_name]

        tracker = FieldBitmaskWriter()

        meta_list = step_meta.get(base_name, [])
        for i, (idx, pdt_orig, data) in enumerate(entries):
            meta = meta_list[i] if i < len(meta_list) else {}
            ndt = meta.get("ndt", i + 1)
            nor = meta.get("nor", -1)
            pdt = meta.get("pdt", pdt_orig if pdt_orig is not None else 0.0)
            step_name = f"{ndt:020d}{nor:020d}"
            if step_name not in field:
                ts = field.create_group(step_name)
                ts.attrs.create("NDT", ndt)
                ts.attrs.create("NOR", nor)
                ts.attrs.create("PDT", pdt)
                ts.attrs.create("RDT", -1)
                ts.attrs.create("ROR", -1)
            else:
                ts = field[step_name]

            typ = ts.create_group("NOE")
            typ.attrs.create("GAU", numpy_void_str)
            typ.attrs.create("PFL", np.bytes_(profile))
            profile_grp = typ.create_group(profile)
            profile_grp.attrs.create("NBR", len(data))
            profile_grp.attrs.create("NGA", 1)
            profile_grp.attrs.create("GAU", numpy_void_str)
            profile_grp.create_dataset("CO", data=data.flatten(order="F"))
            tracker.notify("MED_NODE", "MED_NO_GEOTYPE", step_name)

        tracker.flush(field)

    # Cell data grouped by base field name for multi-timestep support
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
        comp_name = field_comp_names[name_idx] if name_idx < len(field_comp_names) else None
        name_idx += 1

        first_data = entries[0][3]
        n_components = 1 if first_data.ndim == 1 else first_data.shape[-1]

        try:
            field = fields.create_group(base_name)
            field.attrs.create("MAI", np.bytes_(mesh_name))
            field.attrs.create("TYP", numpy_to_med_type.get(first_data.dtype, MED_FLOAT64))
            field.attrs.create("NCO", n_components)
            field.attrs.create("UNI", numpy_void_str)
            field.attrs.create("UNT", numpy_void_str)
            nom = (
                np.bytes_("".join(f"{n:<16}" for n in comp_name))
                if comp_name
                else np.bytes_(f"{'':<16}")
            )
            field.attrs.create("NOM", nom)
        except ValueError:
            field = fields[base_name]

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
                    continue
            else:
                supp = "ELEM"

            meta = meta_list[i] if i < len(meta_list) else {}
            ndt = meta.get("ndt", i + 1)
            nor = meta.get("nor", -1)
            pdt = meta.get("pdt", pdt_orig if pdt_orig is not None else 0.0)
            step_name = f"{ndt:020d}{nor:020d}"

            if step_name not in field:
                ts = field.create_group(step_name)
                ts.attrs.create("NDT", ndt)
                ts.attrs.create("NOR", nor)
                ts.attrs.create("PDT", pdt)
                ts.attrs.create("RDT", -1)
                ts.attrs.create("ROR", -1)
            else:
                ts = field[step_name]

            if supp == "ELNO":
                typ = ts.create_group("NOE." + med_type)
            else:
                typ = ts.create_group("MAI." + med_type)

            typ.attrs.create("GAU", numpy_void_str)
            typ.attrs.create("PFL", np.bytes_(profile))
            profile_grp = typ.create_group(profile)
            profile_grp.attrs.create("NBR", len(data))
            profile_grp.attrs.create("NGA", data.shape[1] if supp == "ELNO" else 1)
            profile_grp.attrs.create("GAU", numpy_void_str)
            profile_grp.create_dataset("CO", data=data.flatten(order="F"))

            tracker.notify("MED_CELL", med_to_geo_type.get(med_type, med_type), step_name)

        tracker.flush(field)

    f.close()

def _write_data(
    fields,
    mesh_name,
    field_name,
    profile,
    name,
    supp,
    data,
    med_type=None,
):
    # Skip for general ELGA fields defined at unknown Gauss points
    if supp == "ELGA":
        return

    # Field
    try:  # a same MED field may contain fields of different natures
        field = fields.create_group(name)
        field.attrs.create("MAI", np.bytes_(mesh_name))
        field.attrs.create("TYP", numpy_to_med_type[data.dtype])
        field.attrs.create("UNI", numpy_void_str)  # physical unit
        field.attrs.create("UNT", numpy_void_str)  # time unit
        n_components = 1 if data.ndim == 1 else data.shape[-1]
        field.attrs.create("NCO", n_components)  # number of components
        # names = _create_component_names(n_components)
        # field.attrs.create("NOM", np.bytes_("".join(f"{name:<16}" for name in names)))

        if field_name:
            field.attrs.create(
                "NOM", np.bytes_("".join(f"{name:<16}" for name in field_name))
            )
        else:
            field.attrs.create("NOM", np.bytes_(f"{'':<16}"))

        # Time-step
        step = "0000000000000000000100000000000000000001"
        time_step = field.create_group(step)
        time_step.attrs.create("NDT", 1)  # time step 1
        time_step.attrs.create("NOR", 1)  # iteration step 1
        time_step.attrs.create("PDT", 0.0)  # current time
        time_step.attrs.create("RDT", -1)  # NDT of the mesh
        time_step.attrs.create("ROR", -1)  # NOR of the mesh

    except ValueError:  # name already exists
        field = fields[name]
        ts_name = list(field.keys())[-1]
        time_step = field[ts_name]

    # Field information
    if supp == "NOEU":
        typ = time_step.create_group("NOE")
    elif supp == "ELNO":
        typ = time_step.create_group("NOE." + med_type)
    else:  # 'ELEM' with only 1 Gauss points!
        typ = time_step.create_group("MAI." + med_type)

    typ.attrs.create("GAU", numpy_void_str)  # no associated Gauss points
    typ.attrs.create("PFL", np.bytes_(profile))
    profile = typ.create_group(profile)
    profile.attrs.create("NBR", len(data))  # number of data
    if supp == "ELNO":
        profile.attrs.create("NGA", data.shape[1])
    else:
        profile.attrs.create("NGA", 1)
    profile.attrs.create("GAU", numpy_void_str)

    # Dataset
    profile.create_dataset("CO", data=data.flatten(order="F"))


def _create_component_names(n_components):
    """To be correctly read in a MED viewer, each component must be a string of width
    16. Since we do not know the physical nature of the data, we just use V1, V2,...
    """
    return [f"V{(i+1)}" for i in range(n_components)]


def _family_name(set_id, name):
    """Return the FAM object name corresponding to the unique set id and a list of
    subset names
    """
    return "FAM_" + str(set_id) + "_" + "_".join(name)


def _write_families(fm_group, tags, group_names=None):
    """Write MED family groups under FAS/[mesh_name]/NOEUD or ELEME.

    A family with no named groups must NOT have a GRO subgroup.
    GRO/NOM must be a H5T_ARRAY{[80] H5T_NATIVE_CHAR} dataset (one 80-char
    slot per group name), NOT a H5T_STRING/S80 dataset.

    If group_names is provided, the original HDF5 family directory name is
    reused instead of being regenerated by _family_name().
    """
    group_names = group_names or {}
    for set_id, name in tags.items():
        gname = group_names.get(set_id, _family_name(set_id, name))
        family = fm_group.create_group(gname)
        family.attrs.create("NUM", set_id)

        if not name:
            continue  # no groups: omit GRO entirely per MED spec

        group = family.create_group("GRO")
        group.attrs.create("NBR", len(name))

        dataset = group.create_dataset(
            "NOM", (len(name),), dtype=np.dtype(("i1", (80,)))
        )
        buf = np.full((len(name), 80), ord(" "), dtype="i1")
        for i, n in enumerate(name):
            name_bytes = n.encode("latin-1", "replace")
            if len(name_bytes) > 80:
                raise WriteError(
                    f"Family name '{n}' is too long for MED format (max 80 bytes)."
                )
            buf[i, : len(name_bytes)] = np.frombuffer(name_bytes, dtype="i1")
        dataset[...] = buf


register_format("med", [".med"], read, {"med": write})

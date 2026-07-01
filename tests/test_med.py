import pathlib

import numpy as np
import pytest 
import copy

import meshlane
from meshlane.med._med import numpy_to_med_type
from meshlane.med._med41 import (
    FieldBitmaskWriter,
    decode_entity_mask,
    decode_geo_mask,
    _bit_set,
    _bit_test,
)

from . import helpers

h5py = pytest.importorskip("h5py")


@pytest.mark.parametrize(
    "mesh",
    [
        helpers.empty_mesh,
        helpers.line_mesh,
        helpers.tri_mesh_2d,
        helpers.tri_mesh,
        helpers.triangle6_mesh,
        helpers.quad_mesh,
        helpers.quad8_mesh,
        helpers.quad_tri_mesh,
        helpers.tet_mesh,
        helpers.tet10_mesh,
        helpers.hex_mesh,
        helpers.hex20_mesh,
        helpers.add_point_data(helpers.tri_mesh, 1),
        helpers.add_point_data(helpers.tri_mesh, 2),
        helpers.add_point_data(helpers.tri_mesh, 3),
        helpers.add_point_data(helpers.hex_mesh, 3),
        helpers.add_cell_data(helpers.tri_mesh, [("a", (), np.float64)]),
        helpers.add_cell_data(helpers.tri_mesh, [("a", (2,), np.float64)]),
        helpers.add_cell_data(helpers.tri_mesh, [("a", (3,), np.float64)]),
        
    ],
)
def test_io(mesh, tmp_path):
    helpers.write_read(tmp_path, meshlane.med.write, meshlane.med.read, mesh, 1.0e-15)


def test_generic_io(tmp_path):
    helpers.generic_io(tmp_path / "test.med")
    # With additional, insignificant suffix:
    helpers.generic_io(tmp_path / "test.0.med")


def test_reference_file_with_mixed_cells(tmp_path):
    this_dir = pathlib.Path(__file__).resolve().parent
    filename = this_dir / "meshes" / "med" / "cylinder.med"
    mesh = meshlane.read(filename)

    # Points
    assert np.isclose(mesh.points.sum(), 16.53169892762988)

    # CellBlock
    ref_num_cells = {"pyramid": 18, "quad": 18, "line": 17, "tetra": 63, "triangle": 4}
    assert {
        cell_block.type: len(cell_block) for cell_block in mesh.cells
    } == ref_num_cells

    # Point tags
    assert mesh.point_data["point_tags"].sum() == 52
    ref_point_tags_info = {2: ["Side"], 3: ["Side", "Top"], 4: ["Top"]}
    assert mesh.point_tags == ref_point_tags_info

    # Cell tags
    ref_sum_cell_tags = {
        "pyramid": -116,
        "quad": -75,
        "line": -48,
        "tetra": -24,
        "triangle": -30,
    }
    assert {
        c.type: sum(d) for c, d in zip(mesh.cells, mesh.cell_data["cell_tags"])
    } == ref_sum_cell_tags
    ref_cell_tags_info = {
        -6: ["Top circle"],
        -7: ["Top", "Top and down"],
        -8: ["Top and down"],
        -9: ["A", "B"],
        -10: ["B"],
        -11: ["B", "C"],
        -12: ["C"],
    }
    assert mesh.cell_tags == ref_cell_tags_info

    helpers.write_read(tmp_path, meshlane.med.write, meshlane.med.read, mesh, 1.0e-15)


def test_reference_file_with_point_cell_data(tmp_path):
    this_dir = pathlib.Path(__file__).resolve().parent
    filename = this_dir / "meshes" / "med" / "box.med"

    mesh = meshlane.read(filename)

    # Points
    assert np.isclose(mesh.points.sum(), 12)

    # CellBlock
    assert {cell_block.type: len(cell_block) for cell_block in mesh.cells} == {
        "hexahedron": 1
    }

    # Point data
    data_u = mesh.point_data["resu____DEPL"]
    assert data_u.shape == (8, 3)
    assert np.isclose(data_u.sum(), 12)

    # Cell data
    # ELNO (1 data point for every node of each element)
    data_eps = mesh.cell_data["resu____EPSI_ELNO"][0]
    assert data_eps.shape == (1, 8, 6)  # (n_cells, n_nodes_per_element, n_components)
    data_eps_mean = np.mean(data_eps, axis=1)[0]
    eps_ref = np.array([1, 0, 0, 0.5, 0.5, 0])
    assert np.allclose(data_eps_mean, eps_ref)

    data_sig = mesh.cell_data["resu____SIEF_ELNO"][0]
    assert data_sig.shape == (1, 8, 6)  # (n_cells, n_nodes_per_element, n_components)
    data_sig_mean = np.mean(data_sig, axis=1)[0]
    sig_ref = np.array(
        [7328.44611253, 2645.87030114, 2034.06063679, 1202.6, 569.752, 0]
    )
    assert np.allclose(data_sig_mean, sig_ref)

    data_psi = mesh.cell_data["resu____ENEL_ELNO"][0]
    assert data_psi.shape == (1, 8, 1)  # (n_cells, n_nodes_per_element, n_components)

    # ELEM (1 data point for each element)
    data_psi_elem = mesh.cell_data["resu____ENEL_ELEM"][0]
    assert np.isclose(np.mean(data_psi, axis=1)[0, 0], data_psi_elem[0])

    helpers.write_read(tmp_path, meshlane.med.write, meshlane.med.read, mesh, 1.0e-15)



def test_read_med_without_fas(tmp_path):
    """Un fichier MED sans section FAS ne doit pas crasher."""
    filename = tmp_path / "no_fas.med"

    # Créer un MED minimal sans FAS avec h5py
    with h5py.File(filename, "w") as f:
        info = f.create_group("INFOS_GENERALES")
        info.attrs.create("MAJ", 3)
        info.attrs.create("MIN", 0)
        info.attrs.create("REL", 0)

        ens = f.create_group("ENS_MAA")
        maa = ens.create_group("mesh")
        maa.attrs.create("DIM", 2)
        maa.attrs.create("ESP", 2)
        maa.attrs.create("REP", 0)
        maa.attrs.create("UNT", np.bytes_(""))
        maa.attrs.create("UNI", np.bytes_(""))
        maa.attrs.create("SRT", 1)
        maa.attrs.create("NOM", np.bytes_(f"{'X':<16}{'Y':<16}"))
        maa.attrs.create("DES", np.bytes_("test"))
        maa.attrs.create("TYP", 0)

        step = maa.create_group("-0000000000000000001-0000000000000000001")
        step.attrs.create("CGT", 1)
        step.attrs.create("NDT", -1)
        step.attrs.create("NOR", -1)
        step.attrs.create("PDT", -1.0)

        # 3 points, 1 triangle - pas de FAS
        noe = step.create_group("NOE")
        noe.attrs.create("CGT", 1)
        noe.attrs.create("CGS", 1)
        noe.attrs.create("PFL", np.bytes_("MED_NO_PROFILE_INTERNAL"))
        pts = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0])  # 3 points × 2D, Fortran order
        coo = noe.create_dataset("COO", data=pts)
        coo.attrs.create("CGT", 1)
        coo.attrs.create("NBR", 3)

        mai = step.create_group("MAI")
        mai.attrs.create("CGT", 1)
        tr3 = mai.create_group("TR3")
        tr3.attrs.create("CGT", 1)
        tr3.attrs.create("CGS", 1)
        tr3.attrs.create("PFL", np.bytes_("MED_NO_PROFILE_INTERNAL"))
        nod = tr3.create_dataset("NOD", data=np.array([1, 2, 3]))  # 1-indexed
        nod.attrs.create("CGT", 1)
        nod.attrs.create("NBR", 1)

    # Doit lire sans crasher
    mesh = meshlane.med.read(filename)
    assert len(mesh.points) == 3
    assert len(mesh.cells) == 1
    assert mesh.cells[0].type == "triangle"


def test_read_med_without_gro(tmp_path):
    """Une famille sans sous-groupe GRO ne doit pas crasher."""
    filename = tmp_path / "no_gro.med"

    # Écrire un mesh normal puis modifier le FAS
    mesh = helpers.tri_mesh
    meshlane.med.write(filename, mesh)

    # Ajouter une famille SANS GRO dans le FAS
    with h5py.File(filename, "a") as f:
        fas_mesh = None
        if "FAS" in f:
            for key in f["FAS"]:
                fas_mesh = f["FAS"][key]
                break

        if fas_mesh is not None:
            if "ELEME" not in fas_mesh:
                fas_mesh.create_group("ELEME")
            eleme = fas_mesh["ELEME"]
            fam = eleme.create_group("FAM_NO_GRO")
            fam.attrs.create("NUM", -99)
            # Pas de GRO ici

    mesh_out = meshlane.med.read(filename)
    assert len(mesh_out.points) > 0
    assert len(mesh_out.cells) > 0


def test_write_multi_blocks_same_type_with_cell_data(tmp_path):
    """Multiple blocks of the same type with cell_data must be merged."""
    from meshlane._mesh import CellBlock

    points = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, 0.0],
        [2.0, 1.0],
    ])

    cells = [
        CellBlock("triangle", np.array([[0, 1, 2], [1, 3, 2]])),
        CellBlock("triangle", np.array([[1, 4, 5], [1, 5, 3]])),
    ]

    cell_data = {
        "cell_tags": [
            np.array([-1, -1]),
            np.array([-2, -2]),
        ]
    }

    mesh = meshlane.Mesh(points, cells, cell_data=cell_data)
    filename = tmp_path / "multi_blocks.med"

    meshlane.med.write(filename, mesh)

    # Re-read: triangles are merged into 1 block
    mesh_out = meshlane.med.read(filename)
    total_tri = sum(
        len(c.data) for c in mesh_out.cells if c.type == "triangle"
    )
    assert total_tri == 4

    # Cell tags must be merged in the correct order
    assert "cell_tags" in mesh_out.cell_data
    tags = np.concatenate([
        t for c, t in zip(mesh_out.cells, mesh_out.cell_data["cell_tags"])
        if c.type == "triangle"
    ])
    assert np.array_equal(tags, np.array([-1, -1, -2, -2]))


def test_read_med_partial_cell_data(tmp_path):
    """A field defined on only one cell type must not crash."""
    filename = tmp_path / "partial.med"

    from meshlane._mesh import CellBlock

    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])

    cells = [
        CellBlock("triangle", np.array([[0, 1, 2]])),
        CellBlock("tetra", np.array([[0, 1, 2, 3]])),
    ]

    mesh = meshlane.Mesh(points, cells)
    meshlane.med.write(filename, mesh)

    # Add a CHA field only on tetra via h5py
    with h5py.File(filename, "a") as f:
        if "CHA" not in f:
            f.create_group("CHA")

        field = f["CHA"].create_group("test_field")
        field.attrs.create("MAI", np.bytes_("mesh"))
        field.attrs.create("TYP", 6)
        field.attrs.create("UNI", np.bytes_(""))
        field.attrs.create("UNT", np.bytes_(""))
        field.attrs.create("NCO", 1)
        field.attrs.create("NOM", np.bytes_(f"{'':<16}"))

        step = field.create_group("0000000000000000000100000000000000000001")
        step.attrs.create("NDT", 1)
        step.attrs.create("NOR", 1)
        step.attrs.create("PDT", 0.0)
        step.attrs.create("RDT", -1)
        step.attrs.create("ROR", -1)

        profile = "MED_NO_PROFILE_INTERNAL"
        typ = step.create_group("MAI.TE4")
        typ.attrs.create("GAU", np.bytes_(""))
        typ.attrs.create("PFL", np.bytes_(profile))
        pfl = typ.create_group(profile)
        pfl.attrs.create("NBR", 1)
        pfl.attrs.create("NGA", 1)
        pfl.attrs.create("GAU", np.bytes_(""))
        pfl.create_dataset("CO", data=np.array([42.0]))

    # Must read without TypeError: len() of unsized object
    mesh_out = meshlane.med.read(filename)
    assert len(mesh_out.cells) >= 2

    # Field must exist on tetra, None on triangle
    assert "test_field" in mesh_out.cell_data
    field_data = mesh_out.cell_data["test_field"]
    tetra_idx = next(
        i for i, c in enumerate(mesh_out.cells) if c.type == "tetra"
    )
    assert field_data[tetra_idx] is not None
    assert np.isclose(field_data[tetra_idx].flat[0], 42.0)


@pytest.mark.parametrize("dtype, expected_med_type", [
    (np.float32, 4),   # MED_FLOAT32
    (np.float64, 6),   # MED_FLOAT64
    (np.int32,   24),  # MED_INT32
    (np.int64,   26),  # MED_INT64
])
def test_med_type_mapping(dtype, expected_med_type):
    """Check that numpy dtype maps to the correct MED type constant."""
    data = np.array([1, 2, 3], dtype=dtype)
    result = numpy_to_med_type[data.dtype]
    assert result == expected_med_type, (
        f"dtype={dtype.__name__}: "
        f"expected {expected_med_type}, got {result}"
    )


def test_med_type_mapping_unknown_dtype():
    """Unsupported dtype should raise KeyError."""
    data = np.array([1, 2, 3], dtype=np.complex128)
    with pytest.raises(KeyError):
        _ = numpy_to_med_type[data.dtype]


@pytest.mark.parametrize("dtype, expected_med_type", [
    (np.float32, 4),   # MED_FLOAT32
    (np.float64, 6),   # MED_FLOAT64
    (np.int32,   24),  # MED_INT32
    (np.int64,   26),  # MED_INT64
])
def test_med_type_preserved_after_write_read(tmp_path, dtype, expected_med_type):
    """
    Check that TYP written in HDF5 matches the expected MED type
    after a meshlane write.
    """
    filename = tmp_path / f"test_roundtrip_{dtype.__name__}.med"

    mesh = helpers.add_point_data(helpers.tri_mesh, 1)
    for key in mesh.point_data:
        mesh.point_data[key] = mesh.point_data[key].astype(dtype)

    meshlane.med.write(filename, mesh)

    with h5py.File(filename, "r") as f:
        if "CHA" in f:
            for field_name in f["CHA"]:
                written_type = f["CHA"][field_name].attrs.get("TYP")
                if written_type is not None:
                    assert written_type == expected_med_type, (
                        f"Field '{field_name}', dtype={dtype.__name__}: "
                        f"TYP={written_type}, expected={expected_med_type}"
                    )


@pytest.mark.parametrize("med_version, expected", [
    ("4.1.0", (4, 1, 0)),
    ("4.0.0", (4, 0, 0)),
    ("3.0.0", (3, 0, 0)),
])
def test_med_version_written(tmp_path, med_version, expected):
    """Check that the specified MED version is written to the HDF5 file."""
    filename = tmp_path / f"test_v{med_version}.med"
    mesh = helpers.tri_mesh
    meshlane.med.write(filename, mesh, med_version=med_version)

    with h5py.File(filename, "r") as f:
        info = f["INFOS_GENERALES"]
        assert int(info.attrs["MAJ"]) == expected[0]
        assert int(info.attrs["MIN"]) == expected[1]
        assert int(info.attrs["REL"]) == expected[2]


def test_med_version_default(tmp_path):
    """Default MED version should be 4.1.0."""
    filename = tmp_path / "test_default.med"
    mesh = helpers.tri_mesh
    meshlane.med.write(filename, mesh)

    with h5py.File(filename, "r") as f:
        info = f["INFOS_GENERALES"]
        assert int(info.attrs["MAJ"]) == 4
        assert int(info.attrs["MIN"]) == 1
        assert int(info.attrs["REL"]) == 0


def test_bit_set():
    """_bit_set sets the correct bit position."""
    mask = np.uint32(0)
    mask = _bit_set(mask, 0)  # bit 0 -> 0b00001 = 1
    assert int(mask) == 1

    mask = _bit_set(mask, 1)  # bit 1 -> 0b00011 = 3
    assert int(mask) == 3

    mask = _bit_set(mask, 3)  # bit 3 -> 0b01011 = 11
    assert int(mask) == 11


def test_bit_test():
    """_bit_test detects whether a bit is set."""
    mask = np.uint32(0b00101)  # bits 0 and 2 set
    assert _bit_test(mask, 0) is True
    assert _bit_test(mask, 1) is False
    assert _bit_test(mask, 2) is True
    assert _bit_test(mask, 3) is False


def test_decode_entity_mask_empty():
    """A zero mask yields no entities."""
    result = decode_entity_mask(np.uint32(0))
    assert result == []


def test_decode_entity_mask_node():
    """Bit 3 = MED_NODE."""
    mask = np.uint32(0b001000)
    result = decode_entity_mask(mask)
    assert result == ["MED_NODE"]


def test_decode_entity_mask_cell():
    """Bit 0 = MED_CELL."""
    mask = np.uint32(0b000001)
    result = decode_entity_mask(mask)
    assert result == ["MED_CELL"]


def test_decode_entity_mask_multiple():
    """Bits 0 and 3 = MED_CELL + MED_NODE."""
    mask = np.uint32(0b001001)  # bits 0 and 3
    result = decode_entity_mask(mask)
    assert "MED_CELL" in result
    assert "MED_NODE" in result
    assert len(result) == 2


def test_decode_geo_mask_triangle():
    """MED_TRIA3 is at position 4 in MED_CELL."""
    # bit 4 -> 0b010000 = 16
    mask = np.uint32(1 << 4)
    result = decode_geo_mask("MED_CELL", mask)
    assert result == ["MED_TRIA3"]


def test_decode_geo_mask_empty():
    """A zero mask yields no geometry types."""
    result = decode_geo_mask("MED_CELL", np.uint32(0))
    assert result == []


def test_bitmask_writer_notify_node():
    """After notify on MED_NODE, the global entity mask must have bit 3 set."""
    writer = FieldBitmaskWriter()
    step = "0000000000000000000100000000000000000001"
    writer.notify("MED_NODE", "MED_NO_GEOTYPE", step)

    assert _bit_test(writer._g_entity, 3)


def test_bitmask_writer_notify_cell():
    """After notify on MED_CELL/MED_TRIA3, bit 0 (entity) and bit 4 (geo) must be set."""
    writer = FieldBitmaskWriter()
    step = "0000000000000000000100000000000000000001"
    writer.notify("MED_CELL", "MED_TRIA3", step)

    assert _bit_test(writer._g_entity, 0)
    # MED_TRIA3 is at index 4 in MED_CELL
    assert _bit_test(writer._g_geo["MED_CELL"], 4)


def test_bitmask_writer_notify_multiple_steps():
    """Multiple time steps must be tracked separately."""
    writer = FieldBitmaskWriter()
    step1 = "0000000000000000000100000000000000000001"
    step2 = "0000000000000000000200000000000000000002"

    writer.notify("MED_NODE", "MED_NO_GEOTYPE", step1)
    writer.notify("MED_CELL", "MED_TRIA3", step2)

    # step1: only MED_NODE (bit 3)
    assert _bit_test(writer._s_entity[step1], 3)
    assert not _bit_test(writer._s_entity[step1], 0)

    # step2: only MED_CELL (bit 0)
    assert _bit_test(writer._s_entity[step2], 0)
    assert not _bit_test(writer._s_entity[step2], 3)


def test_bitmask_writer_flush(tmp_path):
    """flush() must write LEN, LGN, LNA, LAA attributes to the HDF5 field group."""
    filename = tmp_path / "test_bitmask.med"
    writer = FieldBitmaskWriter()
    step = "0000000000000000000100000000000000000001"
    writer.notify("MED_NODE", "MED_NO_GEOTYPE", step)

    with h5py.File(filename, "w") as f:
        field_grp = f.create_group("test_field")
        field_grp.create_group(step)  # step group must exist
        writer.flush(field_grp)

    with h5py.File(filename, "r") as f:
        grp = f["test_field"]

        # LEN = global entity mask
        assert "LEN" in grp.attrs
        len_mask = np.uint32(int(grp.attrs["LEN"]))
        assert _bit_test(len_mask, 3)  # MED_NODE = bit 3

        # LNA = number of time steps where MED_NODE is present
        assert "LNA" in grp.attrs

        # LAA = number of time steps where all entity types are present
        assert "LAA" in grp.attrs
        assert int(grp.attrs["LAA"]) == 1


def test_bitmask_written_in_real_med_file(tmp_path):
    """After a full meshlane write, bitmask attributes must exist in CHA fields."""
    filename = tmp_path / "test_bitmask_full.med"

    mesh = helpers.add_point_data(helpers.tri_mesh, 1)
    meshlane.med.write(filename, mesh)

    with h5py.File(filename, "r") as f:
        assert "CHA" in f
        for field_name in f["CHA"]:
            field_grp = f["CHA"][field_name]

            assert "LEN" in field_grp.attrs, (
                f"Field '{field_name}': LEN attribute missing"
            )

            assert "LAA" in field_grp.attrs, (
                f"Field '{field_name}': LAA attribute missing"
            )

            # LEN mask must have MED_NODE (bit 3) set
            len_mask = np.uint32(int(field_grp.attrs["LEN"]))
            assert _bit_test(len_mask, 3), (
                f"Field '{field_name}': MED_NODE bit not set in LEN"
            )


def test_polygonal_cells():
    this_dir = pathlib.Path(__file__).resolve().parent
    filename = this_dir / "meshes" / "med" / "voronoi_hex.med"

    mesh = meshlane.read(filename)

    # Points
    assert np.isclose(mesh.points.sum(), 3.869519702231004)

    # Number of points
    assert len(mesh.points) == 124

    # CellBlock: 60 polygons
    ref_num_cells = {"polygon": 60}
    assert {
        cell_block.type: len(cell_block) for cell_block in mesh.cells
    } == ref_num_cells

    # Polygons must have between 4 and 7 vertices
    for cell_block in mesh.cells:
        if cell_block.type == "polygon":
            sizes = [len(cell) for cell in cell_block.data]
            assert min(sizes) == 4
            assert max(sizes) == 7

    # Point data and cell data must be present
    assert "point_tags" in mesh.point_data
    assert "cell_tags" in mesh.cell_data


def test_polygonal_cells_write_read(tmp_path):
    """Round-trip: read polygon mesh, write it, read it back."""
    this_dir = pathlib.Path(__file__).resolve().parent
    filename = this_dir / "meshes" / "med" / "voronoi_hex.med"

    mesh = meshlane.read(filename)
    out = tmp_path / "polygons_roundtrip.med"
    meshlane.med.write(out, mesh)

    mesh2 = meshlane.med.read(out)
    assert len(mesh2.points) == len(mesh.points)
    assert np.allclose(mesh2.points, mesh.points)

    # Same number of polygon cells
    orig_count = sum(len(c) for c in mesh.cells if c.type == "polygon")
    read_count = sum(len(c) for c in mesh2.cells if c.type == "polygon")
    assert orig_count == read_count

    # Same polygon sizes
    for cb1, cb2 in zip(mesh.cells, mesh2.cells):
        if cb1.type == "polygon":
            sizes1 = [len(c) for c in cb1.data]
            sizes2 = [len(c) for c in cb2.data]
            assert sizes1 == sizes2


def test_family_group_names_round_trip(tmp_path):
    """Family group names must survive a write/read round-trip."""
    filename = tmp_path / "fam_round_trip.med"
    mesh = helpers.tri_mesh
    mesh.point_tags = {-1: ["alpha", "beta"], -2: ["gamma"]}
    meshlane.med.write(filename, mesh)

    mesh_out = meshlane.med.read(filename)
    assert mesh_out.point_tags == {-1: ["alpha", "beta"], -2: ["gamma"]}


def test_family_with_no_groups_omits_GRO(tmp_path):
    """A family with an empty group list must NOT create a GRO subgroup."""
    filename = tmp_path / "fam_empty.med"
    mesh = helpers.tri_mesh
    mesh.point_tags = {-42: []}
    meshlane.med.write(filename, mesh)

    with h5py.File(filename, "r") as f:
        family = f["FAS/mesh/NOEUD/FAM_-42_"]
        assert "GRO" not in family
        assert int(family.attrs["NUM"]) == -42


def test_nom_dataset_dtype_is_array_i1_80(tmp_path):
    """
    The GRO/NOM dataset must have the dtype H5T_ARRAY{[80] char},
    i.e. np.dtype(('i1', (80,))).
    """
    this_dir = pathlib.Path(__file__).resolve().parent
    filename = this_dir / "meshes" / "med" / "cylinder.med"
    filename_out = tmp_path / "input_code_aster.med"

    mesh_out = meshlane.med.read(filename)
    meshlane.med.write(filename_out, mesh_out)

    with h5py.File(filename_out, "r") as f:
        mesh_name = list(f["ENS_MAA"].keys())[0]
        fas = f["FAS"][mesh_name]
        for section in ("NOEUD", "ELEME"):
            if section not in fas:
                continue
            for gname, grp in fas[section].items():
                if "GRO" not in grp:
                    continue
                nom_ds = grp["GRO"]["NOM"]
                assert nom_ds.dtype == np.dtype(("i1", (80,))), (
                    f"FAS/{section}/{gname}/GRO/NOM : "
                    f"expected dtype ('i1', (80,)), got {nom_ds.dtype}"
                )


def test_nom_dataset_padded_with_spaces(tmp_path):
    """
    The padding of GRO/NOM must be spaces (0x20),
    not zeros (0x00).
    """
    this_dir = pathlib.Path(__file__).resolve().parent
    filename = this_dir / "meshes" / "med" / "cylinder.med"
    filename_out = tmp_path / "input_code_aster.med"

    mesh_out = meshlane.med.read(filename)
    meshlane.med.write(filename_out, mesh_out)

    with h5py.File(filename_out, "r") as f:
        mesh_name = list(f["ENS_MAA"].keys())[0]
        fas = f["FAS"][mesh_name]
        for section in ("NOEUD", "ELEME"):
            if section not in fas:
                continue
            for gname, grp in fas[section].items():
                if "GRO" not in grp:
                    continue
                nom_data = grp["GRO"]["NOM"][()]
                for row in nom_data:
                    row_bytes = bytes(row)
                    name_str = row_bytes.decode("latin-1").rstrip()
                    end_idx = len(name_str)
                    padding = row_bytes[end_idx:]
                    assert all(b == ord(" ") for b in padding), (
                        f"FAS/{section}/{gname}/GRO/NOM : "
                        f"padding must be spaces (0x20), "
                        f"found {[hex(b) for b in padding[:5]]}"
                    )


def test_empty_family_has_no_gro(tmp_path):
    """
    A family without group names must NOT have
    a GRO subgroup according to the MED spec.
    """
    from meshlane._mesh import CellBlock, Mesh

    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    cells = [CellBlock("triangle", np.array([[0, 1, 2]]))]
    mesh = Mesh(
        points, cells,
        point_data={"point_tags": np.array([0, 0, 0])},
    )
    mesh.point_tags = {-1: []}  # family with no names
    mesh.cell_tags = {}

    filename = tmp_path / "empty_family.med"
    meshlane.med.write(filename, mesh)

    with h5py.File(filename, "r") as f:
        mesh_name = list(f["ENS_MAA"].keys())[0]
        fas = f["FAS"][mesh_name]
        if "NOEUD" in fas:
            for gname, grp in fas["NOEUD"].items():
                if grp.attrs.get("NUM", 0) == -1:
                    assert "GRO" not in grp, (
                        f"Family '{gname}' without names must not "
                        f"have a GRO subgroup"
                    )


def test_family_name_too_long_raises_write_error(tmp_path):
    """
    A family name > 80 bytes must raise a WriteError.
    """
    from meshlane._mesh import CellBlock, Mesh
    from meshlane._exceptions import WriteError

    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    cells = [CellBlock("triangle", np.array([[0, 1, 2]]))]
    mesh = Mesh(
        points, cells,
        point_data={"point_tags": np.array([-1, -1, -1])},
    )
    mesh.point_tags = {-1: ["A" * 81]}  # 81 characters > 80
    mesh.cell_tags = {}

    filename = tmp_path / "toolong.med"
    with pytest.raises(WriteError, match="too long"):
        meshlane.med.write(filename, mesh)


def test_metadata_defaults_roundtrip(tmp_path):
    """A bare mesh (no metadata set) must come back with the documented
    defaults. This single test covers every default/empty branch of write():
    `getattr(..., "mesh")`, the `numpy_void_str` fallback for empty units, and
    the default description string. Those branches are otherwise executed but
    never asserted, so a broken default would go unnoticed.
    """
    filename = tmp_path / "defaults.med"
    meshlane.med.write(filename, helpers.tri_mesh)  # not mutated -> no deepcopy
 
    out = meshlane.med.read(filename)
 
    assert out.mesh_name == "mesh"
    assert out.description == "Mesh created with meshlane"
    assert out.unit_time == ""
    assert out.unit_coords == ""
 
 
@pytest.mark.parametrize(
    "attr, med_key, value",
    [
        ("description", "DES", "My simulation mesh"),
        ("unit_time", "UNT", "s"),
        ("unit_coords", "UNI", "m"),
    ],
)
def test_metadata_custom_roundtrip(tmp_path, attr, med_key, value):
    """A custom value set on the Mesh object must survive the full loop:
    write (write-side getattr picks it up) -> read (read-side reads it back)
    -> write again (the value read from disk is re-written). The final HDF5
    check proves write() did not silently fall back to its default.
    """
    f1 = tmp_path / "custom_a.med"
    f2 = tmp_path / "custom_b.med"
 
    mesh = copy.deepcopy(helpers.tri_mesh)
    setattr(mesh, attr, value)
    meshlane.med.write(f1, mesh)
 
    # Read side: the attribute is reconstructed on the Mesh object.
    out = meshlane.med.read(f1)
    assert getattr(out, attr) == value
 
    # Write side: the value must be written back, not overwritten by a default.
    meshlane.med.write(f2, out)
    with h5py.File(f2, "r") as f:
        name = next(iter(f["ENS_MAA"]))
        stored = f["ENS_MAA"][name].attrs[med_key].decode().rstrip("\x00")
        assert stored == value
 
 
def test_mesh_name_roundtrip(tmp_path):
    """The mesh name is stored as the ENS_MAA group key (not as an attribute),
    so it has its own code path and gets its own test. Setting `mesh_name` and
    letting write() build the file keeps ENS_MAA/<name> and FAS/<name>
    consistent -- unlike a manual HDF5 group rename, which would leave a
    dangling FAS group.
    """
    f1 = tmp_path / "name_a.med"
    f2 = tmp_path / "name_b.med"
 
    mesh = copy.deepcopy(helpers.tri_mesh)
    mesh.mesh_name = "my_custom_mesh"
    meshlane.med.write(f1, mesh)
 
    out = meshlane.med.read(f1)
    assert out.mesh_name == "my_custom_mesh"
 
    # The custom name must be preserved on the next write, and the default
    # name "mesh" must not reappear.
    meshlane.med.write(f2, out)
    with h5py.File(f2, "r") as f:
        assert "my_custom_mesh" in f["ENS_MAA"]
        assert "mesh" not in f["ENS_MAA"]
 
 
def test_read_strips_surrounding_whitespace(tmp_path):
    """MED files written by other tools (e.g. Salome) may pad fixed-width
    string fields. This justifies the `.strip()` cleanup in read(). We inject
    leading/trailing spaces directly into the HDF5 attribute (spaces survive
    the storage round-trip, unlike trailing NULs which both numpy and h5py
    drop) and check that read() returns the trimmed value.
    """
    filename = tmp_path / "padded.med"
    meshlane.med.write(filename, helpers.tri_mesh)
 
    with h5py.File(filename, "a") as f:
        name = next(iter(f["ENS_MAA"]))
        f["ENS_MAA"][name].attrs["DES"] = np.bytes_("   Salome mesh   ")
 
    out = meshlane.med.read(filename)
    assert out.description == "Salome mesh"


def test_point_tag_groups_attribute_exists_after_read():
    """
    After reading, the Mesh must have the point_tag_groups attribute.
    """
    this_dir = pathlib.Path(__file__).resolve().parent
    filename = this_dir / "meshes" / "med" / "cylinder.med"
    mesh_out = meshlane.med.read(filename)
    assert hasattr(mesh_out, "point_tag_groups"), (
        "The Mesh must have the point_tag_groups attribute"
    )
    assert isinstance(mesh_out.point_tag_groups, dict), (
        "point_tag_groups must be a dict"
    )


def test_cell_tag_groups_attribute_exists_after_read():
    """
    After reading, the Mesh must have the cell_tag_groups attribute.
    """
    this_dir = pathlib.Path(__file__).resolve().parent
    filename = this_dir / "meshes" / "med" / "cylinder.med"
    mesh_out = meshlane.med.read(filename)
    assert hasattr(mesh_out, "cell_tag_groups"), (
        "The Mesh must have the cell_tag_groups attribute"
    )
    assert isinstance(mesh_out.cell_tag_groups, dict), (
        "cell_tag_groups must be a dict"
    )


def test_parse_med_field_name_single():
    """
    _parse_med_field_name sur un nom sans pattern
    doit retourner (name, None, None).
    """
    from meshlane.med._med import _parse_med_field_name

    base, idx, pdt = _parse_med_field_name("Temperature")
    assert base == "Temperature"
    assert idx is None
    assert pdt is None


def test_parse_med_field_name_multi():
    """
    _parse_med_field_name doit décomposer 'Temperature[2] - 1.5'
    en ('Temperature', 2, 1.5).
    """
    from meshlane.med._med import _parse_med_field_name

    base, idx, pdt = _parse_med_field_name("Temperature[2] - 1.5")
    assert base == "Temperature"
    assert idx == 2
    assert pdt == pytest.approx(1.5)


def test_multi_timestep_grouped_under_single_hdf5_field(tmp_path):
    """
    Plusieurs timesteps d'un même champ doivent être écrits
    sous un seul groupe HDF5 dans CHA, pas comme des champs séparés.
    Sans PR16, chaque 'Temperature[i] - t' créait un groupe séparé.
    """
    from meshlane._mesh import Mesh, CellBlock

    points = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0], [1.0, 1.0, 0.0],
    ])
    cells = [CellBlock("triangle", np.array([[0, 1, 2], [1, 3, 2]]))]
    mesh = Mesh(
        points, cells,
        point_data={
            "Temperature[0] - 0.0": np.array([1.0, 2.0, 3.0, 4.0]),
            "Temperature[1] - 1.0": np.array([5.0, 6.0, 7.0, 8.0]),
        },
    )
    filename = tmp_path / "multi_ts.med"
    meshlane.med.write(filename, mesh)

    with h5py.File(filename, "r") as f:
        assert "CHA" in f, "Le groupe CHA doit exister"
        cha_keys = list(f["CHA"].keys())

        assert "Temperature" in cha_keys, (
            "Les timesteps doivent être regroupés sous 'Temperature'"
        )
        assert "Temperature[0] - 0.0" not in cha_keys, (
            "Le nom avec [0] ne doit pas être un champ séparé"
        )
        assert "Temperature[1] - 1.0" not in cha_keys, (
            "Le nom avec [1] ne doit pas être un champ séparé"
        )
        assert len(f["CHA"]["Temperature"].keys()) == 2, (
            "Il doit y avoir exactement 2 sous-groupes de timestep"
        )


def test_no_cha_group_when_no_fields(tmp_path):
    """
    Sans champs, le groupe CHA ne doit pas être créé.
    Sans PR16, CHA était toujours créé même vide.
    """
    mesh = helpers.tri_mesh
    filename = tmp_path / "no_fields.med"
    meshlane.med.write(filename, mesh)

    with h5py.File(filename, "r") as f:
        assert "CHA" not in f, (
            "Le groupe CHA ne doit pas exister quand il n'y a pas de champs"
        )


def test_multi_timestep_roundtrip_box(tmp_path):
    """
    Un fichier MED avec plusieurs timesteps doit survivre
    à un cycle read→write avec les bonnes valeurs.
    On utilise box.med qui contient déjà des champs.
    """
    this_dir = pathlib.Path(__file__).resolve().parent
    filename = this_dir / "meshes" / "med" / "box.med"
    filename_out = tmp_path / "box_roundtrip.med"

    mesh_out = meshlane.med.read(filename)
    meshlane.med.write(filename_out, mesh_out)

    mesh_rt = meshlane.med.read(filename_out)

    for key in mesh_out.point_data:
        if key == "point_tags":
            continue
        assert key in mesh_rt.point_data, (
            f"Le champ nodal '{key}' doit être présent après round-trip"
        )
        assert np.allclose(
            mesh_out.point_data[key],
            mesh_rt.point_data[key],
            equal_nan=True,
        ), f"Les valeurs du champ '{key}' doivent être identiques après round-trip"


def test_field_units_preserved_after_read(tmp_path):
    """
    Field units (UNI, UNT) must be read and stored in
    field_data['med:field_units'].
    Without PR14, these were ignored on read.
    """
    filename = tmp_path / "field_units.med"

    mesh = helpers.add_point_data(helpers.tri_mesh, 1)
    meshlane.med.write(filename, mesh)

    with h5py.File(filename, "a") as f:
        for field_name in f["CHA"]:
            f["CHA"][field_name].attrs["UNI"] = np.bytes_("Pa")
            f["CHA"][field_name].attrs["UNT"] = np.bytes_("s")

    mesh_out = meshlane.med.read(filename)

    assert "med:field_units" in mesh_out.field_data, (
        "field_data must contain 'med:field_units' after read"
    )
    for field_name, (uni, unt) in mesh_out.field_data["med:field_units"].items():
        assert uni == np.bytes_("Pa"), (
            f"UNI of field '{field_name}': expected b'Pa', got {uni}"
        )
        assert unt == np.bytes_("s"), (
            f"UNT of field '{field_name}': expected b's', got {unt}"
        )


def test_field_units_roundtrip(tmp_path):
    """
    Field units must survive a read->write cycle.
    Without PR14, write() always overwrote units with empty strings.
    """
    filename1 = tmp_path / "field_units_orig.med"
    filename2 = tmp_path / "field_units_rt.med"

    mesh = helpers.add_point_data(helpers.tri_mesh, 1)
    meshlane.med.write(filename1, mesh)

    with h5py.File(filename1, "a") as f:
        for field_name in f["CHA"]:
            f["CHA"][field_name].attrs["UNI"] = np.bytes_("MPa")
            f["CHA"][field_name].attrs["UNT"] = np.bytes_("s")

    mesh_out = meshlane.med.read(filename1)
    meshlane.med.write(filename2, mesh_out)

    with h5py.File(filename2, "r") as f:
        for field_name in f["CHA"]:
            assert f["CHA"][field_name].attrs["UNI"] == np.bytes_("MPa"), (
                f"UNI of field '{field_name}' must be preserved after round-trip"
            )
            assert f["CHA"][field_name].attrs["UNT"] == np.bytes_("s"), (
                f"UNT of field '{field_name}' must be preserved after round-trip"
            )


def test_step_metadata_preserved_after_read(tmp_path):
    """
    Timestep metadata NDT, NOR, PDT must be read and stored in
    field_data['med:step_meta'].
    Without PR14, these were ignored on read.
    """
    filename = tmp_path / "step_meta.med"

    mesh = helpers.add_point_data(helpers.tri_mesh, 1)
    meshlane.med.write(filename, mesh)

    with h5py.File(filename, "a") as f:
        for field_name in f["CHA"]:
            ts_name = list(f["CHA"][field_name].keys())[0]
            f["CHA"][field_name][ts_name].attrs["NDT"] = 7
            f["CHA"][field_name][ts_name].attrs["NOR"] = 3
            f["CHA"][field_name][ts_name].attrs["PDT"] = 2.5

    mesh_out = meshlane.med.read(filename)

    assert "med:step_meta" in mesh_out.field_data, (
        "field_data must contain 'med:step_meta' after read"
    )
    for field_name, meta_list in mesh_out.field_data["med:step_meta"].items():
        assert len(meta_list) >= 1
        meta = meta_list[0]
        assert meta["ndt"] == 7, f"NDT expected 7, got {meta['ndt']}"
        assert meta["nor"] == 3, f"NOR expected 3, got {meta['nor']}"
        assert meta["pdt"] == pytest.approx(2.5), (
            f"PDT expected 2.5, got {meta['pdt']}"
        )


def test_step_metadata_roundtrip(tmp_path):
    """
    NDT, NOR, PDT must survive a read->write cycle.
    Without PR14, write() always overwrote them with 1/1/0.0.
    """
    filename1 = tmp_path / "step_meta_orig.med"
    filename2 = tmp_path / "step_meta_rt.med"

    mesh = helpers.add_point_data(helpers.tri_mesh, 1)
    meshlane.med.write(filename1, mesh)

    with h5py.File(filename1, "a") as f:
        for field_name in f["CHA"]:
            ts_name = list(f["CHA"][field_name].keys())[0]
            f["CHA"][field_name][ts_name].attrs["NDT"] = 10
            f["CHA"][field_name][ts_name].attrs["NOR"] = 5
            f["CHA"][field_name][ts_name].attrs["PDT"] = 3.14

    mesh_out = meshlane.med.read(filename1)
    meshlane.med.write(filename2, mesh_out)

    with h5py.File(filename2, "r") as f:
        for field_name in f["CHA"]:
            ts_name = list(f["CHA"][field_name].keys())[0]
            ts = f["CHA"][field_name][ts_name]
            assert ts.attrs["NDT"] == 10, "NDT must be preserved after round-trip"
            assert ts.attrs["NOR"] == 5, "NOR must be preserved after round-trip"
            assert ts.attrs["PDT"] == pytest.approx(3.14), (
                "PDT must be preserved after round-trip"
            )


def test_metadata_latin1_roundtrip(tmp_path):
    """Non-ASCII Latin-1 metadata (µm, °C, French accents) must round-trip.
    MED stores strings as 8-bit char arrays, so Latin-1 is the supported
    encoding. Plain ASCII and Latin-1 supplements must both be preserved
    without UnicodeEncodeError on write.
    """
    filename = tmp_path / "latin1.med"
    mesh = copy.deepcopy(helpers.tri_mesh)
    mesh.unit_coords = "µm"
    mesh.unit_time = "µs"
    mesh.description = "Maillage généré par Salome"

    meshlane.med.write(filename, mesh)
    out = meshlane.med.read(filename)

    assert out.unit_coords == "µm"
    assert out.unit_time == "µs"
    assert out.description == "Maillage généré par Salome"


def test_med_multi_write_read_two_meshes(tmp_path):
    """
    write_med_multi must write two meshes and read_med_multi must
    return them with the correct number of points and cells.
    """
    from meshlane._mesh import Mesh, CellBlock

    mesh1 = Mesh(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        [CellBlock("triangle", np.array([[0, 1, 2]]))],
    )
    mesh2 = Mesh(
        np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0], [1.0, 1.0, 0.0],
        ]),
        [CellBlock("triangle", np.array([[0, 1, 2], [1, 3, 2]]))],
    )

    filename = tmp_path / "two_meshes.med"
    meshlane.med.write_med_multi(filename, [mesh1, mesh2], mesh_names=["mesh_a", "mesh_b"])

    meshes, names = meshlane.med.read_med_multi(filename)

    assert names == ["mesh_a", "mesh_b"], (
        f"Mesh names must be preserved, got {names}"
    )
    assert len(meshes[0].points) == 3, (
        "mesh_a must have 3 points"
    )
    assert len(meshes[1].points) == 4, (
        "mesh_b must have 4 points"
    )
    assert len(meshes[0].cells[0].data) == 1, (
        "mesh_a must have 1 triangle"
    )
    assert len(meshes[1].cells[0].data) == 2, (
        "mesh_b must have 2 triangles"
    )


def test_med_multi_default_mesh_names(tmp_path):
    """
    Without explicit mesh_names, meshes must be named mesh_0, mesh_1, etc.
    """
    from meshlane._mesh import Mesh, CellBlock

    mesh1 = Mesh(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        [CellBlock("triangle", np.array([[0, 1, 2]]))],
    )
    mesh2 = Mesh(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        [CellBlock("triangle", np.array([[0, 1, 2]]))],
    )

    filename = tmp_path / "default_names.med"
    meshlane.med.write_med_multi(filename, [mesh1, mesh2])

    meshes, names = meshlane.med.read_med_multi(filename)
    assert "mesh_0" in names, f"Default name 'mesh_0' expected, got {names}"
    assert "mesh_1" in names, f"Default name 'mesh_1' expected, got {names}"


def test_med_multi_field_collision_disambiguated(tmp_path):
    """
    When two meshes have a field with the same name, the HDF5 group
    must be disambiguated with @<mesh_name> suffix.
    On read-back, the field name must be the original (without @).
    """
    from meshlane._mesh import Mesh, CellBlock

    mesh1 = Mesh(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        [CellBlock("triangle", np.array([[0, 1, 2]]))],
        point_data={"pressure": np.array([1.0, 2.0, 3.0])},
    )
    mesh2 = Mesh(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        [CellBlock("triangle", np.array([[0, 1, 2]]))],
        point_data={"pressure": np.array([4.0, 5.0, 6.0])},
    )

    filename = tmp_path / "collision.med"
    meshlane.med.write_med_multi(filename, [mesh1, mesh2], mesh_names=["m1", "m2"])

    # HDF5 must use @suffix for collision
    with h5py.File(filename, "r") as f:
        cha_keys = list(f["CHA"].keys())
        assert "pressure" in cha_keys or "pressure@m1" in cha_keys, (
            f"Expected 'pressure' or 'pressure@m1' in CHA, got {cha_keys}"
        )
        assert "pressure@m2" in cha_keys, (
            f"Expected 'pressure@m2' in CHA, got {cha_keys}"
        )

    # Read-back must restore original field name without @
    meshes, names = meshlane.med.read_med_multi(filename)
    assert "pressure" in meshes[0].point_data, (
        "Field 'pressure' must be restored without @ suffix on read"
    )
    assert "pressure" in meshes[1].point_data, (
        "Field 'pressure' must be restored without @ suffix on read"
    )
    assert not any("@" in k for k in meshes[0].point_data), (
        "No @ suffix must appear in point_data keys after read"
    )
    assert not any("@" in k for k in meshes[1].point_data), (
        "No @ suffix must appear in point_data keys after read"
    )


def test_med_multi_no_field_collision(tmp_path):
    """
    When two meshes have different field names, no @ suffix must be used.
    """
    from meshlane._mesh import Mesh, CellBlock

    mesh1 = Mesh(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        [CellBlock("triangle", np.array([[0, 1, 2]]))],
        point_data={"temperature": np.array([1.0, 2.0, 3.0])},
    )
    mesh2 = Mesh(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        [CellBlock("triangle", np.array([[0, 1, 2]]))],
        point_data={"pressure": np.array([4.0, 5.0, 6.0])},
    )

    filename = tmp_path / "no_collision.med"
    meshlane.med.write_med_multi(filename, [mesh1, mesh2], mesh_names=["m1", "m2"])

    with h5py.File(filename, "r") as f:
        cha_keys = list(f["CHA"].keys())
        assert "temperature" in cha_keys, (
            "Field 'temperature' must not be renamed when no collision"
        )
        assert "pressure" in cha_keys, (
            "Field 'pressure' must not be renamed when no collision"
        )
        assert not any("@" in k for k in cha_keys), (
            f"No @ suffix expected when no collision, got {cha_keys}"
        )


def test_med_multi_points_preserved(tmp_path):
    """
    Point coordinates must be exactly preserved after a write/read round-trip.
    """
    from meshlane._mesh import Mesh, CellBlock

    pts1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    pts2 = np.array([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2.0, 1.0, 0.0]])

    mesh1 = Mesh(pts1, [CellBlock("triangle", np.array([[0, 1, 2]]))])
    mesh2 = Mesh(pts2, [CellBlock("triangle", np.array([[0, 1, 2]]))])

    filename = tmp_path / "points.med"
    meshlane.med.write_med_multi(filename, [mesh1, mesh2], mesh_names=["m1", "m2"])

    meshes, _ = meshlane.med.read_med_multi(filename)
    assert np.allclose(meshes[0].points, pts1), (
        "Points of mesh1 must be preserved after round-trip"
    )
    assert np.allclose(meshes[1].points, pts2), (
        "Points of mesh2 must be preserved after round-trip"
    )


def test_med_multi_hdf5_structure(tmp_path):
    """
    The HDF5 file must contain ENS_MAA with all mesh names
    and FAS with one group per mesh.
    """
    from meshlane._mesh import Mesh, CellBlock

    mesh1 = Mesh(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        [CellBlock("triangle", np.array([[0, 1, 2]]))],
    )
    mesh2 = Mesh(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        [CellBlock("triangle", np.array([[0, 1, 2]]))],
    )

    filename = tmp_path / "structure.med"
    meshlane.med.write_med_multi(filename, [mesh1, mesh2], mesh_names=["alpha", "beta"])

    with h5py.File(filename, "r") as f:
        assert "ENS_MAA" in f, "ENS_MAA must exist"
        assert "alpha" in f["ENS_MAA"], "alpha must be in ENS_MAA"
        assert "beta" in f["ENS_MAA"], "beta must be in ENS_MAA"
        assert "FAS" in f, "FAS must exist"
        assert "alpha" in f["FAS"], "alpha must be in FAS"
        assert "beta" in f["FAS"], "beta must be in FAS"
        assert "INFOS_GENERALES" in f, "INFOS_GENERALES must exist"


# Reference cells in meshio/VTK ordering + the MED (MEDCoupling INTERP_KERNEL
# CellModel.cxx) face definitions for each 3D type.
_MED_ORIENT_REF = {
    "tetra": (
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float),
        [[0, 1, 2], [0, 3, 1], [1, 3, 2], [2, 3, 0]],
    ),
    "pyramid": (
        np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 1]], float),
        [[0, 1, 2, 3], [0, 4, 1], [1, 4, 2], [2, 4, 3], [3, 4, 0]],
    ),
    "wedge": (
        np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1]], float
        ),
        [[0, 1, 2], [3, 5, 4], [0, 3, 4, 1], [1, 4, 5, 2], [2, 5, 3, 0]],
    ),
    "hexahedron": (
        np.array(
            [
                [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
            ],
            float,
        ),
        [[0, 1, 2, 3], [4, 7, 6, 5], [0, 4, 5, 1],
         [1, 5, 6, 2], [2, 6, 7, 3], [3, 7, 4, 0]],
    ),
}


def _all_med_faces_outward(pts, perm, med_faces):
    """True iff, after applying ``perm``, every MED-defined face points outward."""
    h = pts[perm]
    centroid = h.mean(axis=0)
    for f in med_faces:
        fp = h[f]
        normal = np.cross(fp[1] - fp[0], fp[2] - fp[0])
        if np.dot(normal, fp.mean(axis=0) - centroid) <= 0:
            return False
    return True


def test_med_node_perm_matches_medcoupling_faces():
    """The meshio<->MED 3D node permutations must produce valid MED cells: after
    permutation, every face defined by MEDCoupling's INTERP_KERNEL cell model
    (CellModel.cxx) must point outward. Pins the ordering to the authoritative
    MED source, independent of any reference .med file."""
    from meshlane.med._med import _med_node_perm

    for cell_type, (pts, med_faces) in _MED_ORIENT_REF.items():
        assert _all_med_faces_outward(pts, _med_node_perm[cell_type], med_faces), (
            f"{cell_type}: MED faces not all outward after permutation"
        )


def test_identity_perm_is_not_med_orientation():
    """Negative control: the identity permutation must NOT yield valid MED cells.
    Guards against a future change silently dropping a permutation to
    [0, 1, 2, ...] (meshio order), which would still be wrong for MED."""
    for cell_type, (pts, med_faces) in _MED_ORIENT_REF.items():
        identity = list(range(len(pts)))
        assert not _all_med_faces_outward(pts, identity, med_faces), (
            f"{cell_type}: identity permutation unexpectedly passed the MED "
            "outward-face check (the check is not discriminating)"
        )

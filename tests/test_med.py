import pathlib

import numpy as np
import pytest

import meshio
from meshio.med._med import numpy_to_med_type
from meshio.med._med41 import (
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
    helpers.write_read(tmp_path, meshio.med.write, meshio.med.read, mesh, 1.0e-15)


def test_generic_io(tmp_path):
    helpers.generic_io(tmp_path / "test.med")
    # With additional, insignificant suffix:
    helpers.generic_io(tmp_path / "test.0.med")


def test_reference_file_with_mixed_cells(tmp_path):
    this_dir = pathlib.Path(__file__).resolve().parent
    filename = this_dir / "meshes" / "med" / "cylinder.med"
    mesh = meshio.read(filename)

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

    helpers.write_read(tmp_path, meshio.med.write, meshio.med.read, mesh, 1.0e-15)


def test_reference_file_with_point_cell_data(tmp_path):
    this_dir = pathlib.Path(__file__).resolve().parent
    filename = this_dir / "meshes" / "med" / "box.med"

    mesh = meshio.read(filename)

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

    helpers.write_read(tmp_path, meshio.med.write, meshio.med.read, mesh, 1.0e-15)


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

        # 3 points, 1 triangle — pas de FAS
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
    mesh = meshio.med.read(filename)
    assert len(mesh.points) == 3
    assert len(mesh.cells) == 1
    assert mesh.cells[0].type == "triangle"


def test_read_med_without_gro(tmp_path):
    """Une famille sans sous-groupe GRO ne doit pas crasher."""
    filename = tmp_path / "no_gro.med"

    # Écrire un mesh normal puis modifier le FAS
    mesh = helpers.tri_mesh
    meshio.med.write(filename, mesh)

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

    mesh_out = meshio.med.read(filename)
    assert len(mesh_out.points) > 0
    assert len(mesh_out.cells) > 0


def test_write_multi_blocks_same_type_with_cell_data(tmp_path):
    """Multiple blocks of the same type with cell_data must be merged."""
    from meshio._mesh import CellBlock

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

    mesh = meshio.Mesh(points, cells, cell_data=cell_data)
    filename = tmp_path / "multi_blocks.med"

    meshio.med.write(filename, mesh)

    # Re-read: triangles are merged into 1 block
    mesh_out = meshio.med.read(filename)
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

    from meshio._mesh import CellBlock

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

    mesh = meshio.Mesh(points, cells)
    meshio.med.write(filename, mesh)

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
    mesh_out = meshio.med.read(filename)
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
    after a meshio write.
    """
    filename = tmp_path / f"test_roundtrip_{dtype.__name__}.med"

    mesh = helpers.add_point_data(helpers.tri_mesh, 1)
    for key in mesh.point_data:
        mesh.point_data[key] = mesh.point_data[key].astype(dtype)

    meshio.med.write(filename, mesh)

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
    meshio.med.write(filename, mesh, med_version=med_version)

    with h5py.File(filename, "r") as f:
        info = f["INFOS_GENERALES"]
        assert int(info.attrs["MAJ"]) == expected[0]
        assert int(info.attrs["MIN"]) == expected[1]
        assert int(info.attrs["REL"]) == expected[2]


def test_med_version_default(tmp_path):
    """Default MED version should be 4.1.0."""
    filename = tmp_path / "test_default.med"
    mesh = helpers.tri_mesh
    meshio.med.write(filename, mesh)

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
    """After a full meshio write, bitmask attributes must exist in CHA fields."""
    filename = tmp_path / "test_bitmask_full.med"

    mesh = helpers.add_point_data(helpers.tri_mesh, 1)
    meshio.med.write(filename, mesh)

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

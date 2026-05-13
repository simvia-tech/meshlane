import pathlib

import numpy as np
import pytest

import meshio

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


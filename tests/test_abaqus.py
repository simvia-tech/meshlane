import pathlib

import numpy as np
import pytest

import meshlane

from . import helpers


@pytest.mark.parametrize(
    "mesh",
    [
        helpers.empty_mesh,
        helpers.tri_mesh,
        helpers.triangle6_mesh,
        helpers.quad_mesh,
        helpers.quad8_mesh,
        helpers.tri_quad_mesh,
        helpers.tet_mesh,
        helpers.tet10_mesh,
        helpers.hex_mesh,
        helpers.hex20_mesh,
    ],
)
def test(mesh, tmp_path):
    helpers.write_read(tmp_path, meshlane.abaqus.write, meshlane.abaqus.read, mesh, 1.0e-15)


@pytest.mark.parametrize(
    "filename, ref_sum, ref_num_cells, ref_num_cell_sets",
    [
        ("UUea.inp", 4950.0, 50, 10),
        ("nle1xf3c.inp", 32.215275528, 12, 3),
        ("element_elset.inp", 6.0, 2, 3),
        ("wInclude_main.inp", 1.5, 2, 0),
    ],
)
def test_reference_file(filename, ref_sum, ref_num_cells, ref_num_cell_sets):
    this_dir = pathlib.Path(__file__).resolve().parent
    filename = this_dir / "meshes" / "abaqus" / filename

    mesh = meshlane.read(filename)

    assert np.isclose(np.sum(mesh.points), ref_sum)
    assert sum(len(cells.data) for cells in mesh.cells) == ref_num_cells
    assert len(mesh.cell_sets) == ref_num_cell_sets


def test_elset(tmp_path):
    points = np.array(
        [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 0.5, 0.0], [0.0, 0.5, 0.0]]
    )
    cells = [
        ("triangle", np.array([[0, 1, 2]])),
        ("triangle", np.array([[0, 1, 3]])),
    ]
    cell_sets = {
        "right": [np.array([0]), np.array([])],
        "left": [np.array([]), np.array([1])],
    }
    mesh_ref = meshlane.Mesh(points, cells, cell_sets=cell_sets)

    filepath = tmp_path / "test.inp"
    meshlane.abaqus.write(filepath, mesh_ref)
    mesh = meshlane.abaqus.read(filepath)

    assert np.allclose(mesh_ref.points, mesh.points)

    assert len(mesh_ref.cells) == len(mesh.cells)
    for ic, cell in enumerate(mesh_ref.cells):
        assert cell.type == mesh.cells[ic].type
        assert np.allclose(cell.data, mesh.cells[ic].data)

    assert sorted(mesh_ref.cell_sets.keys()) == sorted(mesh.cell_sets.keys())
    for k, v in mesh_ref.cell_sets.items():
        for ic in range(len(mesh_ref.cells)):
            assert np.allclose(v[ic], mesh.cell_sets[k][ic])


def _nodes(n):
    return "*NODE\n" + "".join(
        f"{i}, {float(i)}, {float(i % 3)}, {float(i % 2)}\n" for i in range(1, n + 1)
    )


def _write_deck(path, n_nodes, body):
    path.write_text(_nodes(n_nodes) + body)
    return path


def test_thermal_and_gasket_types(tmp_path):
    # thermal (DC3D*) and gasket (GK3D8) map to their geometry
    body = (
        "*ELEMENT, TYPE=DC3D10, ELSET=T\n1, 1,2,3,4,5,6,7,8,9,10\n"
        "*ELEMENT, TYPE=DC3D8, ELSET=H\n2, 1,2,3,4,5,6,7,8\n"
        "*ELEMENT, TYPE=GK3D8, ELSET=G\n3, 1,2,3,4,5,6,7,8\n"
    )
    f = _write_deck(tmp_path / "t.inp", 10, body)
    mesh = meshlane.abaqus.read(f)
    counts = {}
    for c in mesh.cells:
        counts[c.type] = counts.get(c.type, 0) + len(c.data)
    assert counts == {"tetra10": 1, "hexahedron": 2}


def test_skip_non_mesh_elements(tmp_path):
    # connectors/springs/masses are not cells and must be dropped (no error)
    body = (
        "*ELEMENT, TYPE=C3D4, ELSET=S\n1, 1,2,3,4\n"
        "*ELEMENT, TYPE=SPRING2, ELSET=SP\n2, 1,2\n"
        "*ELEMENT, TYPE=MASS, ELSET=M\n3, 1\n"
        "*ELEMENT, TYPE=CONN3D2, ELSET=C\n4, 1,2\n"
        "*ELEMENT, TYPE=DCOUP3D, ELSET=D\n5, 1\n"
    )
    f = _write_deck(tmp_path / "s.inp", 4, body)
    mesh = meshlane.abaqus.read(f)
    assert {c.type for c in mesh.cells} == {"tetra"}


def test_unknown_type_warns_and_skips(tmp_path, capsys):
    # an unrecognized type is skipped with a warning, the rest still reads
    body = (
        "*ELEMENT, TYPE=C3D4, ELSET=S\n1, 1,2,3,4\n"
        "*ELEMENT, TYPE=FOOBAR9, ELSET=X\n2, 1,2,3,4\n"
    )
    f = _write_deck(tmp_path / "u.inp", 4, body)
    mesh = meshlane.abaqus.read(f)
    assert {c.type for c in mesh.cells} == {"tetra"}
    assert "FOOBAR9" in capsys.readouterr().err


def test_assembly_instances(tmp_path):
    # a part instanced three times: identity, translated, and rotated. Each
    # instance keeps its own node numbering, so the copies must not collide.
    deck = """*Part, name=P1
*Node
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
3, 0.0, 1.0, 0.0
4, 0.0, 0.0, 1.0
*Element, type=C3D4
1, 1, 2, 3, 4
*Nset, nset=CORNER
1,
*End Part
*Assembly, name=Assembly
*Instance, name=I1, part=P1
*End Instance
*Instance, name=I2, part=P1
10.0, 0.0, 0.0
*End Instance
*Instance, name=I3, part=P1
0.0, 0.0, 0.0
0.0,0.0,0.0, 0.0,0.0,1.0, 90.0
*End Instance
*Nset, nset=TOP, instance=I2
2
*End Assembly
"""
    f = tmp_path / "asm.inp"
    f.write_text(deck)
    m = meshlane.read(f)
    assert len(m.points) == 12                       # 3 instances x 4 nodes
    assert sum(len(c.data) for c in m.cells) == 3    # 3 tetra
    assert np.allclose(m.points[4], [10.0, 0.0, 0.0])            # I2 translated
    assert np.allclose(m.points[9], [0.0, 1.0, 0.0], atol=1e-9)  # I3 (1,0,0)->(0,1,0)
    assert len(m.point_sets["CORNER"]) == 3          # part set, unioned over 3
    assert np.allclose(m.points[m.point_sets["TOP"][0]], [11.0, 0.0, 0.0])

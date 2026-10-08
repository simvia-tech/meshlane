import pathlib

import numpy as np
import pytest

import meshlane

from . import helpers

test_set = [
    # helpers.empty_mesh,
    helpers.tri_mesh_2d,
    helpers.tet_mesh,
    helpers.hex_mesh,
]
this_dir = pathlib.Path(__file__).resolve().parent


@pytest.mark.parametrize("mesh", test_set)
def test(mesh, tmp_path):
    helpers.write_read(tmp_path, meshlane.su2.write, meshlane.su2.read, mesh, 1.0e-15)


@pytest.mark.parametrize(
    "filename, ref_num_cells, ref_num_points,ref_num_unique_tags,sum_tags",
    [("square.su2", 16, 9, 4, 20), ("mixgrid.su2", 30, 16, 6, 62)],
)
def test_structured(
    filename, ref_num_cells, ref_num_points, ref_num_unique_tags, sum_tags
):
    filename = this_dir / "meshes" / "su2" / filename

    mesh = meshlane.read(filename)

    assert sum(len(block.data) for block in mesh.cells) == ref_num_cells
    assert len(mesh.points) == ref_num_points

    all_tags = np.concatenate([tags for tags in mesh.cell_data["su2:tag"]])

    assert sum(all_tags) == sum_tags

    all_unique_tags = np.unique(all_tags)

    assert len(all_unique_tags) == ref_num_unique_tags + 1


SQUARE = this_dir / "meshes" / "su2" / "square.su2"


def marker_names(path):
    return [
        line.split("=", 1)[1].strip()
        for line in pathlib.Path(path).read_text().splitlines()
        if line.startswith("MARKER_TAG")
    ]


def test_read_marker_names():
    mesh = meshlane.read(SQUARE)
    assert list(mesh.cell_sets) == ["lower", "right", "upper", "left"]
    lines = [b.type for b in mesh.cells].index("line")
    assert [len(s[lines]) for s in mesh.cell_sets.values()] == [2, 2, 2, 2]


def test_round_trip_with_markers(tmp_path):
    mesh = meshlane.read(SQUARE)
    meshlane.write(tmp_path / "square.su2", mesh)
    assert marker_names(tmp_path / "square.su2") == ["lower", "right", "upper", "left"]
    mesh2 = meshlane.read(tmp_path / "square.su2")
    for name, members in mesh.cell_sets.items():
        for block, m1, m2 in zip(mesh.cells, members, mesh2.cell_sets[name]):
            assert np.array_equal(
                np.sort(block.data[m1], axis=1),
                np.sort(
                    mesh2.cells[[b.type for b in mesh2.cells].index(block.type)].data[
                        m2
                    ],
                    axis=1,
                ),
            )


def test_write_markers_from_cell_sets_3d(tmp_path):
    points = helpers.hex_mesh.points
    mesh = meshlane.Mesh(
        points,
        [
            ("hexahedron", [[0, 1, 2, 3, 4, 5, 6, 7]]),
            ("quad", [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4]]),
        ],
        cell_sets={
            "bottom wall": [np.array([], dtype=int), np.array([0])],
            "top": [np.array([], dtype=int), np.array([1])],
        },
    )
    meshlane.write(tmp_path / "hex.su2", mesh)
    text = (tmp_path / "hex.su2").read_text()
    assert "NDIME= 3" in text
    # the ungrouped quad goes to a default marker; spaces are not allowed
    assert marker_names(tmp_path / "hex.su2") == ["bottom_wall", "top", "boundary"]
    mesh2 = meshlane.read(tmp_path / "hex.su2")
    assert [len(s[1]) for s in mesh2.cell_sets.values()] == [1, 1, 1]


def test_write_markers_from_cell_tags(tmp_path):
    # the representation the MED and OpenFOAM readers use
    mesh = meshlane.read(SQUARE)
    mesh.cell_sets = {}
    lines = [b.type for b in mesh.cells].index("line")
    tags = mesh.cell_data.pop("su2:tag")
    mesh.cell_data["cell_tags"] = [-t for t in tags]
    mesh.cell_tags = {-1: ["lower"], -2: ["right"], -3: ["upper"], -4: ["left"]}
    meshlane.write(tmp_path / "square.su2", mesh)
    assert marker_names(tmp_path / "square.su2") == ["lower", "right", "upper", "left"]
    assert len(tags[lines]) == 8


def test_write_2d_mesh_with_z_coordinate(tmp_path):
    # typical Gmsh/Salome 2D mesh: 3D points with z = 0
    mesh = meshlane.Mesh(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        [("triangle", [[0, 1, 2], [0, 2, 3]]), ("line", [[0, 1], [1, 2]])],
        cell_sets={"wall": [np.array([], dtype=int), np.array([0, 1])]},
    )
    meshlane.write(tmp_path / "tri.su2", mesh)
    text = (tmp_path / "tri.su2").read_text()
    assert "NDIME= 2" in text
    assert "NELEM= 2" in text
    mesh2 = meshlane.read(tmp_path / "tri.su2")
    assert mesh2.points.shape == (4, 2)
    assert [len(x) for x in mesh2.cell_sets["wall"]] == [0, 2]


def test_write_non_planar_surface_is_rejected(tmp_path):
    mesh = meshlane.Mesh(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.5], [1.0, 1.0, 0.0]],
        [("triangle", [[0, 1, 2]])],
    )
    with pytest.raises(meshlane.WriteError):
        meshlane.write(tmp_path / "surf.su2", mesh)


def test_read_crlf(tmp_path):
    path = tmp_path / "square_crlf.su2"
    path.write_bytes(SQUARE.read_bytes().replace(b"\n", b"\r\n"))
    mesh = meshlane.read(path)
    assert len(mesh.points) == 9
    assert list(mesh.cell_sets) == ["lower", "right", "upper", "left"]


def test_read_spaces_around_equal_sign(tmp_path):
    path = tmp_path / "spaces.su2"
    text = (
        SQUARE.read_text()
        .replace("NDIME= 2", "NDIME = 2")
        .replace("NELEM= 8", "NELEM =8")
    )
    path.write_text(text)
    mesh = meshlane.read(path)
    assert sum(len(b) for b in mesh.cells) == 16


def test_read_multizone_is_rejected(tmp_path):
    path = tmp_path / "multizone.su2"
    body = SQUARE.read_text()
    path.write_text("NZONE= 2\nIZONE= 1\n" + body + "IZONE= 2\n" + body)
    with pytest.raises(meshlane.ReadError, match="read_multi"):
        meshlane.su2.read(path)


def test_write_warns_type_name(tmp_path, capsys):
    mesh = meshlane.Mesh(
        helpers.tet10_mesh.points,
        helpers.tet10_mesh.cells + [meshlane.CellBlock("tetra", [[0, 1, 2, 3]])],
    )
    meshlane.write(tmp_path / "t.su2", mesh)
    assert "tetra10" in capsys.readouterr().err


# sections SU2 writes after the markers (periodicity, FFD boxes), abridged
EXTRA = """NPERIODIC= 1
PERIODIC_INDEX= 0
0.000000000000000e+00 0.000000000000000e+00 0.000000000000000e+00
FFD_NBOX= 1
FFD_NLEVEL= 1
FFD_TAG= BOX
FFD_LEVEL= 0
FFD_DEGREE_I= 1
FFD_DEGREE_J= 1
FFD_CORNER_POINTS= 4
0 0
1 0
1 1
0 1
FFD_CONTROL_POINTS= 0
FFD_SURFACE_POINTS= 1
lower 1 0.5 0.0
"""


def test_extra_sections_are_kept(tmp_path, capsys):
    path = tmp_path / "ffd.su2"
    path.write_text(SQUARE.read_text() + EXTRA)
    mesh = meshlane.read(path)
    assert "could not parse" not in capsys.readouterr().err
    assert mesh.su2_extra == EXTRA

    meshlane.write(tmp_path / "out.su2", mesh)
    assert (tmp_path / "out.su2").read_text().endswith(EXTRA)
    assert meshlane.read(tmp_path / "out.su2").su2_extra == EXTRA


def test_extra_sections_dropped_when_points_change(tmp_path, capsys):
    path = tmp_path / "ffd.su2"
    path.write_text(SQUARE.read_text() + EXTRA)
    mesh = meshlane.read(path)
    # a point and a triangle added: the FFD point indices may no longer hold
    mesh.points = np.vstack([mesh.points, [[1.5, 1.5]]])
    mesh.cells.append(meshlane.CellBlock("triangle", np.array([[5, 9, 8]])))
    mesh.cell_data = {}
    mesh.cell_sets = {}
    meshlane.write(tmp_path / "out.su2", mesh)
    assert "FFD_NBOX" not in (tmp_path / "out.su2").read_text()
    assert "FFD" in capsys.readouterr().err


def test_quadratic_cells_use_corner_nodes(tmp_path, capsys):
    meshlane.write(tmp_path / "t.su2", helpers.tet10_mesh)
    assert "tetra10" in capsys.readouterr().err
    mesh = meshlane.read(tmp_path / "t.su2")
    assert [(b.type, len(b)) for b in mesh.cells] == [("tetra", 1)]
    # the mid-edge nodes are unused, hence dropped
    corners = helpers.tet10_mesh.points[helpers.tet10_mesh.cells[0].data[0, :4]]
    assert np.array_equal(np.sort(mesh.points, axis=0), np.sort(corners, axis=0))


def test_cells_of_another_dimension_are_reported(tmp_path, capsys):
    mesh = meshlane.Mesh(
        helpers.hex_mesh.points,
        [("hexahedron", [[0, 1, 2, 3, 4, 5, 6, 7]]), ("line", [[4, 5], [5, 6]])],
        cell_sets={"edge": [np.array([], dtype=int), np.array([0, 1])]},
    )
    meshlane.write(tmp_path / "hex.su2", mesh)
    err = capsys.readouterr().err
    assert "line" in err and "edge" in err


def test_marker_name_collision_is_reported(tmp_path, capsys):
    mesh = meshlane.Mesh(
        helpers.hex_mesh.points,
        [
            ("hexahedron", [[0, 1, 2, 3, 4, 5, 6, 7]]),
            ("quad", [[0, 3, 2, 1], [4, 5, 6, 7]]),
        ],
        cell_sets={
            "a b": [np.array([], dtype=int), np.array([0])],
            "a_b": [np.array([], dtype=int), np.array([1])],
        },
    )
    meshlane.write(tmp_path / "hex.su2", mesh)
    assert "a_b" in capsys.readouterr().err
    assert marker_names(tmp_path / "hex.su2") == ["a_b"]


def two_zones(tmp_path):
    """A multi-zone file: the square, then the square shifted with other markers."""
    body = SQUARE.read_text()
    second = body
    for a, b in [("lower", "lower2"), ("right", "right2"), ("upper", "upper2")]:
        second = second.replace(f"MARKER_TAG= {a}", f"MARKER_TAG= {b}")
    path = tmp_path / "zones.su2"
    path.write_text("NZONE= 2\n\nIZONE= 1\n" + body + "\nIZONE= 2\n" + second)
    return path


def test_read_multi(tmp_path):
    meshes, names = meshlane.su2.read_multi(two_zones(tmp_path))
    assert names == ["zone_1", "zone_2"]
    assert [len(m.points) for m in meshes] == [9, 9]
    assert list(meshes[0].cell_sets) == ["lower", "right", "upper", "left"]
    assert list(meshes[1].cell_sets) == ["lower2", "right2", "upper2", "left"]


def test_read_multi_single_zone():
    meshes, names = meshlane.su2.read_multi(SQUARE)
    assert names == ["zone_1"] and len(meshes) == 1
    assert list(meshes[0].cell_sets) == ["lower", "right", "upper", "left"]


def test_write_multi_round_trip(tmp_path):
    meshes, _ = meshlane.su2.read_multi(two_zones(tmp_path))
    out = tmp_path / "out.su2"
    meshlane.su2.write_multi(out, meshes, mesh_names=["fluid", "solid"])
    text = out.read_text()
    assert text.startswith("NZONE= 2\nIZONE= 1\n")
    assert "IZONE= 2" in text
    meshes2, names = meshlane.su2.read_multi(out)
    assert names == ["fluid", "solid"]
    for m1, m2 in zip(meshes, meshes2):
        assert np.array_equal(m1.points, m2.points)
        assert list(m1.cell_sets) == list(m2.cell_sets)
        for b1, b2 in zip(m1.cells, m2.cells):
            assert b1.type == b2.type and np.array_equal(b1.data, b2.data)


def multizone_through_med(tmp_path):
    pytest.importorskip("h5py")
    med = tmp_path / "zones.med"
    su2 = tmp_path / "back.su2"
    meshlane._cli.main(["convert", str(two_zones(tmp_path)), str(med)])
    assert len(meshlane.med.read_med_multi(med)[0]) == 2
    meshlane._cli.main(["convert", str(med), str(su2)])
    return meshlane.su2.read_multi(su2)


def test_cli_multizone_to_med_and_back(tmp_path):
    meshes, names = multizone_through_med(tmp_path)
    assert names == ["zone_1", "zone_2"]
    original, _ = meshlane.su2.read_multi(two_zones(tmp_path))
    for m1, m2 in zip(original, meshes):
        assert np.allclose(m1.points, m2.points)
        assert sorted((b.type, len(b)) for b in m1.cells) == sorted(
            (b.type, len(b)) for b in m2.cells
        )


@pytest.mark.xfail(
    strict=True,
    reason="write_med_multi does not turn cell_sets into MED families (MED audit)",
)
def test_cli_multizone_to_med_and_back_keeps_markers(tmp_path):
    meshes, _ = multizone_through_med(tmp_path)
    assert [sorted(m.cell_sets) for m in meshes] == [
        sorted(["lower", "right", "upper", "left"]),
        sorted(["lower2", "right2", "upper2", "left"]),
    ]


def test_cli_multizone_to_single_mesh_format_fails(tmp_path):
    with pytest.raises(SystemExit):
        meshlane._cli.main(
            ["convert", str(two_zones(tmp_path)), str(tmp_path / "zones.vtu")]
        )

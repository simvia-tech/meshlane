import numpy as np
import pytest

import meshlane

from . import helpers


@pytest.mark.parametrize(
    "mesh",
    [
        # helpers.empty_mesh,
        helpers.line_mesh,
        helpers.tri_mesh,
        helpers.tri_mesh_2d,
        helpers.tri_quad_mesh,
        helpers.quad_mesh,
        helpers.quad8_mesh,
        helpers.triangle6_mesh,
        helpers.tet_mesh,
        helpers.tet10_mesh,
        helpers.hex_mesh,
        helpers.hex20_mesh,
        helpers.wedge_mesh,
        helpers.pyramid_mesh,
        helpers.polygon_mesh,
        helpers.polygon_mesh_one_cell,
        helpers.polygon2_mesh,
    ],
)
def test(mesh, tmp_path):
    helpers.write_read(tmp_path, meshlane.cgns.write, meshlane.cgns.read, mesh, 1.0e-15)


def test_polyhedron_faces_not_duplicated_on_roundtrip(tmp_path):
    """Polyhedra must reference the polygon faces already written, not duplicate
    them into a second NGON pool: writing then reading a mesh whose polyhedron
    faces are all present in a polygon block must not introduce phantom polygon
    cells (regression for the 2583 -> 6875 polygon inflation on round-trip)."""
    points = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    faces = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
    mesh = meshlane.Mesh(
        points,
        [
            ("polygon", faces),
            ("polyhedron4", [faces]),
        ],
    )

    p = tmp_path / "poly.cgns"
    meshlane.cgns.write(p, mesh)
    back = meshlane.cgns.read(p)

    n_polygons = sum(len(cb.data) for cb in back.cells if cb.type == "polygon")
    n_polyhedra = sum(
        len(cb.data) for cb in back.cells if cb.type.startswith("polyhedron")
    )
    assert n_polygons == 4, f"expected 4 polygons, got {n_polygons} (faces duplicated)"
    assert n_polyhedra == 1, f"expected 1 polyhedron, got {n_polyhedra}"


def test_polyhedron_face_orientation_roundtrip(tmp_path):
    """A face shared by two polyhedra must come back with opposite winding in
    the two cells. CGNS encodes that as the sign of the NFACE_n reference; the
    reader must honour it rather than returning both cells the canonical order.
    Regression for F2 (21% of face references flipped on particles_example)."""
    points = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ]
    # Two tetrahedra sharing face (0, 1, 2), listed in opposite windings.
    cell_a = [[0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3]]
    cell_b = [[2, 1, 0], [0, 1, 4], [1, 2, 4], [0, 2, 4]]
    mesh = meshlane.Mesh(points, [("polyhedron4", [cell_a, cell_b])])

    p = tmp_path / "orient.cgns"
    meshlane.cgns.write(p, mesh)
    back = meshlane.cgns.read(p)

    blocks = [cb for cb in back.cells if cb.type.startswith("polyhedron")]
    assert len(blocks) == 1
    got_a, got_b = blocks[0].data

    for expected, got, label in ((cell_a, got_a, "A"), (cell_b, got_b, "B")):
        assert len(got) == len(expected), f"cell {label}: face count changed"
        for e_face, g_face in zip(expected, got):
            assert list(g_face) == list(e_face), (
                f"cell {label}: face winding not preserved: "
                f"{list(g_face)} != {list(e_face)}"
            )

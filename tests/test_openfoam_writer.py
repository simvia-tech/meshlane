"""
Tests for the OpenFOAM polyMesh writer.
"""

from __future__ import annotations

import numpy as np
import pytest

import meshlane
from meshlane import WriteError
from meshlane.openfoam._openfoam import (
    _parse_boundary,
    _read_faces,
    _read_foam_lines,
    _read_int_list,
    _read_points,
)

from . import helpers

# Two unit cubes side by side along x: cell 0 in x < 1, cell 1 in x > 1
TWO_HEX_POINTS = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
    [0.0, 1.0, 1.0],
    [2.0, 0.0, 0.0],
    [2.0, 1.0, 0.0],
    [2.0, 0.0, 1.0],
    [2.0, 1.0, 1.0],
]
TWO_HEX_CELLS = [[0, 1, 2, 3, 4, 5, 6, 7], [1, 8, 9, 2, 5, 10, 11, 6]]


def two_hex_mesh(with_patches=False):
    cells = [("hexahedron", TWO_HEX_CELLS)]
    cell_sets = {}
    if with_patches:
        quads = [
            [0, 4, 7, 3],  # inlet, x = 0
            [8, 9, 11, 10],  # outlet, x = 2
            [0, 1, 5, 4],  # walls from here on
            [1, 8, 10, 5],
            [3, 7, 6, 2],
            [2, 6, 11, 9],
            [0, 3, 2, 1],
            [1, 2, 9, 8],
            [4, 5, 6, 7],
            [5, 10, 11, 6],
        ]
        cells.append(("quad", quads))
        cell_sets = {
            "inlet": [np.array([], dtype=int), np.array([0])],
            "outlet": [np.array([], dtype=int), np.array([1])],
            "walls": [np.array([], dtype=int), np.arange(2, 10)],
        }
    return meshlane.Mesh(TWO_HEX_POINTS, cells, cell_sets=cell_sets)


def mixed_mesh():
    """A hexahedron with a wedge, a pyramid and a tetrahedron glued to it."""
    points = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
        [2.0, 0.0, 0.0],
        [2.0, 0.0, 1.0],
        [0.5, 0.5, 1.5],
        [1.3, 0.3, 1.5],
    ]
    return meshlane.Mesh(
        points,
        [
            ("hexahedron", [[0, 1, 2, 3, 4, 5, 6, 7]]),
            ("tetra", [[5, 9, 6, 11]]),
            ("wedge", [[1, 8, 2, 5, 9, 6]]),
            ("pyramid", [[4, 5, 6, 7, 10]]),
        ],
    )


def read_polymesh(poly):
    points = _read_points(poly / "points")
    faces = [list(f) for f in _read_faces(poly / "faces")]
    owner = _read_int_list(poly / "owner")
    neighbour = _read_int_list(poly / "neighbour")
    boundary = _parse_boundary(_read_foam_lines(poly / "boundary"))
    return points, faces, owner, neighbour, boundary


def area_vector(points, face):
    p = points[face]
    c = p.mean(axis=0)
    return 0.5 * sum(np.cross(p[k] - c, p[(k + 1) % len(p)] - c) for k in range(len(p)))


def check_polymesh(poly):
    """The conditions OpenFOAM's checkMesh verifies on the topology."""
    points, faces, owner, neighbour, boundary = read_polymesh(poly)
    n_faces = len(faces)
    n_internal = len(neighbour)
    n_cells = int(owner.max()) + 1

    # owner < neighbour, internal faces in upper triangular order
    assert np.all(owner[:n_internal] < neighbour)
    order = np.lexsort((neighbour, owner[:n_internal]))
    assert np.array_equal(order, np.arange(n_internal))

    # patches cover the boundary faces contiguously, without gaps
    start = n_internal
    for info in boundary.values():
        assert info["startFace"] == start
        start += info["nFaces"]
    assert start == n_faces

    # every cell is closed: its outward area vectors sum to zero
    total = np.zeros((n_cells, 3))
    for i, face in enumerate(faces):
        a = area_vector(points, face)
        total[owner[i]] += a
        if i < n_internal:
            total[neighbour[i]] -= a
    assert np.allclose(total, 0.0)

    # faces point out of their owner
    centres = np.zeros((n_cells, 3))
    counts = np.zeros(n_cells)
    for i, face in enumerate(faces):
        for c in [owner[i]] + ([neighbour[i]] if i < n_internal else []):
            centres[c] += points[face].mean(axis=0)
            counts[c] += 1
    centres /= counts[:, None]
    for i, face in enumerate(faces):
        a = area_vector(points, face)
        assert np.dot(a, points[face].mean(axis=0) - centres[owner[i]]) > 0

    return points, faces, owner, neighbour, boundary


def test_two_hexahedra(tmp_path):
    meshlane.openfoam.write(tmp_path / "case.foam", two_hex_mesh())

    poly = tmp_path / "constant" / "polyMesh"
    assert sorted(p.name for p in poly.iterdir()) == [
        "boundary",
        "faces",
        "neighbour",
        "owner",
        "points",
    ]
    assert (tmp_path / "case.foam").exists()

    points, faces, owner, neighbour, boundary = check_polymesh(poly)
    assert np.array_equal(points, TWO_HEX_POINTS)
    assert len(faces) == 11
    assert list(neighbour) == [1]
    assert owner[0] == 0
    assert sorted(faces[0]) == [1, 2, 5, 6]
    # no groups: every boundary face goes to defaultFaces
    assert boundary == {"defaultFaces": {"type": "patch", "nFaces": 10, "startFace": 1}}

    owner_text = (poly / "owner").read_text()
    assert "nPoints:12  nCells:2  nFaces:11  nInternalFaces:1" in owner_text


def test_mixed_cell_types(tmp_path):
    meshlane.openfoam.write(tmp_path / "case.foam", mixed_mesh())
    _, faces, _, neighbour, _ = check_polymesh(tmp_path / "constant" / "polyMesh")
    # 6 + 4 + 5 + 5 faces, 3 of them shared
    assert len(faces) == 17
    assert len(neighbour) == 3

    mesh = meshlane.openfoam.read(tmp_path / "case.foam")
    counts = {b.type: len(b) for b in mesh.cells if b.dim == 3}
    assert counts == {"hexahedron": 1, "tetra": 1, "wedge": 1, "pyramid": 1}


def test_patches_from_cell_sets(tmp_path):
    meshlane.openfoam.write(tmp_path / "case.foam", two_hex_mesh(with_patches=True))
    _, faces, owner, _, boundary = check_polymesh(tmp_path / "constant" / "polyMesh")

    assert {k: (v["nFaces"], v["startFace"]) for k, v in boundary.items()} == {
        "inlet": (1, 1),
        "outlet": (1, 2),
        "walls": (8, 3),
    }
    assert sorted(faces[1]) == [0, 3, 4, 7]
    assert owner[1] == 0
    assert sorted(faces[2]) == [8, 9, 10, 11]
    assert owner[2] == 1


def test_default_faces_for_ungrouped_boundary(tmp_path):
    mesh = two_hex_mesh(with_patches=True)
    mesh.cell_sets = {"inlet": mesh.cell_sets["inlet"]}
    meshlane.openfoam.write(tmp_path / "case.foam", mesh)
    boundary = check_polymesh(tmp_path / "constant" / "polyMesh")[4]
    assert {k: v["nFaces"] for k, v in boundary.items()} == {
        "inlet": 1,
        "defaultFaces": 9,
    }


def test_patches_from_cell_tags(tmp_path):
    # the representation the OpenFOAM and MED readers use
    mesh = two_hex_mesh(with_patches=True)
    mesh.cell_sets = {}
    tags = np.array([-1, -2] + [-3] * 8)
    mesh.cell_data["cell_tags"] = [np.zeros(2, dtype=int), tags]
    mesh.cell_tags = {-1: ["inlet"], -2: ["outlet"], -3: ["walls"]}
    meshlane.openfoam.write(tmp_path / "case.foam", mesh)
    boundary = check_polymesh(tmp_path / "constant" / "polyMesh")[4]
    assert {k: v["nFaces"] for k, v in boundary.items()} == {
        "inlet": 1,
        "outlet": 1,
        "walls": 8,
    }


def test_round_trip_keeps_patches_and_types(tmp_path):
    mesh = two_hex_mesh(with_patches=True)
    mesh.patch_types = {"walls": "wall"}
    meshlane.openfoam.write(tmp_path / "a" / "a.foam", mesh)

    mesh2 = meshlane.openfoam.read(tmp_path / "a" / "a.foam")
    assert mesh2.patch_types == {"inlet": "patch", "outlet": "patch", "walls": "wall"}
    hexes = [b for b in mesh2.cells if b.type == "hexahedron"]
    assert len(hexes) == 1
    assert sorted(map(sorted, hexes[0].data.tolist())) == sorted(
        map(sorted, TWO_HEX_CELLS)
    )

    meshlane.openfoam.write(tmp_path / "b" / "b.foam", mesh2)
    boundary = check_polymesh(tmp_path / "b" / "constant" / "polyMesh")[4]
    assert {k: (v["type"], v["nFaces"]) for k, v in boundary.items()} == {
        "inlet": ("patch", 1),
        "outlet": ("patch", 1),
        "walls": ("wall", 8),
    }


def test_polyhedron_cells(tmp_path):
    # a unit cube given as a general polyhedron, faces in arbitrary orientation
    points = np.array(TWO_HEX_POINTS[:8])
    faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5]]
    faces.append([0, 4, 7, 3])
    data = np.empty(1, dtype=object)
    data[0] = [np.array(f) for f in faces]
    mesh = meshlane.Mesh(points, [meshlane.CellBlock("polyhedron8", data)])

    meshlane.openfoam.write(tmp_path / "case.foam", mesh)
    check_polymesh(tmp_path / "constant" / "polyMesh")
    mesh2 = meshlane.openfoam.read(tmp_path / "case.foam")
    assert [b.type for b in mesh2.cells if b.dim == 3] == ["hexahedron"]


def test_quadratic_cells_use_corner_nodes(tmp_path):
    meshlane.openfoam.write(tmp_path / "case.foam", helpers.tet10_mesh)
    points = check_polymesh(tmp_path / "constant" / "polyMesh")[0]
    # the mid-edge nodes are unused, hence dropped
    corners = helpers.tet10_mesh.points[helpers.tet10_mesh.cells[0].data[0, :4]]
    assert np.array_equal(np.sort(points, axis=0), np.sort(corners, axis=0))
    mesh = meshlane.openfoam.read(tmp_path / "case.foam")
    assert [b.type for b in mesh.cells if b.dim == 3] == ["tetra"]


def test_interior_surface_cells_are_ignored(tmp_path):
    mesh = two_hex_mesh()
    mesh.cells.append(meshlane.CellBlock("quad", np.array([[1, 2, 6, 5]])))
    mesh.cell_sets = {"baffle": [np.array([], dtype=int), np.array([0])]}
    meshlane.openfoam.write(tmp_path / "case.foam", mesh)
    boundary = check_polymesh(tmp_path / "constant" / "polyMesh")[4]
    assert list(boundary) == ["defaultFaces"]


@pytest.mark.parametrize("target", ["case", "case/constant/polyMesh"])
def test_directory_targets(tmp_path, target):
    meshlane.openfoam.write(tmp_path / target, two_hex_mesh())
    assert (tmp_path / "case" / "constant" / "polyMesh" / "faces").exists()


def test_write_by_extension(tmp_path):
    meshlane.write(tmp_path / "case.foam", two_hex_mesh())
    mesh = meshlane.read(tmp_path / "case.foam")
    assert [b.type for b in mesh.cells if b.dim == 3] == ["hexahedron"]


def test_full_precision(tmp_path):
    mesh = two_hex_mesh()
    mesh.points = mesh.points + np.pi * 1.0e-9
    meshlane.openfoam.write(tmp_path / "case.foam", mesh)
    points = _read_points(tmp_path / "constant" / "polyMesh" / "points")
    assert np.array_equal(points, mesh.points)


def test_2d_mesh_is_rejected(tmp_path):
    with pytest.raises(WriteError):
        meshlane.openfoam.write(tmp_path / "case.foam", helpers.tri_mesh_2d)


def test_surface_mesh_is_rejected(tmp_path):
    with pytest.raises(WriteError):
        meshlane.openfoam.write(tmp_path / "case.foam", helpers.tri_mesh)


def test_face_shared_by_three_cells_is_rejected(tmp_path):
    points = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1], [1, 1, 1]]
    cells = [("tetra", [[0, 1, 2, 3], [0, 2, 1, 4], [0, 1, 2, 5]])]
    with pytest.raises(WriteError):
        meshlane.openfoam.write(tmp_path / "case.foam", meshlane.Mesh(points, cells))


def test_buffer_is_rejected():
    import io

    with pytest.raises(WriteError):
        meshlane.openfoam.write(io.StringIO(), two_hex_mesh())


def test_unused_points_are_dropped(tmp_path):
    # an isolated node and a line off the mesh: checkMesh rejects unused points
    mesh = two_hex_mesh()
    mesh.points = np.vstack([mesh.points, [[5.0, 5.0, 5.0], [6.0, 5.0, 5.0]]])
    mesh.cells.append(meshlane.CellBlock("vertex", np.array([[12]])))
    mesh.cells.append(meshlane.CellBlock("line", np.array([[12, 13]])))
    meshlane.openfoam.write(tmp_path / "case.foam", mesh)
    points = check_polymesh(tmp_path / "constant" / "polyMesh")[0]
    assert np.array_equal(points, TWO_HEX_POINTS)


def test_overlapping_groups_use_the_first(tmp_path):
    mesh = two_hex_mesh(with_patches=True)
    mesh.cell_sets["inlet_and_outlet"] = [np.array([], dtype=int), np.array([0, 1])]
    meshlane.openfoam.write(tmp_path / "case.foam", mesh)
    boundary = check_polymesh(tmp_path / "constant" / "polyMesh")[4]
    assert list(boundary) == ["inlet", "outlet", "walls"]


def test_stale_files_are_moved_aside(tmp_path):
    poly = tmp_path / "constant" / "polyMesh"
    meshlane.openfoam.write(tmp_path / "case.foam", mixed_mesh())
    # leftovers of a snappyHexMesh run on the previous mesh
    (poly / "cellZones").write_text("stale")
    (poly / "faces.gz").write_bytes(b"stale")
    (poly / "sets").mkdir()
    (poly / "sets" / "region").write_text("stale")

    for backup in ["polyMesh.orig", "polyMesh.orig.1"]:
        meshlane.openfoam.write(tmp_path / "case.foam", two_hex_mesh())
        assert sorted(p.name for p in poly.iterdir()) == [
            "boundary",
            "faces",
            "neighbour",
            "owner",
            "points",
        ]
        assert sorted(p.name for p in (poly.parent / backup).iterdir()) == [
            "cellZones",
            "faces.gz",
            "sets",
        ]
        check_polymesh(poly)
        # leave the same leftovers for the second write
        (poly / "cellZones").write_text("stale")
        (poly / "faces.gz").write_bytes(b"stale")
        (poly / "sets").mkdir()
        (poly / "sets" / "region").write_text("stale")


@pytest.mark.parametrize(
    "mesh",
    [
        two_hex_mesh(with_patches=True),
        mixed_mesh(),
        pytest.param("polyhedron", id="polyhedron"),
    ],
)
def test_binary_matches_ascii(tmp_path, mesh):
    if mesh == "polyhedron":
        faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6]]
        faces += [[1, 2, 6, 5], [0, 4, 7, 3]]
        data = np.empty(1, dtype=object)
        data[0] = [np.array(f) for f in faces]
        mesh = meshlane.Mesh(
            TWO_HEX_POINTS[:8], [meshlane.CellBlock("polyhedron8", data)]
        )
    meshlane.openfoam.write(tmp_path / "a" / "a.foam", mesh)
    meshlane.openfoam.write(tmp_path / "b" / "b.foam", mesh, binary=True)

    poly = tmp_path / "b" / "constant" / "polyMesh"
    headers = {f: (poly / f).read_bytes().split(b"}")[0] for f in ["faces", "owner"]}
    assert b"format      binary;" in headers["owner"]
    assert b"class       faceCompactList;" in headers["faces"]
    assert b"format      ascii;" in (poly / "boundary").read_bytes()

    ascii_ = read_polymesh(tmp_path / "a" / "constant" / "polyMesh")
    binary = check_polymesh(poly)
    assert np.array_equal(ascii_[0], binary[0])  # points, bit for bit
    assert ascii_[1] == binary[1]  # faces
    for a, b in zip(ascii_[2:4], binary[2:4]):  # owner, neighbour
        assert np.array_equal(a, b)
    assert ascii_[4] == binary[4]  # boundary


def test_binary_round_trip(tmp_path):
    mesh = two_hex_mesh(with_patches=True)
    mesh.patch_types = {"walls": "wall"}
    meshlane.openfoam.write(tmp_path / "case.foam", mesh, binary=True)
    mesh2 = meshlane.openfoam.read(tmp_path / "case.foam")
    assert mesh2.patch_types == {"inlet": "patch", "outlet": "patch", "walls": "wall"}
    assert [len(b) for b in mesh2.cells if b.type == "hexahedron"] == [2]

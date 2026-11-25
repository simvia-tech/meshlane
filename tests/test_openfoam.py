"""
Tests for the OpenFOAM polyMesh reader.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from meshlane.openfoam._openfoam import (
    _build_boundary_cells,
    _build_boundary_polygons,
    _build_volume_cells,
    _data_start,
    _detect_format,
    _parse_boundary,
    _parse_faces_ascii,
    _parse_int_list_ascii,
    _parse_points_ascii,
    _read_binary_faces,
    _read_binary_labels,
    _read_binary_points,
    _read_faces,
    _read_foam_lines,
    _read_int_list,
    _read_points,
    _resolve_polymesh,
    _strip_comments,
    read,
)

# ---------------------------------------------------------------------------
# Shared header templates
# ---------------------------------------------------------------------------

ASCII_HEADER = """FoamFile
{{
    version     2.0;
    format      ascii;
    class       {cls};
    object      {obj};
}}
"""

BINARY_HEADER = """FoamFile
{{
    version     2.0;
    format      binary;
    arch        "LSB;label={lab};scalar={sca}";
    class       {cls};
    object      {obj};
}}
"""


def _corner_triple(p, i=1, j=2, k=3):
    """Scalar triple product of the edges from p[0] to p[i], p[j], p[k]."""
    return float(np.dot(p[i] - p[0], np.cross(p[j] - p[0], p[k] - p[0])))


# Outward faces of the meshlane hexahedron, in local node indices.
HEX_FACES = [
    (0, 3, 2, 1),
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (1, 2, 6, 5),
    (2, 3, 7, 6),
    (3, 0, 4, 7),
]


def _hex_volume(points, conn):
    """Signed volume of a hexahedron, from its outward faces (each face fanned
    around its centroid, closed with the cell centroid)."""
    X = points[np.asarray(conn)]
    c = X.mean(axis=0)
    vol = 0.0
    for f in HEX_FACES:
        F = X[list(f)]
        fc = F.mean(axis=0)
        for i in range(len(f)):
            a, b = F[i] - c, F[(i + 1) % len(f)] - c
            vol += np.dot(a, np.cross(b, fc - c)) / 6
    return vol


def _write_ascii_points(path: Path, points: np.ndarray) -> None:
    lines = [ASCII_HEADER.format(cls="vectorField", obj="points")]
    lines.append(f"{len(points)}")
    lines.append("(")
    for p in points:
        lines.append(f"({p[0]} {p[1]} {p[2]})")
    lines.append(")")
    path.write_text("\n".join(lines))


def _write_ascii_faces(path: Path, faces: list[list[int]]) -> None:
    lines = [ASCII_HEADER.format(cls="faceList", obj="faces")]
    lines.append(f"{len(faces)}")
    lines.append("(")
    for f in faces:
        lines.append(f"{len(f)}({' '.join(map(str, f))})")
    lines.append(")")
    path.write_text("\n".join(lines))


def _write_ascii_labels(path: Path, labels: list[int], obj: str = "owner") -> None:
    lines = [ASCII_HEADER.format(cls="labelList", obj=obj)]
    lines.append(f"{len(labels)}")
    lines.append("(")
    for v in labels:
        lines.append(str(v))
    lines.append(")")
    path.write_text("\n".join(lines))


def _write_ascii_boundary(path: Path, patches: dict) -> None:
    lines = [ASCII_HEADER.format(cls="polyBoundaryMesh", obj="boundary")]
    lines.append(f"{len(patches)}")
    lines.append("(")
    for name, info in patches.items():
        lines.append(f"    {name}")
        lines.append("    {")
        lines.append(f"        type        {info['type']};")
        lines.append(f"        nFaces      {info['nFaces']};")
        lines.append(f"        startFace   {info['startFace']};")
        lines.append("    }")
    lines.append(")")
    path.write_text("\n".join(lines))


@pytest.fixture
def hex_cube_data():
    """A single hexahedral cell bounded by 6 quad faces."""
    points = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=float,
    )

    faces = [
        [0, 3, 2, 1],  # bottom  (z=0)
        [4, 5, 6, 7],  # top     (z=1)
        [0, 1, 5, 4],  # front   (y=0)
        [2, 3, 7, 6],  # back    (y=1)
        [1, 2, 6, 5],  # right   (x=1)
        [0, 4, 7, 3],  # left    (x=0)
    ]
    owner: list[int] = [0] * 6
    neighbour: list[int] = []
    boundary = {
        "bottom": {"type": "wall", "nFaces": 1, "startFace": 0},
        "top": {"type": "wall", "nFaces": 1, "startFace": 1},
        "sides": {"type": "wall", "nFaces": 4, "startFace": 2},
    }
    return points, faces, owner, neighbour, boundary


@pytest.fixture
def case_dir(tmp_path, hex_cube_data):
    """Minimal single-hex case written to a temporary directory."""
    points, faces, owner, neighbour, boundary = hex_cube_data
    poly = tmp_path / "constant" / "polyMesh"
    poly.mkdir(parents=True)
    _write_ascii_points(poly / "points", points)
    _write_ascii_faces(poly / "faces", faces)
    _write_ascii_labels(poly / "owner", owner, "owner")
    _write_ascii_boundary(poly / "boundary", boundary)
    # No neighbour file - all faces are boundary faces
    (tmp_path / "case.foam").write_text("")
    return tmp_path


class TestStripComments:
    def test_block_comment(self):
        assert _strip_comments("a /* foo */ b") == "a  b"

    def test_line_comment(self):
        assert _strip_comments("a // foo\nb").strip() == "a \nb"

    def test_multiline_block(self):
        assert _strip_comments("a /* foo\nbar */ c") == "a  c"

    def test_no_comments(self):
        assert _strip_comments("hello") == "hello"

    def test_adjacent_block_comments(self):
        assert _strip_comments("/*a*//*b*/x") == "x"

    def test_empty_string(self):
        assert _strip_comments("") == ""


class TestReadFoamLines:
    """Tests for the _read_foam_lines dispatcher."""

    def test_strips_comments_and_header(self, tmp_path):
        path = tmp_path / "f"
        path.write_text(
            ASCII_HEADER.format(cls="x", obj="y") + "\n// comment\n3\n(\na\nb\nc\n)\n"
        )
        lines = _read_foam_lines(path)
        # Header and comment must be gone; data lines must survive
        assert all("FoamFile" not in l for l in lines)
        assert all("//" not in l for l in lines)
        assert any("3" in l for l in lines)

    def test_returns_list_of_strings(self, tmp_path):
        path = tmp_path / "f"
        path.write_text(ASCII_HEADER.format(cls="x", obj="y") + "\n5\n")
        result = _read_foam_lines(path)
        assert isinstance(result, list)
        assert all(isinstance(l, str) for l in result)


class TestParsePointsAscii:
    def test_basic(self):
        lines = [
            "3",
            "(",
            "(0 0 0)",
            "(1.0 2.0 3.0)",
            "(-1 -2 -3.5e-1)",
            ")",
        ]
        pts = _parse_points_ascii(lines)
        assert pts.shape == (3, 3)
        np.testing.assert_allclose(pts[1], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(pts[2], [-1, -2, -0.35])

    def test_empty(self):
        pts = _parse_points_ascii(["0", "(", ")"])
        assert pts.shape == (0,)

    def test_scientific_notation(self):
        lines = ["1", "(", "(1e2 -3.0e-1 0)", ")"]
        pts = _parse_points_ascii(lines)
        np.testing.assert_allclose(pts[0], [100.0, -0.3, 0.0])

    def test_malformed_line_skipped(self):
        # A line with only 2 numbers should be silently skipped
        lines = ["1", "(", "(1 2)", ")"]
        pts = _parse_points_ascii(lines)
        assert len(pts) == 0


class TestParseFacesAscii:
    def test_basic(self):
        lines = ["2", "(", "3(0 1 2)", "4(0 1 2 3)", ")"]
        faces = _parse_faces_ascii(lines)
        assert faces == [[0, 1, 2], [0, 1, 2, 3]]

    def test_with_spaces(self):
        lines = ["1", "(", "3 (10 20 30)", ")"]
        faces = _parse_faces_ascii(lines)
        assert faces == [[10, 20, 30]]

    def test_pentagon(self):
        lines = ["1", "(", "5(0 1 2 3 4)", ")"]
        faces = _parse_faces_ascii(lines)
        assert faces == [[0, 1, 2, 3, 4]]

    def test_empty_block(self):
        lines = ["0", "(", ")"]
        faces = _parse_faces_ascii(lines)
        assert faces == []


class TestParseIntListAscii:
    def test_basic(self):
        lines = ["3", "(", "0", "1", "2", ")"]
        arr = _parse_int_list_ascii(lines)
        np.testing.assert_array_equal(arr, [0, 1, 2])

    def test_multi_per_line(self):
        lines = ["4", "(", "0 1 2 3", ")"]
        arr = _parse_int_list_ascii(lines)
        np.testing.assert_array_equal(arr, [0, 1, 2, 3])

    def test_single_value(self):
        lines = ["1", "(", "42", ")"]
        arr = _parse_int_list_ascii(lines)
        np.testing.assert_array_equal(arr, [42])

    def test_empty(self):
        lines = ["0", "(", ")"]
        arr = _parse_int_list_ascii(lines)
        assert len(arr) == 0


class TestParseBoundary:
    def test_basic(self):
        lines = [
            "inlet",
            "{",
            "    type        patch;",
            "    nFaces      10;",
            "    startFace   100;",
            "}",
            "outlet",
            "{",
            "    type        patch;",
            "    nFaces      5;",
            "    startFace   110;",
            "}",
        ]
        patches = _parse_boundary(lines)
        assert "inlet" in patches
        assert "outlet" in patches
        assert patches["inlet"]["nFaces"] == 10
        assert patches["inlet"]["startFace"] == 100
        assert patches["inlet"]["type"] == "patch"
        assert patches["outlet"]["nFaces"] == 5

    def test_empty(self):
        assert _parse_boundary([]) == {}

    def test_wall_type(self):
        lines = [
            "wall1",
            "{",
            "    type wall;",
            "    nFaces 4;",
            "    startFace 20;",
            "}",
        ]
        patches = _parse_boundary(lines)
        assert patches["wall1"]["type"] == "wall"


class TestBuildBoundaryPolygons:
    def test_groups_by_size(self):
        faces = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5]]
        tags = [-1, -1, -2]
        cells, tag_arrays = _build_boundary_polygons(faces, tags)
        names = sorted(cb.type for cb in cells)
        assert "polygon5" in names
        assert "polygon6" in names

    def test_single_polygon_type(self):
        faces = [[0, 1, 2, 3, 4]] * 3
        tags = [-1, -1, -2]
        cells, tag_arrays = _build_boundary_polygons(faces, tags)
        assert len(cells) == 1
        assert cells[0].type == "polygon5"
        assert len(cells[0].data) == 3

    def test_tags_preserved(self):
        faces = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
        tags = [-1, -2]
        _, tag_arrays = _build_boundary_polygons(faces, tags)
        all_tags = np.concatenate(tag_arrays)
        assert -1 in all_tags
        assert -2 in all_tags


class TestBuildVolumeCells:
    def test_single_hex(self, hex_cube_data):
        points, faces, owner, neighbour, _ = hex_cube_data
        owner_arr = np.array(owner)
        neighbour_arr = np.array(neighbour, dtype=int)
        cells = _build_volume_cells(1, faces, owner_arr, neighbour_arr)
        hex_blocks = [cb for cb in cells if cb.type == "hexahedron"]
        assert len(hex_blocks) == 1
        assert len(hex_blocks[0].data) == 1

    def test_returns_cell_blocks(self, hex_cube_data):
        points, faces, owner, neighbour, _ = hex_cube_data
        cells = _build_volume_cells(
            1, faces, np.array(owner), np.array(neighbour, dtype=int)
        )
        from meshlane._mesh import CellBlock

        assert all(isinstance(cb, CellBlock) for cb in cells)

    def test_two_hexes(self):
        """Two hexahedral cells sharing one internal face."""
        faces = [
            [1, 2, 6, 5],  # internal
            [0, 3, 2, 1],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [3, 7, 6, 2],
            [0, 4, 7, 3],
            [1, 8, 9, 2],
            [5, 6, 11, 10],
            [1, 5, 10, 8],
            [2, 9, 11, 6],
            [8, 10, 11, 9],
        ]
        owner = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        neighbour = np.array([1])
        cells = _build_volume_cells(2, faces, owner, neighbour)
        n_hex = sum(len(cb.data) for cb in cells if cb.type == "hexahedron")
        assert n_hex == 2


def _single_cell(faces):
    """One cell bounded by outward faces (all owned by cell 0)."""
    return _build_volume_cells(
        1, faces, np.zeros(len(faces), dtype=int), np.array([], dtype=int)
    )


def _volume(points, faces):
    """Signed volume enclosed by outward faces (fanned around their centroid)."""
    c = points[np.unique(np.concatenate(faces))].mean(axis=0)
    vol = 0.0
    for f in faces:
        F = points[list(f)]
        fc = F.mean(axis=0)
        for i in range(len(f)):
            vol += np.dot(F[i] - c, np.cross(F[(i + 1) % len(f)] - c, fc - c)) / 6
    return vol


# Outward faces of each meshlane cell type, in local node indices.
CELL_FACES = {
    "tetra": [(0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)],
    "pyramid": [(0, 3, 2, 1), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)],
    "wedge": [(0, 2, 1), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)],
    "hexahedron": [list(f) for f in HEX_FACES],
}

SINGLE_CELLS = {
    "tetra": (
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]],
    ),
    "pyramid": (
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 1]],
        [[0, 3, 2, 1], [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]],
    ),
    "wedge": (
        [[0, 0, 0], [1, 0, 0], [0.5, 1, 0], [0, 0, 1], [1, 0, 1], [0.5, 1, 1]],
        [[0, 2, 1], [3, 4, 5], [0, 1, 4, 3], [1, 2, 5, 4], [2, 0, 3, 5]],
    ),
}


class TestBuildVolumeCellTypes:
    @pytest.mark.parametrize("kind", sorted(SINGLE_CELLS))
    def test_type_and_positive_volume(self, kind):
        points, faces = SINGLE_CELLS[kind]
        points = np.array(points, dtype=float)
        (block,) = _single_cell(faces)
        assert block.type == kind
        (conn,) = block.data
        assert sorted(conn) == list(range(len(points)))
        cell = [[conn[i] for i in f] for f in CELL_FACES[kind]]
        assert _volume(points, cell) > 0

    def test_hex_positive_volume(self, hex_cube_data):
        points, faces, *_ = hex_cube_data
        (block,) = _single_cell(faces)
        assert block.type == "hexahedron"
        assert _hex_volume(points, block.data[0]) > 0

    def test_warped_hex_keeps_positive_volume(self, hex_cube_data):
        """A warped hex whose corner triple product has the wrong sign must not be
        flipped: the orientation comes from the outward faces, not from one corner.
        """
        points, faces, *_ = hex_cube_data
        points = points.copy()
        points[4] = [0.5, 0.5, -0.1]  # node 4 dips below the bottom face
        # At corner 0 (edges to nodes 1, 3, 4) the triple product is negative
        # although the cell volume is positive.
        assert _corner_triple(points, 1, 3, 4) < 0
        (block,) = _single_cell(faces)
        assert block.type == "hexahedron"
        assert _hex_volume(points, block.data[0]) > 0

    def test_six_quads_not_a_hex_become_polyhedron(self):
        """Six quads that do not close a hexahedron are kept, as a polyhedron."""
        faces = [[0, 1, 2, 3]] * 6
        (block,) = _single_cell(faces)
        assert block.type == "polyhedron4"
        assert len(block.data) == 1

    def test_polyhedra_grouped_by_node_count(self):
        """Cells that are not a standard type keep their outward faces."""
        # a cube with one face split in two triangles (7 faces, 8 nodes)
        split = [[0, 3, 2], [0, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6]]
        split += [[1, 2, 6, 5], [0, 4, 7, 3]]
        tet = [[8, 10, 9], [8, 9, 11], [9, 10, 11], [8, 11, 10]]
        faces = split + tet
        owner = np.array([0] * len(split) + [1] * len(tet))
        cells = _build_volume_cells(2, faces, owner, np.array([], dtype=int))
        types = {cb.type: len(cb.data) for cb in cells}
        assert types == {"tetra": 1, "polyhedron8": 1}
        (poly,) = [cb for cb in cells if cb.type == "polyhedron8"]
        assert sorted(map(tuple, map(list, poly.data[0]))) == sorted(map(tuple, split))

    def test_neighbour_faces_are_reversed(self):
        """A face is outward for its owner and reversed for its neighbour."""
        cells = _build_volume_cells(
            2,
            [[1, 2, 6, 5], [0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [3, 7, 6, 2]]
            + [[0, 4, 7, 3], [1, 2, 9, 8], [5, 10, 11, 6], [1, 8, 10, 5]]
            + [[2, 6, 11, 9], [8, 9, 11, 10]],
            np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
            np.array([1]),
        )
        (block,) = cells
        points = np.array(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1]]
            + [[1, 1, 1], [0, 1, 1], [2, 0, 0], [2, 1, 0], [2, 0, 1], [2, 1, 1]],
            dtype=float,
        )
        assert all(_hex_volume(points, c) > 0 for c in block.data)

    def test_zero_cells(self):
        assert _build_volume_cells(0, [], np.array([], dtype=int), np.array([])) == []


class TestBuildBoundaryCells:
    def test_quads_only(self, hex_cube_data):
        points, faces, owner, neighbour, boundary = hex_cube_data
        cells, tags, patch_tags = _build_boundary_cells(boundary, faces)
        quad_blocks = [cb for cb in cells if cb.type == "quad"]
        assert len(quad_blocks) == 1
        assert len(quad_blocks[0].data) == 6

    def test_patch_tag_ids_negative(self, hex_cube_data):
        _, faces, *_, boundary = hex_cube_data
        _, _, patch_tags = _build_boundary_cells(boundary, faces)
        for fam_id in patch_tags:
            assert fam_id < 0

    def test_patch_names_present(self, hex_cube_data):
        _, faces, *_, boundary = hex_cube_data
        _, _, patch_tags = _build_boundary_cells(boundary, faces)
        all_names = [n for names in patch_tags.values() for n in names]
        assert "bottom" in all_names
        assert "top" in all_names
        assert "sides" in all_names

    def test_triangle_faces(self, tmp_path):
        """Boundary with triangle faces should produce a 'triangle' CellBlock."""
        faces = [
            [0, 1, 2],  # triangle boundary face
            [3, 4, 5],
        ]
        boundary = {"tri_patch": {"type": "wall", "nFaces": 2, "startFace": 0}}
        cells, tags, patch_tags = _build_boundary_cells(boundary, faces)
        tri_blocks = [cb for cb in cells if cb.type == "triangle"]
        assert len(tri_blocks) == 1
        assert len(tri_blocks[0].data) == 2

    def test_polygon_faces(self):
        """Boundary faces with 5+ nodes should produce polygonN CellBlocks."""
        faces = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
        boundary = {"poly_patch": {"type": "wall", "nFaces": 2, "startFace": 0}}
        cells, tags, patch_tags = _build_boundary_cells(boundary, faces)
        poly_blocks = [cb for cb in cells if cb.type.startswith("polygon")]
        assert len(poly_blocks) == 1
        assert poly_blocks[0].type == "polygon5"

    def test_empty_boundary(self):
        cells, tags, patch_tags = _build_boundary_cells({}, [])
        assert cells == []
        assert tags == []
        assert patch_tags == {}


class TestDetectFormat:
    def test_ascii(self, tmp_path):
        path = tmp_path / "points"
        path.write_text(ASCII_HEADER.format(cls="vectorField", obj="points"))
        fmt, lb, sb = _detect_format(path)
        assert fmt == "ascii"

    def test_binary_with_arch(self, tmp_path):
        path = tmp_path / "points"
        path.write_text(
            BINARY_HEADER.format(lab=32, sca=64, cls="vectorField", obj="points")
        )
        fmt, lb, sb = _detect_format(path)
        assert fmt == "binary"
        assert lb == 4
        assert sb == 8

    def test_binary_label64(self, tmp_path):
        path = tmp_path / "points"
        path.write_text(
            BINARY_HEADER.format(lab=64, sca=64, cls="vectorField", obj="points")
        )
        fmt, lb, sb = _detect_format(path)
        assert lb == 8
        assert sb == 8

    def test_default_label_and_scalar_ascii(self, tmp_path):
        """ASCII file without arch line → defaults (8, 8)."""
        path = tmp_path / "points"
        path.write_text(ASCII_HEADER.format(cls="vectorField", obj="points"))
        fmt, lb, sb = _detect_format(path)
        assert lb == 8
        assert sb == 8


class TestBinaryReaders:
    def test_read_binary_points(self, tmp_path):
        pts = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype="<f8")
        header = BINARY_HEADER.format(lab=32, sca=64, cls="vectorField", obj="points")
        path = tmp_path / "points"
        with open(path, "wb") as f:
            f.write(header.encode("ascii"))
            f.write(b"\n2\n(")
            f.write(pts.tobytes())
        read_pts = _read_binary_points(path, scalar_bytes=8)
        np.testing.assert_allclose(read_pts, pts)

    def test_read_binary_labels(self, tmp_path):
        labels = np.array([0, 1, 2, 3, 4], dtype="<i4")
        header = BINARY_HEADER.format(lab=32, sca=64, cls="labelList", obj="owner")
        path = tmp_path / "owner"
        with open(path, "wb") as f:
            f.write(header.encode("ascii"))
            f.write(b"\n5\n(")
            f.write(labels.tobytes())
        read_lab = _read_binary_labels(path, label_bytes=4)
        np.testing.assert_array_equal(read_lab, labels)

    def test_read_binary_faces(self, tmp_path):
        header = BINARY_HEADER.format(lab=32, sca=64, cls="faceList", obj="faces")
        faces_data = [[0, 1, 2], [0, 1, 2, 3]]
        path = tmp_path / "faces"
        # Real OpenFOAM faceList: each face is a labelList "M ( <binary> )".
        with open(path, "wb") as f:
            f.write(header.encode("ascii"))
            f.write(b"\n2\n(\n")
            for face in faces_data:
                f.write(f"{len(face)}(".encode("ascii"))
                f.write(np.array(face, dtype="<i4").tobytes())
                f.write(b")\n")
            f.write(b")\n")
        read_faces = _read_binary_faces(path, label_bytes=4)
        assert read_faces.to_lists() == faces_data

    def test_read_binary_points_wrong_size(self, tmp_path):
        """Too few bytes in the file should raise ValueError."""
        header = BINARY_HEADER.format(lab=32, sca=64, cls="vectorField", obj="points")
        path = tmp_path / "points"
        with open(path, "wb") as f:
            f.write(header.encode("ascii"))
            f.write(b"\n10\n(")
            f.write(b"\x00" * 8)  # far too few bytes
        with pytest.raises(ValueError):
            _read_binary_points(path, scalar_bytes=8)

    def test_read_binary_labels_wrong_size(self, tmp_path):
        header = BINARY_HEADER.format(lab=32, sca=64, cls="labelList", obj="owner")
        path = tmp_path / "owner"
        with open(path, "wb") as f:
            f.write(header.encode("ascii"))
            f.write(b"\n100\n(")
            f.write(b"\x00" * 4)  # far too few bytes
        with pytest.raises(ValueError):
            _read_binary_labels(path, label_bytes=4)


class TestReadDispatchers:
    """Tests for _read_points, _read_faces, _read_int_list (format dispatchers)."""

    def test_read_points_ascii(self, tmp_path):
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
        path = tmp_path / "points"
        _write_ascii_points(path, pts)
        result = _read_points(path)
        np.testing.assert_allclose(result, pts)

    def test_read_points_binary(self, tmp_path):
        pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="<f8")
        header = BINARY_HEADER.format(lab=32, sca=64, cls="vectorField", obj="points")
        path = tmp_path / "points"
        with open(path, "wb") as f:
            f.write(header.encode("ascii"))
            f.write(b"\n2\n(")
            f.write(pts.tobytes())
        result = _read_points(path)
        np.testing.assert_allclose(result, pts)

    def test_read_faces_ascii(self, tmp_path):
        faces = [[0, 1, 2], [0, 1, 2, 3]]
        path = tmp_path / "faces"
        _write_ascii_faces(path, faces)
        result = _read_faces(path)
        assert result.to_lists() == faces

    def test_read_faces_binary(self, tmp_path):
        header = BINARY_HEADER.format(lab=32, sca=64, cls="faceList", obj="faces")
        faces_data = [[0, 1, 2]]
        path = tmp_path / "faces"
        with open(path, "wb") as f:
            f.write(header.encode("ascii"))
            f.write(b"\n1\n(\n")
            f.write(b"3(")
            f.write(np.array([0, 1, 2], dtype="<i4").tobytes())
            f.write(b")\n)\n")
        result = _read_faces(path)
        assert result.to_lists() == faces_data

    def test_read_int_list_ascii(self, tmp_path):
        labels = [0, 1, 2, 3]
        path = tmp_path / "owner"
        _write_ascii_labels(path, labels, "owner")
        result = _read_int_list(path)
        np.testing.assert_array_equal(result, labels)

    def test_read_int_list_binary(self, tmp_path):
        labels = np.array([5, 6, 7], dtype="<i4")
        header = BINARY_HEADER.format(lab=32, sca=64, cls="labelList", obj="owner")
        path = tmp_path / "owner"
        with open(path, "wb") as f:
            f.write(header.encode("ascii"))
            f.write(b"\n3\n(")
            f.write(labels.tobytes())
        result = _read_int_list(path)
        np.testing.assert_array_equal(result, labels)


class TestResolvePolymesh:
    def test_from_foam_file(self, case_dir):
        foam = case_dir / "case.foam"
        poly = _resolve_polymesh(foam)
        assert poly.name == "polyMesh"
        assert poly.exists()

    def test_from_case_dir(self, case_dir):
        poly = _resolve_polymesh(case_dir)
        assert poly.name == "polyMesh"

    def test_from_polymesh_dir(self, case_dir):
        poly = _resolve_polymesh(case_dir / "constant" / "polyMesh")
        assert poly.name == "polyMesh"

    def test_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _resolve_polymesh(tmp_path / "nonexistent.foam")

    def test_case_dir_without_constant(self, tmp_path):
        """A bare polyMesh dir inside the case root should also be found."""
        pm = tmp_path / "polyMesh"
        pm.mkdir()
        poly = _resolve_polymesh(tmp_path)
        assert poly.name == "polyMesh"


class TestReadFull:
    def test_read_hex_cube(self, case_dir):
        mesh = read(case_dir / "case.foam")
        assert mesh.points.shape == (8, 3)
        hex_blocks = [cb for cb in mesh.cells if cb.type == "hexahedron"]
        assert len(hex_blocks) == 1
        assert len(hex_blocks[0].data) == 1
        quad_blocks = [cb for cb in mesh.cells if cb.type == "quad"]
        assert len(quad_blocks) == 1
        assert len(quad_blocks[0].data) == 6
        assert hasattr(mesh, "cell_tags")
        assert len(mesh.cell_tags) == 3
        for fam_id in mesh.cell_tags:
            assert fam_id < 0

    def test_read_via_case_dir(self, case_dir):
        mesh = read(case_dir)
        assert mesh.points.shape == (8, 3)

    def test_missing_neighbour_ok(self, case_dir):
        poly = case_dir / "constant" / "polyMesh"
        assert not (poly / "neighbour").exists()
        mesh = read(case_dir / "case.foam")
        assert mesh is not None

    def test_mesh_has_point_tags(self, case_dir):
        mesh = read(case_dir / "case.foam")
        assert hasattr(mesh, "point_tags")
        assert isinstance(mesh.point_tags, dict)

    def test_cell_data_length_matches_cells(self, case_dir):
        mesh = read(case_dir / "case.foam")
        if "cell_tags" in mesh.cell_data:
            assert len(mesh.cell_data["cell_tags"]) == len(mesh.cells)


class TestPublicAPI:
    """The format must be reachable through the public ``meshlane.read`` entry
    point. This is checked in a clean subprocess (only ``import meshlane``) so it
    exercises the package __init__ registration, not the direct
    ``meshlane.openfoam._openfoam`` import used elsewhere in this file.
    """

    def test_meshio_read_foam(self, case_dir):
        import subprocess
        import sys

        foam = case_dir / "case.foam"
        code = (
            "import meshlane;"
            f"m = meshlane.read(r'{foam}');"
            "assert len(m.points) == 8, m.points.shape;"
            "assert any(cb.type == 'hexahedron' for cb in m.cells)"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
        )
        assert (
            result.returncode == 0
        ), f"meshlane.read('*.foam') failed via public API:\n{result.stderr}"

    def test_openfoam_read_exposed(self):
        """meshlane.openfoam.read must exist (package __init__ ran)."""
        import meshlane

        assert hasattr(meshlane.openfoam, "read")


class TestReadTwoCellMesh:
    """Two hexahedral cells sharing one internal face."""

    @pytest.fixture
    def two_hex_case(self, tmp_path):
        points = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
                [2, 0, 0],
                [2, 1, 0],
                [2, 0, 1],
                [2, 1, 1],
            ],
            dtype=float,
        )

        faces = [
            [1, 2, 6, 5],
            [0, 3, 2, 1],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [3, 7, 6, 2],
            [0, 4, 7, 3],
            [1, 8, 9, 2],
            [5, 6, 11, 10],
            [1, 5, 10, 8],
            [2, 9, 11, 6],
            [8, 10, 11, 9],
        ]
        owner = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        neighbour = [1]
        boundary = {"walls": {"type": "wall", "nFaces": 10, "startFace": 1}}

        poly = tmp_path / "constant" / "polyMesh"
        poly.mkdir(parents=True)
        _write_ascii_points(poly / "points", points)
        _write_ascii_faces(poly / "faces", faces)
        _write_ascii_labels(poly / "owner", owner, "owner")
        _write_ascii_labels(poly / "neighbour", neighbour, "neighbour")
        _write_ascii_boundary(poly / "boundary", boundary)
        (tmp_path / "case.foam").write_text("")
        return tmp_path

    def test_two_cells_detected(self, two_hex_case):
        mesh = read(two_hex_case / "case.foam")
        assert mesh.points.shape == (12, 3)
        n_vol = sum(
            len(cb.data)
            for cb in mesh.cells
            if cb.type in ("tetra", "pyramid", "wedge", "hexahedron")
            or cb.type.startswith("polyhedron")
        )
        assert n_vol == 2

    def test_correct_point_count(self, two_hex_case):
        mesh = read(two_hex_case / "case.foam")
        assert mesh.points.shape[1] == 3

    def test_boundary_faces_present(self, two_hex_case):
        mesh = read(two_hex_case / "case.foam")
        boundary_types = {"triangle", "quad"} | {
            cb.type for cb in mesh.cells if cb.type.startswith("polygon")
        }
        boundary_blocks = [cb for cb in mesh.cells if cb.type in boundary_types]
        assert len(boundary_blocks) > 0


class TestRobustness:
    def test_data_start_finds_count(self, tmp_path):
        content = (
            BINARY_HEADER.format(lab=32, sca=64, cls="x", obj="y") + "\n5\n(binary"
        )
        path = tmp_path / "f"
        path.write_text(content)
        n, pos = _data_start(path.read_bytes())
        assert n == 5

    def test_parse_boundary_empty(self):
        assert _parse_boundary([]) == {}

    def test_parse_points_malformed(self):
        lines = ["1", "(", "(1 2)", ")"]
        pts = _parse_points_ascii(lines)
        assert len(pts) == 0

    def test_strip_comments_only_comment(self):
        assert _strip_comments("/* entire line */").strip() == ""

    def test_parse_faces_empty(self):
        faces = _parse_faces_ascii(["0", "(", ")"])
        assert faces == []

    def test_parse_int_list_empty(self):
        arr = _parse_int_list_ascii(["0", "(", ")"])
        assert len(arr) == 0

    def test_build_boundary_cells_out_of_range_face(self):
        """startFace + nFaces beyond faces list length must not crash."""
        faces = [[0, 1, 2, 3]]
        boundary = {"p": {"type": "wall", "nFaces": 10, "startFace": 0}}
        cells, tags, patch_tags = _build_boundary_cells(boundary, faces)
        # Only the one valid face should be included
        total = sum(len(cb.data) for cb in cells)
        assert total == 1


def test_read_binary_face_compact_list(tmp_path):
    """Binary faces as OpenFOAM writes them: offsets, then all node ids."""
    header = BINARY_HEADER.format(lab=32, sca=64, cls="faceCompactList", obj="faces")
    offsets = np.array([0, 4, 7], dtype="<i4")
    nodes = np.array([0, 1, 2, 3, 4, 5, 6], dtype="<i4")
    raw = (
        header.encode()
        + b"3\n("
        + offsets.tobytes()
        + b")\n7\n("
        + nodes.tobytes()
        + b")\n"
    )
    path = tmp_path / "faces"
    path.write_bytes(raw)
    assert _read_faces(path).to_lists() == [[0, 1, 2, 3], [4, 5, 6]]

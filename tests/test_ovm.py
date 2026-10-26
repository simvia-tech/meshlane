import io

import numpy as np
import pytest

import meshlane
from meshlane import ReadError

from . import helpers

# Example file of the OVM specification (without its property sections), cf.
# <https://www.graphics.rwth-aachen.de/media/openvolumemesh_static/Documentation/OpenVolumeMesh-Doc-Latest/file_format.html>.
# Its half-face normals point outwards, unlike the OpenVolumeMesh convention.
SPEC_CUBE = """OVM ASCII
Vertices
8
-1.0 -1.0 -1.0
1.0 -1.0 -1.0
1.0 1.0 -1.0
-1.0 1.0 -1.0
-1.0 -1.0 1.0
1.0 -1.0 1.0
1.0 1.0 1.0
-1.0 1.0 1.0
Edges
12
0 1
1 2
2 3
3 0
4 5
5 6
6 7
7 4
0 4
1 5
2 6
3 7
Faces
6
4 0 2 4 6
4 8 10 12 14
4 18 10 21 3
4 16 15 23 6
4 20 12 23 5
4 0 18 9 17
Polyhedra
1
6 1 2 5 6 9 10
"""

# Properties as written by OpenVolumeMesh: after the Polyhedra section
SPEC_CUBE_PROPERTIES = """VProp double "weights"
1.363
6.334
2.766
8.348
4.214
2.136
7.114
0.651
EProp int "AlgoHex::FeatureEdges"
1
1
0
1
1
0
0
1
0
0
1
1
CProp vec3d "direction"
1 0 0
"""


def _parse(text):
    """Minimal independent parser of the four OVM topology sections."""
    lines = [
        line.split()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    k = 1
    sections = {}
    for name in ["Vertices", "Edges", "Faces", "Polyhedra"]:
        assert lines[k] == [name]
        n = int(lines[k + 1][0])
        sections[name] = lines[k + 2 : k + 2 + n]
        k += 2 + n
    points = np.array(sections["Vertices"], dtype=float)
    edges = [tuple(int(v) for v in e) for e in sections["Edges"]]
    faces = [[int(h) for h in f[1:]] for f in sections["Faces"]]
    cells = [[int(h) for h in c[1:]] for c in sections["Polyhedra"]]
    return points, edges, faces, cells


def _halfface_vertices(edges, faces, hf):
    verts = [edges[he // 2][he % 2] for he in faces[hf // 2]]
    return verts[::-1] if hf % 2 else verts


@pytest.mark.parametrize(
    "mesh",
    [
        helpers.empty_mesh,
        helpers.line_mesh,
        helpers.tri_mesh,
        helpers.tri_mesh_2d,
        helpers.quad_mesh,
        helpers.quad_tri_mesh,
        helpers.tet_mesh,
        helpers.hex_mesh,
        helpers.wedge_mesh,
        helpers.pyramid_mesh,
    ],
)
def test_io(mesh, tmp_path):
    helpers.write_read(
        tmp_path, meshlane.ovm.write, meshlane.ovm.read, mesh, 1.0e-15, ".ovm"
    )


def test_generic_io(tmp_path):
    helpers.generic_io(tmp_path / "test.ovm")


def test_shared_face_written_once():
    # two hexahedra sharing the face x = 1
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
        [2.0, 1.0, 0.0],
        [2.0, 0.0, 1.0],
        [2.0, 1.0, 1.0],
    ]
    cells = [
        ("hexahedron", [[0, 1, 2, 3, 4, 5, 6, 7], [1, 8, 9, 2, 5, 10, 11, 6]]),
    ]
    buf = io.StringIO()
    meshlane.ovm.write(buf, meshlane.Mesh(points, cells))
    pts, edges, faces, polyhedra = _parse(buf.getvalue())

    assert len(edges) == 20
    assert len(faces) == 11
    # the shared face is cited once from each side
    halffaces = [hf for c in polyhedra for hf in c]
    assert sorted(set(hf // 2 for hf in halffaces)) == list(range(11))
    shared = [f for f in range(11) if sum(hf // 2 == f for hf in halffaces) == 2]
    assert len(shared) == 1
    assert {hf % 2 for hf in halffaces if hf // 2 == shared[0]} == {0, 1}
    # Euler characteristic of a solid without holes
    assert len(pts) - len(edges) + len(faces) - len(polyhedra) == 1


@pytest.mark.parametrize(
    "mesh",
    [helpers.tet_mesh, helpers.hex_mesh, helpers.wedge_mesh, helpers.pyramid_mesh],
)
def test_halfface_normals_point_inwards(mesh):
    buf = io.StringIO()
    meshlane.ovm.write(buf, mesh)
    points, edges, faces, cells = _parse(buf.getvalue())

    for cell in cells:
        verts = [v for hf in cell for v in _halfface_vertices(edges, faces, hf)]
        centre = points[list(set(verts))].mean(axis=0)
        for hf in cell:
            p = points[_halfface_vertices(edges, faces, hf)]
            normal = np.cross(p[1] - p[0], p[2] - p[0])
            assert np.dot(normal, centre - p.mean(axis=0)) > 0


def test_faces_are_closed_loops():
    buf = io.StringIO()
    meshlane.ovm.write(buf, helpers.hex_mesh)
    _, edges, faces, _ = _parse(buf.getvalue())
    for face in faces:
        for he, next_he in zip(face, face[1:] + face[:1]):
            target = edges[he // 2][1 - he % 2]
            source = edges[next_he // 2][next_he % 2]
            assert target == source


@pytest.mark.parametrize("text", [SPEC_CUBE, SPEC_CUBE + SPEC_CUBE_PROPERTIES])
def test_read_spec_cube(text):
    mesh = meshlane.ovm.read(io.StringIO(text))

    assert mesh.points.shape == (8, 3)
    assert len(mesh.cells) == 1
    assert mesh.cells[0].type == "hexahedron"
    hexa = mesh.cells[0].data[0]
    assert sorted(hexa) == list(range(8))
    # meshio ordering: 0-1-2-3 counterclockwise seen from 4-5-6-7
    p = mesh.points[hexa]
    assert np.linalg.det([p[1] - p[0], p[3] - p[0], p[4] - p[0]]) > 0
    for k in range(4):
        # node k + 4 sits above node k
        assert np.allclose(p[k + 4] - p[k], p[4] - p[0])


def test_read_lenient_syntax():
    text = SPEC_CUBE.replace("Vertices", "vertices").replace("Faces", "FACES")
    text = text.replace("\n0 1\n", "\n  0\t 1  \n")
    text = text.replace("4 0 2 4 6", "4  0 2  4 6")
    text = "# a comment\n\n" + text.replace("Edges\n", "\n# edges\nEdges\n\n")
    mesh = meshlane.ovm.read(io.StringIO(text))
    assert mesh.cells[0].type == "hexahedron"


def test_read_from_extension(tmp_path):
    path = tmp_path / "cube.ovm"
    path.write_text(SPEC_CUBE)
    mesh = meshlane.read(path)
    assert mesh.cells[0].type == "hexahedron"


def test_read_polygon_faces():
    # a surface mesh: one pentagon, no polyhedra
    text = """OVM ASCII
Vertices
5
0 0 0
1 0 0
1.5 1 0
0.5 1.5 0
-0.5 1 0
Edges
5
0 1
1 2
2 3
3 4
4 0
Faces
1
5 0 2 4 6 8
Polyhedra
0
"""
    mesh = meshlane.ovm.read(io.StringIO(text))
    assert len(mesh.cells) == 1
    assert mesh.cells[0].type == "polygon"
    assert np.array_equal(mesh.cells[0].data, [[0, 1, 2, 3, 4]])


def test_write_2d_points():
    buf = io.StringIO()
    meshlane.ovm.write(buf, helpers.tri_mesh_2d)
    points, _, _, _ = _parse(buf.getvalue())
    assert points.shape == (4, 3)
    assert np.all(points[:, 2] == 0.0)


def test_write_full_precision(tmp_path):
    points = np.array([[np.pi, np.e, 1.0 / 3.0], [1.0e-300, 2.0, 3.0], [0.1, 0.2, 0.3]])
    mesh = meshlane.Mesh(points, [("triangle", [[0, 1, 2]])])
    path = tmp_path / "precision.ovm"
    meshlane.ovm.write(path, mesh)
    assert np.array_equal(meshlane.ovm.read(path).points, points)


def test_read_binary_header():
    with pytest.raises(ReadError):
        meshlane.ovm.read(io.StringIO("OVM BINARY\n"))


def test_read_properties_between_sections():
    # the layout shown on the specification page, rejected by OpenVolumeMesh too
    text = SPEC_CUBE.replace(
        "Edges\n", 'Vertex_Property "w"\nfloat\n' + "1.0\n" * 8 + "Edges\n"
    )
    with pytest.raises(ReadError):
        meshlane.ovm.read(io.StringIO(text))


@pytest.mark.parametrize(
    "old, new",
    [
        ("Polyhedra\n1\n6 1 2 5 6 9 10", "Polyhedra\n1\n6 1 2 5 6 9 12"),  # index
        ("Faces\n6\n4 0 2 4 6", "Faces\n6\n4 0 2 4 99"),  # index
        ("4 18 10 21 3", "4 18 10 21"),  # valence mismatch
        ("4 18 10 21 3", "4 18 10 3 21"),  # open loop
        ("Edges\n12\n0 1", "Edges\n12\n0 8"),  # vertex index
    ],
)
def test_read_invalid(old, new):
    assert old in SPEC_CUBE
    with pytest.raises(ReadError):
        meshlane.ovm.read(io.StringIO(SPEC_CUBE.replace(old, new)))


def test_read_inconsistent_polyhedron():
    # four triangular half-faces that do not bound a tetrahedron: they span 5
    # vertices (face 0 is cited from both sides)
    text = """OVM ASCII
Vertices
5
0 0 0
1 0 0
0 1 0
0 0 1
1 1 1
Edges
7
0 1
1 2
2 0
0 3
1 3
2 4
1 4
Faces
3
3 0 2 4
3 0 8 7
3 2 10 13
Polyhedra
1
4 0 2 4 1
"""
    with pytest.raises(ReadError):
        meshlane.ovm.read(io.StringIO(text))

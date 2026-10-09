import h5py
import numpy as np
import pytest

import meshlane
from meshlane.cgns import _cgns

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


def _write_mixed_file(path, points, conn, n_elem, offsets=None):
    """Hand-build a CGNS/HDF5 file with a single MIXED (type 20) section."""
    pts = np.asarray(points, dtype=np.float64)
    with h5py.File(path, "w") as f:
        _cgns._init_root(f)
        base = _cgns._create_node(f, "Base", "CGNSBase_t", "I4", [3, 3])
        zone = _cgns._create_node(
            base, "Zone", "Zone_t", "I4", [[len(pts)], [n_elem], [0]]
        )
        _cgns._write_string_node(zone, "ZoneType", "ZoneType_t", "Unstructured")
        grid = _cgns._create_node(zone, "GridCoordinates", "GridCoordinates_t")
        for i, nm in enumerate(["CoordinateX", "CoordinateY", "CoordinateZ"]):
            _cgns._create_node(
                grid, nm, "DataArray_t", "R8", np.ascontiguousarray(pts[:, i])
            )
        sec = _cgns._create_node(zone, "Mixed", "Elements_t", "I4", [20, 0])
        _cgns._create_node(sec, "ElementRange", "IndexRange_t", "I8", [1, n_elem])
        _cgns._create_node(sec, "ElementConnectivity", "DataArray_t", "I8", conn)
        if offsets is not None:
            _cgns._create_node(sec, "ElementStartOffset", "DataArray_t", "I8", offsets)


MIXED_POINTS = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
]
# TETRA_4 (code 10) over nodes 1-4, then TRI_3 (code 5) over nodes 1-3. 1-based.
MIXED_CONN = [10, 1, 2, 3, 4, 5, 1, 2, 3]


@pytest.mark.parametrize("offsets", [None, [0, 5, 9]])
def test_read_mixed_section(tmp_path, offsets):
    """MIXED sections must split into one block per element type, with and
    without ElementStartOffset (real CGNS example files omit it)."""
    p = tmp_path / "mixed.cgns"
    _write_mixed_file(p, MIXED_POINTS, MIXED_CONN, 2, offsets)

    mesh = meshlane.cgns.read(p)

    got = {cb.type: np.asarray(cb.data).tolist() for cb in mesh.cells}
    assert got == {"tetra": [[0, 1, 2, 3]], "triangle": [[0, 1, 2]]}
    assert mesh.points.shape == (4, 3)


def test_mixed_roundtrips_as_homogeneous_sections(tmp_path):
    """MIXED is read-only: the writer re-emits homogeneous Elements_t sections,
    which must preserve every element."""
    p = tmp_path / "mixed.cgns"
    _write_mixed_file(p, MIXED_POINTS, MIXED_CONN, 2, None)

    mesh = meshlane.cgns.read(p)
    q = tmp_path / "out.cgns"
    meshlane.cgns.write(q, mesh)
    back = meshlane.cgns.read(q)

    assert {cb.type: len(cb.data) for cb in back.cells} == {"tetra": 1, "triangle": 1}


def test_all_sections_unsupported_raises(tmp_path):
    """A file whose every element section is unreadable must raise, not
    silently degrade to a point cloud. Regression for F1."""
    p = tmp_path / "unsupported.cgns"
    # Element type 99 does not exist in the SIDS enumeration.
    _write_mixed_file(p, MIXED_POINTS, [1, 2, 3], 1, None)
    with h5py.File(p, "r+") as f:
        f["Base"]["Zone"]["Mixed"][" data"][...] = np.array([99, 0], dtype=np.int32)

    with pytest.raises(meshlane.ReadError, match="element section"):
        meshlane.cgns.read(p)


def test_point_cloud_without_element_sections_reads(tmp_path):
    """A zone with no Elements_t sections at all is a valid point cloud."""
    pts = np.asarray(MIXED_POINTS, dtype=np.float64)
    p = tmp_path / "points.cgns"
    with h5py.File(p, "w") as f:
        _cgns._init_root(f)
        base = _cgns._create_node(f, "Base", "CGNSBase_t", "I4", [3, 3])
        zone = _cgns._create_node(base, "Zone", "Zone_t", "I4", [[len(pts)], [0], [0]])
        _cgns._write_string_node(zone, "ZoneType", "ZoneType_t", "Unstructured")
        grid = _cgns._create_node(zone, "GridCoordinates", "GridCoordinates_t")
        for i, nm in enumerate(["CoordinateX", "CoordinateY", "CoordinateZ"]):
            _cgns._create_node(
                grid, nm, "DataArray_t", "R8", np.ascontiguousarray(pts[:, i])
            )

    mesh = meshlane.cgns.read(p)
    assert mesh.points.shape == (4, 3)
    assert mesh.cells == []


# High-order node-ordering tests.
#
# CGNS and meshio (VTK) order the mid/face nodes of some quadratic elements
# differently (hex20/27, wedge15/18), so those are permuted. A CGNS->CGNS
# round-trip can't catch a wrong permutation, so each cell is written in CGNS
# order and checked to read back at its VTK reference position. pyramid13/14 are
# not yet verified, so they warn instead of being asserted.

_HEX_CORNERS = np.array(
    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
     [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float
)
_WEDGE_CORNERS = np.array(
    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1]], dtype=float
)
_HEX_CENTER = [tuple(range(8))]

# (CGNS ElementType code, corners, CGNS mid-node defs, VTK mid-node defs). Each
# mid-node is defined by the corner indices it is the centroid of, so a node
# order is turned into geometry without depending on the code under test.
_ELEMENTS = {
    "hexahedron20": (18, _HEX_CORNERS,
        [(0,1),(1,2),(2,3),(3,0),(0,4),(1,5),(2,6),(3,7),(4,5),(5,6),(6,7),(7,4)],
        [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]),
    "hexahedron27": (19, _HEX_CORNERS,
        [(0,1),(1,2),(2,3),(3,0),(0,4),(1,5),(2,6),(3,7),(4,5),(5,6),(6,7),(7,4),
         (0,1,2,3),(0,1,5,4),(1,2,6,5),(2,3,7,6),(0,4,7,3),(4,5,6,7)] + _HEX_CENTER,
        [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7),
         (0,4,7,3),(1,2,6,5),(0,1,5,4),(3,2,6,7),(0,1,2,3),(4,5,6,7)] + _HEX_CENTER),
    "wedge15": (15, _WEDGE_CORNERS,
        [(0,1),(1,2),(2,0),(0,3),(1,4),(2,5),(3,4),(4,5),(5,3)],
        [(0,1),(1,2),(2,0),(3,4),(4,5),(5,3),(0,3),(1,4),(2,5)]),
    "wedge18": (16, _WEDGE_CORNERS,
        [(0,1),(1,2),(2,0),(0,3),(1,4),(2,5),(3,4),(4,5),(5,3),
         (0,1,4,3),(1,2,5,4),(2,0,3,5)],
        [(0,1),(1,2),(2,0),(3,4),(4,5),(5,3),(0,3),(1,4),(2,5),
         (0,1,4,3),(1,2,5,4),(2,0,3,5)]),
}


def _nodes(corners, defs):
    return np.vstack([corners] + [corners[list(d)].mean(axis=0) for d in defs])


def _write_single_cgns_cell(path, code, pts, mixed=False):
    """Write a CGNS file with one element whose node i is pts[i] (CGNS order).

    ``mixed=True`` emits it as a ``MIXED`` (type 20) section (type code prefixed
    to the connectivity), which the reader handles on a separate code path.
    """
    n = len(pts)
    with h5py.File(path, "w") as f:
        _cgns._init_root(f)
        base = _cgns._create_node(f, "Base", "CGNSBase_t", "I4", [3, 3])
        zone = _cgns._create_node(base, "Zone", "Zone_t", "I4", [[n], [1], [0]])
        _cgns._write_string_node(zone, "ZoneType", "ZoneType_t", "Unstructured")
        grid = _cgns._create_node(zone, "GridCoordinates", "GridCoordinates_t")
        for i, nm in enumerate(["CoordinateX", "CoordinateY", "CoordinateZ"]):
            _cgns._create_node(grid, nm, "DataArray_t", "R8",
                               np.ascontiguousarray(pts[:, i]))
        if mixed:
            sec = _cgns._create_node(zone, "E", "Elements_t", "I4", [20, 0])  # MIXED
            _cgns._create_node(sec, "ElementRange", "IndexRange_t", "I8", [1, 1])
            # MIXED connectivity: [type_code, node1..nodeN] (nodes 1-based).
            conn = np.concatenate([[code], np.arange(1, n + 1)])
            _cgns._create_node(sec, "ElementConnectivity", "DataArray_t", "I8", conn)
            _cgns._create_node(sec, "ElementStartOffset", "DataArray_t", "I8", [0, n + 1])
        else:
            sec = _cgns._create_node(zone, "E", "Elements_t", "I4", [code, 0])
            _cgns._create_node(sec, "ElementRange", "IndexRange_t", "I8", [1, 1])
            _cgns._create_node(sec, "ElementConnectivity", "DataArray_t", "I8",
                               np.arange(1, n + 1))


@pytest.mark.parametrize("mixed", [False, True], ids=["homogeneous", "mixed"])
@pytest.mark.parametrize("ctype", list(_ELEMENTS))
def test_ordering_geometric(ctype, mixed, tmp_path):
    """Reader: a cell written in CGNS node order reads back in meshio (VTK) order.

    Input placed via the documented CGNS defs; output checked against the VTK
    defs. No extra dependencies, so it runs everywhere. Exercises both the
    homogeneous and the MIXED reader paths (each permutes separately).
    """
    code, corners, cgns_defs, vtk_defs = _ELEMENTS[ctype]
    pts = _nodes(corners, cgns_defs)
    p = tmp_path / "ref.cgns"
    _write_single_cgns_cell(p, code, pts, mixed=mixed)

    mesh = meshlane.cgns.read(p)
    got = mesh.points[mesh.cells[0].data[0]]
    np.testing.assert_allclose(got, _nodes(corners, vtk_defs))


@pytest.mark.parametrize("ctype", list(_ELEMENTS))
def test_roundtrip_identity(ctype, tmp_path):
    """write(read(x)) preserves points and connectivity. With the reader
    validated above, this transitively validates the writer's permutation."""
    code, corners, cgns_defs, _ = _ELEMENTS[ctype]
    pts = _nodes(corners, cgns_defs)
    p = tmp_path / "ref.cgns"
    _write_single_cgns_cell(p, code, pts)

    m1 = meshlane.cgns.read(p)
    q = tmp_path / "out.cgns"
    meshlane.cgns.write(q, m1)
    m2 = meshlane.cgns.read(q)
    np.testing.assert_allclose(m2.points, m1.points)
    assert [b.type for b in m2.cells] == [b.type for b in m1.cells]
    np.testing.assert_array_equal(m2.cells[0].data, m1.cells[0].data)


@pytest.mark.parametrize("ctype,code", [("pyramid13", 21), ("pyramid14", 13)])
def test_pyramid_ordering_warns(ctype, code, tmp_path, capsys):
    """pyramid13/14 ordering is unverified, so read and write must warn."""
    n = {"pyramid13": 13, "pyramid14": 14}[ctype]
    pts = np.arange(3 * n, dtype=float).reshape(n, 3)
    p = tmp_path / "pyr.cgns"
    _write_single_cgns_cell(p, code, pts)

    meshlane.cgns.read(p)
    assert ctype in capsys.readouterr().err

    mesh = meshlane.Mesh(pts, [(ctype, np.arange(n).reshape(1, n))])
    meshlane.cgns.write(tmp_path / "pyr_out.cgns", mesh)
    assert ctype in capsys.readouterr().err

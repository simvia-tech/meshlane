import pathlib

import numpy as np
import pytest

import meshlane
from meshlane.med._med import _reorder_med_cells
from meshlane.med._orient import _signed_volume

from . import cgns_helpers as cg
from . import helpers

h5py = pytest.importorskip("h5py")

MESH_DIR = pathlib.Path(__file__).resolve().parent / "meshes" / "cgns"


def _volumes(mesh):
    """Signed volume of every 3D cell (positive when correctly oriented)."""
    out = []
    for b in mesh.cells:
        if b.type in ("tetra", "pyramid", "wedge", "hexahedron"):
            conn = _reorder_med_cells(b.type, b.data)
            out.append(_signed_volume(b.type, conn, mesh.points))
        elif b.type.startswith("polyhedron"):
            out.append([_enclosed_volume(mesh.points, faces) for faces in b.data])
    return np.concatenate(out)


def _enclosed_volume(points, faces):
    """Volume enclosed by faces, positive when they point outward."""
    v = 0.0
    for f in faces:
        p = points[np.asarray(f)]
        c = p.mean(axis=0)
        for i in range(len(p)):
            v += np.dot(c, np.cross(p[i], p[(i + 1) % len(p)])) / 6.0
    return v


def _counts(mesh):
    out = {}
    for b in mesh.cells:
        out[b.type] = out.get(b.type, 0) + len(b.data)
    return out


# -- element types -------------------------------------------------------------

# CGNS code, meshlane type, number of nodes, meshlane order of the CGNS nodes
# (checked against ParaView's CGNS reader)
_HEXA_20 = list(range(12)) + [16, 17, 18, 19, 12, 13, 14, 15]
ELEMENT_TYPES = [
    (3, "line", 2, None),
    (4, "line3", 3, None),
    (5, "triangle", 3, None),
    (6, "triangle6", 6, None),
    (7, "quad", 4, None),
    (8, "quad8", 8, None),
    (9, "quad9", 9, None),
    (10, "tetra", 4, None),
    (11, "tetra10", 10, None),
    (12, "pyramid", 5, None),
    (21, "pyramid13", 13, None),
    (14, "wedge", 6, None),
    (15, "wedge15", 15, list(range(9)) + [12, 13, 14, 9, 10, 11]),
    (17, "hexahedron", 8, None),
    (18, "hexahedron20", 20, _HEXA_20),
    (19, "hexahedron27", 27, _HEXA_20 + [24, 22, 21, 23, 20, 25, 26]),
]


@pytest.mark.parametrize("code, cell_type, k, order", ELEMENT_TYPES)
def test_read_element_type(tmp_path, code, cell_type, k, order):
    path = tmp_path / "mesh.cgns"
    points = np.random.default_rng(0).random((k, 3))
    with cg.new_file(path) as f:
        zone = cg.unstructured_zone(cg.base(f), points, 1)
        cg.section(zone, "Elem", code, 1, np.arange(1, k + 1), 1)

    mesh = meshlane.read(path)

    assert [b.type for b in mesh.cells] == [cell_type]
    expected = list(range(k)) if order is None else order
    assert mesh.cells[0].data.tolist() == [expected]


@pytest.mark.parametrize(
    "version, code, k, cell_type",
    [
        (3.0, 18, 8, "hexahedron"),  # 3.0 numbering: HEXA_8 was 18
        (3.0, 13, 13, "pyramid13"),  # PYRA_14 was 13, now PYRA_13
        (3.1, 18, 20, "hexahedron20"),
    ],
)
def test_read_old_numbering(tmp_path, version, code, k, cell_type):
    path = tmp_path / "mesh.cgns"
    with cg.new_file(path, version=version) as f:
        zone = cg.unstructured_zone(cg.base(f), np.random.random((k, 3)), 1)
        cg.section(zone, "Elem", code, 1, np.arange(1, k + 1), 1, dtype=np.int32)

    assert [b.type for b in meshlane.read(path).cells] == [cell_type]


# a hexahedron and a tetrahedron on its top face, plus a quad and a triangle
_POINTS = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.5, 0.5, 2.0],
    ]
)
_HEX = [1, 2, 3, 4, 5, 6, 7, 8]
_TET = [5, 6, 7, 9]
_QUAD = [1, 4, 3, 2]
_TRI = [6, 7, 9]


@pytest.mark.parametrize("offsets", [False, True])
def test_read_mixed(tmp_path, offsets):
    elements = [(17, _HEX), (10, _TET), (7, _QUAD), (5, _TRI)]
    conn = [x for code, nodes in elements for x in [code] + nodes]
    starts = np.cumsum([0] + [len(nodes) + 1 for _, nodes in elements])
    path = tmp_path / "mesh.cgns"
    with cg.new_file(path) as f:
        zone = cg.unstructured_zone(cg.base(f), _POINTS, 2)
        cg.section(zone, "Mixed", 20, 1, conn, 4, starts if offsets else None)

    mesh = meshlane.read(path)

    assert _counts(mesh) == {"hexahedron": 1, "tetra": 1, "quad": 1, "triangle": 1}
    assert (_volumes(mesh) > 0).all()


def test_read_mixed_old_polygon(tmp_path):
    # before CGNS 3.0, NGON_n was 21 and MIXED could hold polygons as NGON_n + n
    points = np.array([[0, 0], [1, 0], [2, 0.5], [1, 1], [0, 1], [2, 1.5]], float)
    conn = [26, 1, 2, 3, 4, 5, 5, 3, 6, 4]
    path = tmp_path / "mesh.cgns"
    with cg.new_file(path, version=2.5) as f:
        zone = cg.unstructured_zone(cg.base(f, cell_dim=2, phys_dim=2), points, 2)
        cg.section(zone, "Mixed", 20, 1, conn, 2, dtype=np.int32)

    mesh = meshlane.read(path)

    assert _counts(mesh) == {"polygon5": 1, "triangle": 1}
    assert mesh.points.shape == (6, 2)


# two hexahedra side by side, and a cube with a split face (a polyhedron)
_BOX = np.array(
    [[x, y, z] for z in (0.0, 1.0) for y in (0.0, 1.0) for x in (0.0, 1.0, 2.0)]
)


def _box_faces():
    """NGON faces (1-based nodes) and NFACE cells (1-based faces) of _BOX, with
    each face listed once and the cell faces signed as the CGNS spec says."""
    n = lambda x, y, z: 1 + x + 3 * y + 6 * z  # noqa: E731
    faces = [
        [n(0, 0, 0), n(0, 1, 0), n(1, 1, 0), n(1, 0, 0)],  # bottom left (out: -z)
        [n(1, 0, 0), n(1, 1, 0), n(2, 1, 0), n(2, 0, 0)],  # bottom right
        [n(0, 0, 1), n(1, 0, 1), n(1, 1, 1), n(0, 1, 1)],  # top left (out: +z)
        [n(1, 0, 1), n(2, 0, 1), n(2, 1, 1), n(1, 1, 1)],  # top right
        [n(0, 0, 0), n(1, 0, 0), n(1, 0, 1), n(0, 0, 1)],  # front left (out: -y)
        [n(1, 0, 0), n(2, 0, 0), n(2, 0, 1), n(1, 0, 1)],  # front right
        [n(0, 1, 0), n(0, 1, 1), n(1, 1, 1), n(1, 1, 0)],  # back left (out: +y)
        [n(1, 1, 0), n(1, 1, 1), n(2, 1, 1), n(2, 1, 0)],  # back right
        [n(0, 0, 0), n(0, 0, 1), n(0, 1, 1), n(0, 1, 0)],  # x = 0 (out: -x)
        [n(2, 0, 0), n(2, 1, 0), n(2, 1, 1), n(2, 0, 1)],  # x = 2 (out: +x)
        [n(1, 0, 0), n(1, 1, 0), n(1, 1, 1), n(1, 0, 1)],  # x = 1 (out of left)
    ]
    cells = [[1, 3, 5, 7, 9, 11], [2, 4, 6, 8, 10, -11]]
    return faces, cells


def _write_ngon(zone, faces, cells, cgns4, sign):
    first = 1
    for name, code, items, values in (
        ("NGon", 22, faces, faces),
        ("NFace", 23, cells, [[sign * x for x in c] for c in cells]),
    ):
        if cgns4:
            offsets = np.cumsum([0] + [len(v) for v in values])
            conn = [x for v in values for x in v]
        else:
            offsets = None
            conn = [x for v in values for x in [len(v)] + v]
        cg.section(zone, name, code, first, conn, len(items), offsets)
        first += len(items)


@pytest.mark.parametrize("cgns4", [False, True])
@pytest.mark.parametrize("sign", [1, -1])
def test_read_ngon_nface(tmp_path, cgns4, sign):
    # sign -1 flips the NFACE_n signs, as some writers do: they must not matter
    faces, cells = _box_faces()
    path = tmp_path / "mesh.cgns"
    with cg.new_file(path) as f:
        zone = cg.unstructured_zone(cg.base(f), _BOX, 2)
        _write_ngon(zone, faces, cells, cgns4, sign)

    mesh = meshlane.read(path)

    assert _counts(mesh) == {"hexahedron": 2, "quad": 10}
    volumes = _volumes(mesh)
    assert np.allclose(volumes, 1.0)
    boundary = [f for b in mesh.cells if b.type == "quad" for f in b.data]
    assert np.isclose(_enclosed_volume(mesh.points, boundary), 2.0)


def test_read_nface_inconsistent_signs(tmp_path):
    # one face of the right cell has the wrong sign: its orientation comes from
    # the geometry instead
    faces, cells = _box_faces()
    cells = [cells[0], [-2, 4, 6, 8, 10, -11]]
    path = tmp_path / "mesh.cgns"
    with cg.new_file(path) as f:
        zone = cg.unstructured_zone(cg.base(f), _BOX, 2)
        _write_ngon(zone, faces, cells, cgns4=True, sign=1)

    mesh = meshlane.read(path)

    assert _counts(mesh) == {"hexahedron": 2, "quad": 10}
    assert np.allclose(_volumes(mesh), 1.0)


def test_read_polyhedron(tmp_path):
    # the left cube with its top face split in two triangles: 7 faces
    faces, cells = _box_faces()
    top = faces[2]
    faces = faces + [[top[0], top[1], top[2]], [top[0], top[2], top[3]]]
    cells = [[1, 12, 13, 5, 7, 9, 11]]
    path = tmp_path / "mesh.cgns"
    with cg.new_file(path) as f:
        zone = cg.unstructured_zone(cg.base(f), _BOX, 1)
        _write_ngon(zone, faces, cells, cgns4=True, sign=1)

    mesh = meshlane.read(path)

    assert mesh.cells[0].type == "polyhedron8"
    assert len(mesh.cells[0].data[0]) == 7
    assert np.allclose(_volumes(mesh), [1.0])


def test_read_ngon_2d(tmp_path):
    # in a 2D zone without NFACE_n, NGON_n elements are the cells
    points = np.array([[0, 0], [1, 0], [2, 0.5], [1, 1], [0, 1], [2, 1.5]], float)
    polygons = [[1, 2, 3, 4, 5], [4, 3, 6]]
    offsets = np.cumsum([0] + [len(p) for p in polygons])
    path = tmp_path / "mesh.cgns"
    with cg.new_file(path) as f:
        zone = cg.unstructured_zone(cg.base(f, cell_dim=2, phys_dim=2), points, 2)
        cg.section(zone, "NGon", 22, 1, sum(polygons, []), 2, offsets)

    mesh = meshlane.read(path)

    assert _counts(mesh) == {"triangle": 1, "polygon5": 1}


def test_import_without_h5py():
    # h5py is optional ([all] extra): meshlane must import without it
    import subprocess
    import sys

    code = "import sys; sys.modules['h5py'] = None; import meshlane; meshlane.cgns"
    subprocess.run([sys.executable, "-c", code], check=True)


# -- zones, links, time series -------------------------------------------------


def test_read_multi_zone(tmp_path):
    path = tmp_path / "mesh.cgns"
    with cg.new_file(path) as f:
        base = cg.base(f)
        for name, shift in (("left", 0.0), ("right", 1.0)):
            zone = cg.unstructured_zone(base, _POINTS + [shift, 0, 0], 2, name=name)
            cg.section(zone, "Hex", 17, 1, _HEX, 1)
            cg.section(zone, "Tet", 10, 2, _TET, 1)

    mesh = meshlane.read(path)

    assert len(mesh.points) == 2 * len(_POINTS)
    assert [b.type for b in mesh.cells] == ["hexahedron", "tetra"] * 2
    assert mesh.cells[2].data.min() == len(_POINTS)
    sets = {k: [len(s) for s in v] for k, v in mesh.cell_sets.items()}
    assert sets == {"left": [1, 1, 0, 0], "right": [0, 0, 1, 1]}


def _link(parent, name, filename, path):
    group = parent.create_group(name)
    group.attrs.create("name", np.bytes_(name), dtype="S33")
    group.attrs.create("label", np.bytes_(""), dtype="S33")
    group.attrs.create("type", np.bytes_("LK"), dtype="S3")
    group.create_dataset(" file", data=np.frombuffer(filename.encode(), np.int8))
    group.create_dataset(" path", data=np.frombuffer(path.encode(), np.int8))
    group[" link"] = h5py.ExternalLink(filename, path)


def test_read_links_and_time_series(tmp_path, monkeypatch):
    # the grid of each step is a link to another file, and the base lists the
    # zones of each step: only those of the first one are read
    for step in (0, 1):
        with cg.new_file(tmp_path / f"grid{step}.cgns") as f:
            zone = cg.unstructured_zone(cg.base(f), _POINTS + step, 2)
            cg.section(zone, "Hex", 17, 1, _HEX, 1)
    path = tmp_path / "series.cgns"
    with cg.new_file(path) as f:
        base = cg.base(f)
        iterative = cg.node(
            base, "TimeIterValues", "BaseIterativeData_t", np.int32([2])
        )
        cg.node(iterative, "TimeValues", "DataArray_t", np.float64([0.0, 1.0]))
        cg.node(iterative, "NumberOfZones", "DataArray_t", np.int32([1, 1]))
        names = np.zeros((2, 1, 32), dtype=np.int8)
        for i, name in enumerate(("step0", "step1")):
            names[i, 0, : len(name)] = np.frombuffer(name.encode(), np.int8)
        pointers = iterative.create_group("ZonePointers", track_order=True)
        pointers.attrs.create("name", np.bytes_("ZonePointers"), dtype="S33")
        pointers.attrs.create("label", np.bytes_("DataArray_t"), dtype="S33")
        pointers.attrs.create("type", np.bytes_("C1"), dtype="S3")
        pointers.create_dataset(" data", data=names)
        for step in (0, 1):
            zone = cg.node(base, f"step{step}", "Zone_t", np.int64([[9, 1, 0]]))
            cg.node(zone, "ZoneType", "ZoneType_t", "Unstructured")
            grid = f"grid{step}.cgns"
            _link(zone, "GridCoordinates", grid, "/Base/Zone/GridCoordinates")
            _link(zone, "Hex", grid, "/Base/Zone/Hex")

    monkeypatch.chdir(tmp_path.parent)  # links resolved next to the file
    mesh = meshlane.read(path)

    assert _counts(mesh) == {"hexahedron": 1}
    assert np.allclose(mesh.points, _POINTS)


# -- unsupported files ---------------------------------------------------------


def test_read_adf_is_rejected(tmp_path):
    path = tmp_path / "mesh.cgns"
    path.write_bytes(b"@(#)ADF Database Version A02011>" + b"\0" * 100)
    with pytest.raises(meshlane.ReadError, match="cgnsconvert"):
        meshlane.cgns.read(path)


# -- files written by the CGNS library -----------------------------------------
#
# Compatibility test files of the CGNS library (zlib license), see
# meshes/cgns/README.md. Cell counts checked against ParaView.


@pytest.mark.parametrize(
    "filename, cells",
    [
        # CGNS 3.0 element numbering (HEXA_8 stored as 18)
        ("cgnslib_vers-3000.cgns", {"hexahedron": 2560, "quad": 1216}),
        ("cgnslib_vers-4400.cgns", {"hexahedron": 2560, "quad": 1216}),
        ("cgnslib_vers-3110_hdf5-110.cgns", {"tetra": 13761, "triangle": 1630}),
    ],
)
def test_read_cgns_library_files(filename, cells):
    mesh = meshlane.read(MESH_DIR / filename)

    counts = {}
    for b in mesh.cells:
        counts[b.type] = counts.get(b.type, 0) + len(b.data)
    assert counts == cells
    assert (_volumes(mesh) > 0).all()


# -- boundary conditions, families, fields --------------------------------------


def _hex_tet_zone(f):
    """The hexahedron + tetrahedron of _POINTS, with a quad (bottom) and a
    triangle (a side of the tetrahedron) as boundary elements 3 and 4."""
    zone = cg.unstructured_zone(cg.base(f), _POINTS, 2)
    cg.section(zone, "Hex", 17, 1, _HEX, 1)
    cg.section(zone, "Tet", 10, 2, _TET, 1)
    cg.section(zone, "Bottom", 7, 3, _QUAD, 1)
    cg.section(zone, "Side", 5, 4, _TRI, 1)
    return zone


def _set_sizes(mesh):
    return {k: [len(x) for x in v] for k, v in mesh.cell_sets.items()}


def test_read_bcs(tmp_path):
    path = tmp_path / "mesh.cgns"
    with cg.new_file(path) as f:
        zone = _hex_tet_zone(f)
        cg.bc(zone, "bottom", [3], location="FaceCenter")
        cg.bc(zone, "side", [4, 4], kind="ElementRange")
        cg.bc(zone, "fluid", [1, 2], location="CellCenter")
        cg.bc(zone, "base nodes", [1, 2, 3, 4])  # Vertex: the bottom quad

    mesh = meshlane.read(path)

    assert [b.type for b in mesh.cells] == ["hexahedron", "tetra", "quad", "triangle"]
    assert _set_sizes(mesh) == {
        "bottom": [0, 0, 1, 0],
        "side": [0, 0, 0, 1],
        "fluid": [1, 1, 0, 0],
        "base nodes": [0, 0, 1, 0],
    }
    assert mesh.point_sets["base nodes"].tolist() == [0, 1, 2, 3]


def test_read_bcs_cgns_1(tmp_path):
    # CGNS 1.x: the location of a BC is under its BCDataSet, the family name is
    # the name of the FamilyName_t node, and some writers used 0-based lists
    path = tmp_path / "mesh.cgns"
    with cg.new_file(path, version=1.2) as f:
        zone = cg.unstructured_zone(cg.base(f), _POINTS, 2)
        cg.section(zone, "Bottom", 7, 1, _QUAD, 1, dtype=np.int32)
        hexa = cg.section(zone, "Hex", 17, 2, _HEX, 1, dtype=np.int32)
        cg.node(hexa, "fluid", "FamilyName_t")
        cg.bc(zone, "bottom", [0], kind="ElementList")
        boundary = cg.bc(zone, "bottom faces", [1])
        dataset = cg.node(boundary, "DataSet", "BCDataSet_t", "BCWall")
        cg.node(dataset, "GridLocation", "GridLocation_t", "FaceCenter")

    mesh = meshlane.read(path)

    assert [b.type for b in mesh.cells] == ["quad", "hexahedron"]
    assert _set_sizes(mesh) == {
        "fluid": [0, 1],
        "bottom": [1, 0],
        "bottom faces": [1, 0],
    }


def test_read_families(tmp_path):
    path = tmp_path / "mesh.cgns"
    with cg.new_file(path) as f:
        zone = _hex_tet_zone(f)
        cg.node(zone["Hex"], "FamilyName", "FamilyName_t", "solid")
        cg.node(zone["Hex"], "Extra", "AdditionalFamilyName_t", "heated")
        cg.node(zone["Tet"], "FamilyName", "FamilyName_t", "solid")
        cg.node(zone, "FamilyName", "FamilyName_t", "part")

    mesh = meshlane.read(path)

    assert _set_sizes(mesh) == {
        "solid": [1, 1, 0, 0],
        "heated": [1, 0, 0, 0],
        "part": [1, 1, 0, 0],
    }


def test_read_bc_on_interior_ngon_face(tmp_path):
    # NGON_n faces are returned on the boundary, and also when a BC lists them
    faces, cells = _box_faces()
    path = tmp_path / "mesh.cgns"
    with cg.new_file(path) as f:
        zone = cg.unstructured_zone(cg.base(f), _BOX, 2)
        _write_ngon(zone, faces, cells, cgns4=True, sign=1)
        cg.bc(zone, "baffle", [11], location="FaceCenter")

    mesh = meshlane.read(path)

    assert _counts(mesh) == {"hexahedron": 2, "quad": 11}
    assert sum(len(x) for x in mesh.cell_sets["baffle"]) == 1


def test_read_flow_solutions(tmp_path):
    path = tmp_path / "mesh.cgns"
    with cg.new_file(path) as f:
        zone = _hex_tet_zone(f)
        cg.solution(zone, "Nodes", {"T": np.arange(9)})
        # cell data with a rind cell on each side, and a name used twice
        cg.solution(
            zone, "Cells", {"T": [-1, 10, 20, -1]}, location="CellCenter", rind=[1, 1]
        )

    mesh = meshlane.read(path)

    assert sorted(mesh.point_data) == ["T_Nodes"]
    assert mesh.point_data["T_Nodes"].tolist() == list(range(9))
    cell = mesh.cell_data["T_Cells"]
    assert cell[0].tolist() == [10] and cell[1].tolist() == [20]
    assert np.isnan(cell[2]).all() and np.isnan(cell[3]).all()


def test_read_flow_solution_pointers(tmp_path):
    # one solution per time step: the first one is read
    path = tmp_path / "mesh.cgns"
    with cg.new_file(path) as f:
        zone = _hex_tet_zone(f)
        cg.solution(zone, "Step0", {"T": np.zeros(9)})
        cg.solution(zone, "Step1", {"T": np.ones(9)})
        iterative = cg.node(zone, "ZoneIterativeData", "ZoneIterativeData_t")
        names = np.zeros((2, 32), dtype=np.int8)
        for i, name in enumerate(("Step0", "Step1")):
            names[i, : len(name)] = np.frombuffer(name.encode(), np.int8)
        pointers = iterative.create_group("FlowSolutionPointers", track_order=True)
        pointers.attrs.create("name", np.bytes_("FlowSolutionPointers"), dtype="S33")
        pointers.attrs.create("label", np.bytes_("DataArray_t"), dtype="S33")
        pointers.attrs.create("type", np.bytes_("C1"), dtype="S3")
        pointers.create_dataset(" data", data=names)

    mesh = meshlane.read(path)

    assert list(mesh.point_data) == ["T"]
    assert (mesh.point_data["T"] == 0).all()


def test_read_cgns_library_groups():
    mesh = meshlane.read(MESH_DIR / "cgnslib_vers-4400.cgns")

    sizes = {k: sum(len(x) for x in v) for k, v in mesh.cell_sets.items()}
    assert sizes == {"Ilo": 128, "Ihi": 128, "Walls": 960}
    assert sorted(mesh.point_data) == ["Density", "Pressure"]


# -- writer --------------------------------------------------------------------


@pytest.mark.parametrize(
    "mesh",
    [
        helpers.empty_mesh,
        helpers.tri_mesh,
        helpers.tri_mesh_2d,
        helpers.quad_mesh,
        helpers.quad_tri_mesh,
        helpers.tet_mesh,
        helpers.hex_mesh,
        helpers.wedge_mesh,
        helpers.pyramid_mesh,
        helpers.tet10_mesh,
        helpers.hex20_mesh,
        helpers.triangle6_mesh,
        helpers.quad8_mesh,
        helpers.add_point_data(helpers.tet_mesh, 1),
        helpers.add_cell_data(helpers.hex_mesh, [("a", (), np.float64)]),
    ],
)
def test_write_read(mesh, tmp_path):
    helpers.write_read(
        tmp_path, meshlane.cgns.write, meshlane.cgns.read, mesh, 1.0e-15, ".cgns"
    )


def _string_attr(obj, key):
    tid = obj.attrs.get_id(key).get_type()
    return tid.get_size(), tid.get_strpad()


def test_write_layout(tmp_path):
    # the CGNS library lists children in creation order and reads null-terminated
    # strings: without these, it sees no base in the file
    path = tmp_path / "mesh.cgns"
    meshlane.cgns.write(path, helpers.tet_mesh)

    with h5py.File(path, "r") as f:
        assert f["/"].id.get_create_plist().get_link_creation_order() != 0
        assert _string_attr(f, "label") == (33, h5py.h5t.STR_NULLTERM)
        assert list(f) == [" format", " hdf5version", "CGNSLibraryVersion", "Base"]
        base = f["Base"]
        assert base.attrs["label"] == b"CGNSBase_t"
        assert base[" data"][()].tolist() == [3, 3]
        zone = base["Zone"]
        assert _string_attr(zone, "type") == (3, h5py.h5t.STR_NULLTERM)
        assert zone[" data"].shape == (3, 1)
        assert zone[" data"][()].ravel().tolist() == [5, 2, 0]
        section = next(
            g
            for g in zone.values()
            if isinstance(g, h5py.Group) and g.attrs["label"] == b"Elements_t"
        )
        assert section[" data"].dtype == np.int32  # required by the library
        assert section[" data"][()].tolist() == [10, 0]


def test_write_groups(tmp_path):
    mesh = meshlane.Mesh(
        _POINTS - 0.5,
        [
            ("hexahedron", [np.array(_HEX) - 1]),
            ("tetra", [np.array(_TET) - 1]),
            ("quad", [np.array(_QUAD) - 1]),
            ("triangle", [np.array(_TRI) - 1]),
        ],
        point_sets={"apex": np.array([8])},
        cell_sets={
            "fluid": [
                np.array([0]),
                np.array([0]),
                np.array([], int),
                np.array([], int),
            ],
            "tip": [np.array([], int), np.array([0]), np.array([], int), np.array([0])],
            "bottom": [
                np.array([], int),
                np.array([], int),
                np.array([0]),
                np.array([], int),
            ],
        },
    )
    path = tmp_path / "mesh.cgns"
    meshlane.cgns.write(path, mesh)

    with h5py.File(path, "r") as f:
        families = sorted(
            k for k, g in f["Base"].items() if g.attrs.get("label") == b"Family_t"
        )
        bcs = sorted(f["Base/Zone/ZoneBC"])
    assert families == ["bottom", "fluid", "tip"]
    assert bcs == ["bottom", "tip"]

    out = meshlane.read(path)
    assert [b.type for b in out.cells] == ["hexahedron", "tetra", "quad", "triangle"]
    sizes = {k: sum(len(x) for x in v) for k, v in out.cell_sets.items()}
    assert sizes == {"fluid": 2, "tip": 2, "bottom": 1}
    assert out.point_sets["apex"].tolist() == [8]
    assert (_volumes(out) > 0).all()


def test_write_polyhedra(tmp_path):
    path = tmp_path / "mesh.cgns"
    meshlane.cgns.write(path, helpers.polyhedron_mesh)

    with h5py.File(path, "r") as f:
        zone = f["Base/Zone"]
        codes = sorted(
            int(g[" data"][0])
            for g in zone.values()
            if g.attrs.get("label") == b"Elements_t"
        )
    assert codes == [22, 23]  # NGON_n, NFACE_n

    out = meshlane.read(path)
    volume = {b.type: len(b.data) for b in out.cells if b.dim == 3}
    assert volume == {"tetra": 2, "pyramid": 1, "polyhedron5": 1}
    assert (_volumes(out) > 0).all()
    # the shared faces are written once: the boundary has 11 faces
    assert sum(len(b.data) for b in out.cells if b.dim == 2) == 11


def test_write_polygons_2d(tmp_path):
    path = tmp_path / "mesh.cgns"
    meshlane.cgns.write(path, helpers.polygon_mesh)

    out = meshlane.read(path)
    assert {b.type: len(b.data) for b in out.cells} == {
        "triangle": 2,
        "quad": 1,
        "polygon5": 1,
        "polygon6": 2,
    }


def test_write_long_names_and_vectors(tmp_path):
    mesh = meshlane.Mesh(
        helpers.tet_mesh.points,
        helpers.tet_mesh.cells,
        point_data={"velocity": np.ones((5, 3))},
        cell_sets={"a" * 40: [np.array([0, 1])]},
    )
    path = tmp_path / "mesh.cgns"
    meshlane.cgns.write(path, mesh)

    out = meshlane.read(path)
    assert sorted(out.point_data) == ["velocityX", "velocityY", "velocityZ"]
    assert list(out.cell_sets) == ["a" * 32]


def test_write_cgns_library_file(tmp_path):
    mesh = meshlane.read(MESH_DIR / "cgnslib_vers-4400.cgns")
    path = tmp_path / "mesh.cgns"
    meshlane.cgns.write(path, mesh)

    out = meshlane.read(path)
    assert _counts(out) == _counts(mesh)
    sizes = {k: sum(len(x) for x in v) for k, v in out.cell_sets.items()}
    assert sizes == {"Ilo": 128, "Ihi": 128, "Walls": 960}
    for name in ("Density", "Pressure"):
        assert np.allclose(out.point_data[name], mesh.point_data[name])


def test_write_face_used_three_times(tmp_path):
    # a malformed cell listing a face three times: the third use gets its own face
    cube = [
        np.array(f)
        for f in ([0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6])
    ]
    shared = np.array([3, 0, 4, 7])
    mesh = meshlane.Mesh(
        _POINTS[:8],
        [("polyhedron8", np.array([cube + [shared, shared, shared]], dtype=object))],
    )
    path = tmp_path / "mesh.cgns"
    meshlane.cgns.write(path, mesh)

    with h5py.File(path, "r") as f:
        ngon = f["Base/Zone/NGonElements/ElementRange/ data"][()]
    assert ngon.tolist() == [1, 7]


# -- structured zones, several zones -------------------------------------------


def _grid(x, y, z=None):
    """Nodes of a structured block, shape (ni, nj[, nk], dim)."""
    axes = (x, y) if z is None else (x, y, z)
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)


def test_read_structured(tmp_path):
    # two blocks of 2 x 1 x 1 cells side by side, the right one left-handed
    # (i decreasing), joined by a 1-to-1 interface on x = 2
    left = _grid([0.0, 1.0, 2.0], [0.0, 1.0], [0.0, 1.0])
    right = _grid([4.0, 3.0, 2.0], [0.0, 1.0], [0.0, 1.0])
    path = tmp_path / "mesh.cgns"
    with cg.new_file(path) as f:
        base = cg.base(f)
        for name, x, interface in (("left", left, 3), ("right", right, 3)):
            zone = cg.structured_zone(base, x, name)
            zgc = cg.node(zone, "ZoneGridConnectivity", "ZoneGridConnectivity_t")
            other = "right" if name == "left" else "left"
            conn = cg.point_range(
                zgc,
                "join",
                "GridConnectivity1to1_t",
                [interface, 1, 1],
                [interface, 2, 2],
                value=other,
            )
            cg.node(conn, "Transform", '"int[IndexDimension]"', np.int32([-1, 2, 3]))
        zone_bc = cg.node(base["left"], "ZoneBC", "ZoneBC_t")
        cg.point_range(zone_bc, "inlet", "BC_t", [1, 1, 1], [1, 2, 2])

    mesh = meshlane.read(path)

    assert _counts(mesh) == {"hexahedron": 4, "quad": 1}
    assert len(mesh.points) == 2 * 12 - 4  # interface nodes merged
    assert np.allclose(_volumes(mesh), 1.0)
    assert sum(len(x) for x in mesh.cell_sets["inlet"]) == 1
    assert len(mesh.point_sets["inlet"]) == 4
    inlet = [b for b in mesh.cells if b.type == "quad"][0].data
    assert np.allclose(mesh.points[inlet[0]][:, 0], 0.0)
    # the boundary face points out of the domain (-x)
    p = mesh.points[inlet[0]]
    normal = np.cross(p[1] - p[0], p[2] - p[0])
    assert normal[0] < 0
    sets = {k: [len(x) for x in v] for k, v in mesh.cell_sets.items()}
    assert sets["left"] == [2, 1, 0] and sets["right"] == [0, 0, 2]


def test_read_structured_2d_fields(tmp_path):
    x = _grid([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])
    path = tmp_path / "mesh.cgns"
    with cg.new_file(path) as f:
        zone = cg.structured_zone(cg.base(f, cell_dim=2, phys_dim=2), x)
        cg.point_range(
            cg.node(zone, "ZoneBC", "ZoneBC_t"), "bottom", "BC_t", [1, 1], [3, 1]
        )
        # cell data with a rind cell on each side in i: (2 + 2) x 2 values
        cells = np.array([[-1, 10, 11, -1], [-1, 12, 13, -1]], float).T
        sol = cg.solution(zone, "Cells", {}, location="CellCenter", rind=[1, 1, 0, 0])
        cg.node(sol, "T", "DataArray_t", cells)

    mesh = meshlane.read(path)

    assert _counts(mesh) == {"quad": 4, "line": 2}
    quads = mesh.cells[0].data
    p = mesh.points[quads]
    area = 0.5 * np.sum(
        p[..., 0] * np.roll(p[..., 1], -1, axis=1)
        - np.roll(p[..., 0], -1, axis=1) * p[..., 1],
        axis=1,
    )
    assert np.allclose(area, 1.0)
    assert mesh.cell_data["T"][0].tolist() == [10, 11, 12, 13]
    assert sum(len(x) for x in mesh.cell_sets["bottom"]) == 2


def test_read_multi(tmp_path):
    path = tmp_path / "mesh.cgns"
    with cg.new_file(path) as f:
        base = cg.base(f)
        cg.structured_zone(base, _grid([0.0, 1.0], [0.0, 1.0], [0.0, 1.0]), "box")
        zone = cg.unstructured_zone(base, _POINTS, 2, name="hex_tet")
        cg.section(zone, "Hex", 17, 1, _HEX, 1)
        cg.section(zone, "Tet", 10, 2, _TET, 1)

    meshes, names = meshlane.cgns.read_multi(path)

    assert names == ["box", "hex_tet"]
    assert [_counts(m) for m in meshes] == [
        {"hexahedron": 1},
        {"hexahedron": 1, "tetra": 1},
    ]


def test_write_multi(tmp_path):
    a = meshlane.Mesh(
        helpers.tet_mesh.points,
        helpers.tet_mesh.cells,
        cell_sets={"solid": [np.array([0, 1])]},
    )
    b = meshlane.Mesh(
        helpers.hex_mesh.points + 2.0,
        helpers.hex_mesh.cells,
        cell_sets={"solid": [np.array([0])]},
    )
    path = tmp_path / "mesh.cgns"
    meshlane.cgns.write_multi(path, [a, b], ["tets", "hex"])

    with h5py.File(path, "r") as f:
        families = [
            k for k, g in f["Base"].items() if g.attrs.get("label") == b"Family_t"
        ]
    assert families == ["solid"]
    meshes, names = meshlane.cgns.read_multi(path)
    assert names == ["tets", "hex"]
    assert [_counts(m) for m in meshes] == [{"tetra": 2}, {"hexahedron": 1}]
    assert [sum(len(x) for x in m.cell_sets["solid"]) for m in meshes] == [2, 1]
    merged = meshlane.read(path)
    assert sum(len(x) for x in merged.cell_sets["solid"]) == 3

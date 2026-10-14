import pathlib

import numpy as np
import pytest

import meshlane
from meshlane.med._med import _reorder_med_cells
from meshlane.med._orient import _signed_volume

from . import cgns_helpers as cg

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
    return {b.type: len(b.data) for b in mesh.cells}


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


def test_read_structured_is_rejected(tmp_path):
    path = tmp_path / "mesh.cgns"
    with cg.new_file(path) as f:
        zone = cg.node(
            cg.base(f), "Block", "Zone_t", np.int64([[2, 2, 2], [1, 1, 1], [0, 0, 0]])
        )
        cg.node(zone, "ZoneType", "ZoneType_t", "Structured")
    with pytest.raises(meshlane.ReadError, match="Structured"):
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

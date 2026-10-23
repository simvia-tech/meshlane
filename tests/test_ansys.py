import pathlib

import numpy as np
import pytest

import meshlane
from meshlane.med._med import _reorder_med_cells
from meshlane.med._orient import _signed_volume

from . import helpers

MESH_DIR = pathlib.Path(__file__).resolve().parent / "meshes" / "ansys"


def test_read_legacy_meshlane_files(tmp_path):
    # files written by meshlane before 5.6 (and meshio) list the cell nodes in
    # section 12 instead of faces; binary data follows a newline after "("
    points = helpers.tet_mesh.points
    conn = helpers.tet_mesh.cells[0].data + 1
    ascii_path = tmp_path / "ascii.msh"
    rows = "\n".join(" ".join(repr(float(x)) for x in p) for p in points)
    cells = "\n".join(" ".join(f"{i:x}" for i in c) for c in conn)
    ascii_path.write_text(
        '(1 "meshlane 5.5.0")\n(2 3)\n(10 (0 1 5 0))\n(12 (0 1 2 0))\n'
        f"(10 (1 1 5 1 3)(\n{rows}\n))\n(12 (1 1 2 1 2)(\n{cells}\n))\n"
    )
    binary_path = tmp_path / "binary.msh"
    binary_path.write_bytes(
        b'(1 "meshlane 5.5.0")\n(2 3)\n(10 (0 1 5 0))\n(12 (0 1 2 0))\n'
        + b"(3010 (1 1 5 1 3)(\n"
        + np.asarray(points, "<f8").tobytes()
        + b"\n)End of Binary Section 3010)\n(3012 (1 1 2 1 2)(\n"
        + np.asarray(conn, "<i8").tobytes()
        + b"\n)End of Binary Section 3012)\n"
    )
    for path in (ascii_path, binary_path):
        mesh = meshlane.ansys.read(path)
        assert np.allclose(mesh.points, points)
        assert mesh.cells[0].type == "tetra"
        assert (mesh.cells[0].data == conn - 1).all()


# -- Fluent files built by hand ------------------------------------------------
#
# Fluent stores faces (nodes + the cells on either side), not cell nodes. The
# helpers below write such files from a small conformal mesh, to check that the
# reader rebuilds the cells.

# outward faces of each cell type, in meshlane node order
_OUTWARD = {
    "tetra": [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
    "pyramid": [[0, 3, 2, 1], [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]],
    "wedge": [[0, 2, 1], [3, 4, 5], [0, 1, 4, 3], [1, 2, 5, 4], [2, 0, 3, 5]],
    "hexahedron": [
        [0, 3, 2, 1],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
    ],
}

# Fluent element types of section 12
_FLUENT_CELL_TYPE = {
    "triangle": 1,
    "tetra": 2,
    "quad": 3,
    "hexahedron": 4,
    "pyramid": 5,
    "wedge": 6,
    "polyhedron": 7,
}


def _faces_of(cells):
    """Faces of a conformal mesh, as [nodes, c0, c1] with 1-based cells (0: none)
    and nodes outward for c0. cells: [(type, nodes)] or [("polyhedron", faces)];
    2D cells are ("polygon", counterclockwise nodes)."""
    faces = []
    seen = {}
    for c, (cell_type, data) in enumerate(cells, start=1):
        if cell_type == "polyhedron":
            outward = data
        elif cell_type == "polygon":
            outward = [[a, b] for a, b in zip(data, np.roll(data, -1))]
        else:
            outward = [[data[i] for i in f] for f in _OUTWARD[cell_type]]
        for f in outward:
            key = frozenset(int(n) for n in f)
            if key in seen:
                faces[seen[key]][2] = c
            else:
                seen[key] = len(faces)
                faces.append([[int(n) for n in f], c, 0])
    return faces


def _write_fluent(path, points, cell_types, zones, binary, swap):
    """Write a Fluent mesh file.

    zones: [(zone id, bc type, name, faces)] with faces as built by _faces_of;
    swap stores the nodes so that their normal points into c0 instead of out of
    it (both conventions are measured by the reader)."""
    dim = points.shape[1]
    out = [b'(0 "hand-made Fluent mesh")\n', f"(2 {dim})\n".encode()]

    def section(index, header, ints=None, floats=None):
        head = f"({index} ({' '.join(header)})".encode()
        if ints is None and floats is None:
            out.append(head + b")\n")
        elif binary:
            data = (
                np.asarray(floats, dtype="<f8").tobytes()
                if floats is not None
                else np.asarray([i for row in ints for i in row], dtype="<i4").tobytes()
            )
            out.append(
                head + b"(" + data + f")\nEnd of Binary Section {index})\n".encode()
            )
        elif floats is not None:
            rows = "\n".join(" ".join(repr(float(x)) for x in p) for p in floats)
            out.append(head + f"(\n{rows}\n))\n".encode())
        else:
            rows = "\n".join(" ".join(f"{i:x}" for i in row) for row in ints)
            out.append(head + f"(\n{rows}\n))\n".encode())

    pre = "30" if binary else ""
    n = len(points)
    out.append(f"(10 (0 1 {n:x} 0 {dim}))\n".encode())
    section(f"{pre}10", ["1", "1", f"{n:x}", "1", f"{dim}"], floats=points)

    c = len(cell_types)
    types = [[_FLUENT_CELL_TYPE[t]] for t in cell_types]
    out.append(f"(12 (0 1 {c:x} 0))\n".encode())
    section(f"{pre}12", ["2", "1", f"{c:x}", "1", "0"], ints=types)

    total = sum(len(z[3]) for z in zones)
    out.append(f"(13 (0 1 {total:x} 0))\n".encode())
    first = 1
    for zone, bc, _, faces in zones:
        sizes = {len(f[0]) for f in faces}
        face_type = sizes.pop() if len(sizes) == 1 else 0
        rows = []
        for nodes, c0, c1 in faces:
            nodes = [i + 1 for i in (nodes[::-1] if swap else nodes)]
            rows.append(([len(nodes)] if face_type == 0 else []) + nodes + [c0, c1])
        last = first + len(faces) - 1
        header = [f"{zone:x}", f"{first:x}", f"{last:x}", f"{bc:x}", f"{face_type}"]
        section(f"{pre}13", header, ints=rows)
        first = last + 1

    out.append(b"(45 (2 fluid fluid)())\n")
    for zone, bc, name, _ in zones:
        out.append(f"(45 ({zone} {bc} {name})())\n".encode())
    path.write_bytes(b"".join(out))


def _split_boundary(faces, inlet):
    interior = [f for f in faces if f[2] != 0]
    boundary = [f for f in faces if f[2] == 0]
    return (
        interior,
        [f for f in boundary if inlet(f)],
        [f for f in boundary if not inlet(f)],
    )


def _volumes(mesh):
    """Signed volume of every 3D cell (positive when correctly oriented)."""
    out = []
    for b in mesh.cells:
        if b.type in _OUTWARD:
            out.append(
                _signed_volume(b.type, _reorder_med_cells(b.type, b.data), mesh.points)
            )
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


def _boundary_faces(mesh):
    return [f for b in mesh.cells if b.dim == 2 for f in b.data]


# a hexahedron with a pyramid on top, a wedge on its +x side and a tetrahedron on
# a side of the pyramid
_POINTS_3D = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.5, 0.5, 1.5],  # pyramid apex
        [2.0, 0.0, 0.5],  # wedge ridge
        [2.0, 1.0, 0.5],
        [0.5, -0.5, 1.5],  # tetra apex
    ]
)
_CELLS_3D = [
    ("hexahedron", [0, 1, 2, 3, 4, 5, 6, 7]),
    ("pyramid", [4, 5, 6, 7, 8]),
    ("wedge", [1, 5, 9, 2, 6, 10]),
    ("tetra", [4, 5, 8, 11]),
]


@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("swap", [False, True])
def test_read_fluent_cells(tmp_path, binary, swap):
    faces = _faces_of(_CELLS_3D)
    interior, inlet, walls = _split_boundary(
        faces, lambda f: all(_POINTS_3D[n][2] == 0.0 for n in f[0])
    )
    # parent face of a hanging-node refinement: must be ignored
    parent = [[[0, 1, 2, 3], 1, 0]]
    zones = [
        (3, 2, "default-interior", interior),
        (4, 10, "inlet", inlet),
        (5, 3, "walls", walls),
        (6, 31, "parent", parent),
    ]
    path = tmp_path / "mesh.msh"
    _write_fluent(path, _POINTS_3D, [t for t, _ in _CELLS_3D], zones, binary, swap)

    mesh = meshlane.read(path)

    counts = {b.type: len(b.data) for b in mesh.cells}
    assert counts == {
        "tetra": 1,
        "pyramid": 1,
        "wedge": 1,
        "hexahedron": 1,
        "triangle": 8,
        "quad": 6,
    }
    tet = abs(np.linalg.det(_POINTS_3D[[5, 8, 11]] - _POINTS_3D[4])) / 6
    volumes = _volumes(mesh)
    assert (volumes > 0).all()
    assert np.isclose(volumes.sum(), 1.0 + 1.0 / 6.0 + 0.5 + tet)
    # the boundary faces enclose the volume: they all point outward
    assert np.isclose(
        _enclosed_volume(mesh.points, _boundary_faces(mesh)), volumes.sum()
    )

    sets = {k: sum(len(s) for s in v) for k, v in mesh.cell_sets.items()}
    assert sets == {"fluid": 4, "inlet": 1, "walls": 13}


@pytest.mark.parametrize("binary", [False, True])
def test_read_fluent_polyhedron(tmp_path, binary):
    # a cube whose top face is split in two triangles: 7 faces, so a polyhedron
    points = _POINTS_3D[:8]
    hexa = [[0, 1, 2, 3, 4, 5, 6, 7][i] for f in _OUTWARD["hexahedron"] for i in f]
    hexa = [hexa[4 * i : 4 * i + 4] for i in range(6)]
    faces = [f for f in hexa if f != [4, 5, 6, 7]] + [[4, 5, 6], [4, 6, 7]]
    zones = [(3, 3, "walls", [[f, 1, 0] for f in faces])]
    path = tmp_path / "mesh.msh"
    _write_fluent(path, points, ["polyhedron"], zones, binary, swap=False)

    mesh = meshlane.ansys.read(path)

    assert [b.type for b in mesh.cells] == ["polyhedron8", "triangle", "quad"]
    assert len(mesh.cells[0].data[0]) == 7
    assert np.isclose(_volumes(mesh), [1.0]).all()
    assert np.isclose(_enclosed_volume(points, _boundary_faces(mesh)), 1.0)


@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("swap", [False, True])
def test_read_fluent_2d(tmp_path, binary, swap):
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [2.0, 0.0],
            [1.0, 2.0],
            [0.5, 2.5],
            [0.0, 2.0],
        ]
    )
    cells = [
        ("polygon", [0, 1, 2, 3]),
        ("polygon", [1, 4, 2]),
        ("polygon", [3, 2, 5, 6, 7]),
    ]
    faces = _faces_of(cells)
    interior, inlet, walls = _split_boundary(
        faces, lambda f: all(points[n][1] == 0.0 for n in f[0])
    )
    zones = [
        (3, 2, "default-interior", interior),
        (4, 10, "inlet", inlet),
        (5, 3, "walls", walls),
    ]
    path = tmp_path / "mesh.msh"
    _write_fluent(path, points, ["quad", "triangle", "polyhedron"], zones, binary, swap)

    mesh = meshlane.ansys.read(path)

    blocks = {b.type: b.data.tolist() for b in mesh.cells}
    assert sorted(blocks["quad"][0]) == [0, 1, 2, 3]
    assert sorted(blocks["triangle"][0]) == [1, 2, 4]
    assert len(blocks["polygon5"]) == 1
    assert len(blocks["line"]) == 8
    for kind, area in (("quad", 1.0), ("triangle", 0.5), ("polygon5", 1.25)):
        p = points[blocks[kind][0]]
        shoelace = np.sum(
            p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]
        )
        assert np.isclose(shoelace / 2, area)
    sets = {k: sum(len(s) for s in v) for k, v in mesh.cell_sets.items()}
    assert sets == {"fluid": 3, "inlet": 2, "walls": 6}


# -- files written by Fluent and other Ansys tools -----------------------------
#
# From https://github.com/ansys/example-data (MIT), see meshes/ansys/README.md.
# Expected values cross-checked with OpenFOAM (fluentMeshToFoam, checkMesh).


@pytest.mark.parametrize(
    "filename, cells, measure, sets",
    [
        # 3D, binary (Ansys Prime), cell zone without body
        (
            "hex.msh",
            {"hexahedron": 216, "quad": 216},
            3.2768e-05,
            ["SOLID", "WALL_SOLID"],
        ),
        # 2D, ASCII (GAMBIT), node section without ND
        (
            "solid.msh",
            {"quad": 585, "line": 120},
            0.00593409,
            ["fluid", "inlet", "outlet", "axis"],
        ),
        # 2D, ASCII with CRLF line ends (tmerge), zones in section 39
        ("disk.msh", {"quad": 6192, "line": 316}, 0.0274749, ["fluid-7", "axis-5"]),
        # 2D, triangles and quads in separate cell zones
        (
            "flap.msh",
            {"triangle": 4450, "quad": 30, "line": 240},
            0.002,
            ["solid.5", "fluid.4", "wall:008"],
        ),
    ],
)
def test_read_fluent_files(filename, cells, measure, sets):
    mesh = meshlane.read(MESH_DIR / filename)

    assert {b.type: len(b.data) for b in mesh.cells} == cells
    assert set(sets) <= set(mesh.cell_sets)
    dim = max(b.dim for b in mesh.cells)
    if dim == 3:
        v = _volumes(mesh)
    else:
        v = []
        for b in mesh.cells:
            if b.dim == 2:
                p = mesh.points[b.data]
                v.append(
                    np.sum(
                        p[..., 0] * np.roll(p[..., 1], -1, axis=1)
                        - np.roll(p[..., 0], -1, axis=1) * p[..., 1],
                        axis=1,
                    )
                    / 2
                )
        v = np.concatenate(v)
    assert (v > 0).all()
    assert np.isclose(v.sum(), measure, rtol=1e-5)


def test_read_gmsh_msh_still_works():
    # .msh is shared with Gmsh: the Fluent reader must reject Gmsh files
    mesh = meshlane.read(
        pathlib.Path(__file__).resolve().parent / "meshes" / "msh" / "insulated-2.2.msh"
    )
    assert len(mesh.points) > 0


# -- Fluent HDF5 meshes (.msh.h5) ----------------------------------------------


def _write_fluent_h5(path, points, num_cells, zones):
    """Write a Fluent .msh.h5 file. zones as for _write_fluent."""
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as f:
        m = f.create_group("meshes/1")
        m.attrs["dimension"] = [points.shape[1]]
        m.attrs["cellCount"] = [num_cells]
        m.attrs["faceOffset"] = [0]
        coords = m.create_dataset("nodes/coords/1", data=points)
        coords.attrs["minId"] = [1]
        coords.attrs["maxId"] = [len(points)]
        cz = m.create_group("cells/zoneTopology")
        cz["id"] = [2]
        cz["minId"] = np.array([1], dtype=np.uint64)
        cz["maxId"] = np.array([num_cells], dtype=np.uint64)
        cz["name"] = [b"fluid"]
        first = 1
        rows = {"id": [], "minId": [], "maxId": [], "zoneType": [], "name": []}
        for k, (zone, bc, name, faces) in enumerate(zones, start=1):
            last = first + len(faces) - 1
            for group in ("faces/nodes", "faces/c0", "faces/c1"):
                m.require_group(group)
            g = m["faces/nodes"].create_group(str(k))
            g["nnodes"] = np.array([len(f[0]) for f in faces], dtype=np.int16)
            g["nodes"] = np.array([n + 1 for f in faces for n in f[0]], dtype=np.uint32)
            c0 = m["faces/c0"].create_dataset(str(k), data=[f[1] for f in faces])
            c1 = m["faces/c1"].create_dataset(str(k), data=[f[2] for f in faces])
            for obj in (g, c0, c1):
                obj.attrs["minId"] = [first]
                obj.attrs["maxId"] = [last]
            rows["id"].append(zone)
            rows["minId"].append(first)
            rows["maxId"].append(last)
            rows["zoneType"].append(bc)
            rows["name"].append(name)
            first = last + 1
        fz = m.create_group("faces/zoneTopology")
        for key in ("id", "zoneType"):
            fz[key] = np.array(rows[key], dtype=np.int32)
        for key in ("minId", "maxId"):
            fz[key] = np.array(rows[key], dtype=np.uint64)
        fz["name"] = [";".join(rows["name"]).encode()]


def test_read_fluent_h5_cells(tmp_path):
    faces = _faces_of(_CELLS_3D)
    interior, inlet, walls = _split_boundary(
        faces, lambda f: all(_POINTS_3D[n][2] == 0.0 for n in f[0])
    )
    zones = [
        (3, 2, "default-interior", interior),
        (4, 10, "inlet", inlet),
        (5, 3, "walls", walls),
    ]
    path = tmp_path / "mesh.msh.h5"
    _write_fluent_h5(path, _POINTS_3D, len(_CELLS_3D), zones)

    mesh = meshlane.read(path)

    counts = {b.type: len(b.data) for b in mesh.cells}
    assert counts == {
        "tetra": 1,
        "pyramid": 1,
        "wedge": 1,
        "hexahedron": 1,
        "triangle": 8,
        "quad": 6,
    }
    volumes = _volumes(mesh)
    assert (volumes > 0).all()
    assert np.isclose(
        _enclosed_volume(mesh.points, _boundary_faces(mesh)), volumes.sum()
    )
    sets = {k: sum(len(s) for s in v) for k, v in mesh.cell_sets.items()}
    assert sets == {"fluid": 4, "inlet": 1, "walls": 13}


def test_read_fluent_h5_polyhedron(tmp_path):
    hexa = [[[0, 1, 2, 3, 4, 5, 6, 7][i] for i in f] for f in _OUTWARD["hexahedron"]]
    faces = [f for f in hexa if f != [4, 5, 6, 7]] + [[4, 5, 6], [4, 6, 7]]
    path = tmp_path / "mesh.msh.h5"
    _write_fluent_h5(
        path, _POINTS_3D[:8], 1, [(3, 3, "walls", [[f, 1, 0] for f in faces])]
    )

    mesh = meshlane.read(path)

    assert [b.type for b in mesh.cells] == ["polyhedron8", "triangle", "quad"]
    assert np.isclose(_volumes(mesh), [1.0]).all()


def test_read_fluent_h5_file():
    pytest.importorskip("h5py")
    mesh = meshlane.read(MESH_DIR / "simple-duct.msh.h5")

    assert {b.type: len(b.data) for b in mesh.cells} == {"hexahedron": 10, "quad": 42}
    volumes = _volumes(mesh)
    assert (volumes > 0).all()
    assert np.isclose(volumes.sum(), 10.0)
    sets = {k: sum(len(s) for s in v) for k, v in mesh.cell_sets.items()}
    assert sets == {"fluid": 10, "outflow": 1, "inlet": 1, "symm": 40}


def test_read_fluent_h5_with_msh_extension(tmp_path):
    # Fluent HDF5 meshes are sometimes saved as plain .msh
    pytest.importorskip("h5py")
    path = tmp_path / "mesh.msh"
    path.write_bytes((MESH_DIR / "simple-duct.msh.h5").read_bytes())

    mesh = meshlane.read(path)

    assert {b.type: len(b.data) for b in mesh.cells} == {"hexahedron": 10, "quad": 42}


# -- writer --------------------------------------------------------------------


def _parse(path):
    """Faces, cells and zones of a Fluent file, as the reader parses them."""
    from meshlane.ansys._ansys import _Parser

    parser = _Parser(pathlib.Path(path).read_bytes())
    parser.parse()
    return parser


def _volume_counts(mesh):
    out = {}
    for b in mesh.cells:
        if b.dim == max(c.dim for c in mesh.cells):
            out[b.type] = out.get(b.type, 0) + len(b.data)
    return out


@pytest.mark.parametrize("binary", [True, False])
@pytest.mark.parametrize(
    "mesh",
    [
        helpers.tet_mesh,
        helpers.hex_mesh,
        helpers.wedge_mesh,
        helpers.pyramid_mesh,
        meshlane.Mesh(_POINTS_3D, [(t, [c]) for t, c in _CELLS_3D]),
        helpers.polyhedron_mesh,
    ],
)
def test_write_read_3d(tmp_path, mesh, binary):
    path = tmp_path / "mesh.msh"
    meshlane.ansys.write(path, mesh, binary=binary)

    out = meshlane.ansys.read(path)

    volumes = _volumes(out)
    assert (volumes > 0).all()
    expected = np.abs(_volumes(mesh)) if mesh is not helpers.polyhedron_mesh else None
    if expected is not None:
        assert np.isclose(volumes.sum(), expected.sum())
        assert _volume_counts(out) == _volume_counts(mesh)
    # every boundary face is in the default wall zone, pointing outward
    sizes = {k: sum(len(x) for x in v) for k, v in out.cell_sets.items()}
    boundary = _boundary_faces(out)
    assert sizes == {"fluid": sum(_volume_counts(out).values()), "wall": len(boundary)}
    assert np.isclose(_enclosed_volume(out.points, boundary), volumes.sum())


@pytest.mark.parametrize("binary", [True, False])
@pytest.mark.parametrize("mesh", [helpers.tri_mesh_2d, helpers.quad_mesh])
def test_write_read_2d(tmp_path, mesh, binary):
    path = tmp_path / "mesh.msh"
    meshlane.ansys.write(path, mesh, binary=binary)

    out = meshlane.ansys.read(path)

    assert out.points.shape[1] == 2
    assert _volume_counts(out) == _volume_counts(mesh)
    for b in out.cells:
        if b.dim == 2:
            p = out.points[b.data]
            area = np.sum(
                p[..., 0] * np.roll(p[..., 1], -1, axis=1)
                - np.roll(p[..., 0], -1, axis=1) * p[..., 1],
                axis=1,
            )
            assert (area > 0).all()


@pytest.mark.parametrize("dim", [2, 3])
def test_write_face_orientation(tmp_path, dim):
    # Fluent's normal of a face (z x edge in 2D) points to c0; c1 = 0 on the
    # boundary, so boundary normals point into the domain
    mesh = helpers.quad_mesh if dim == 2 else helpers.hex_mesh
    mesh = meshlane.Mesh(mesh.points, mesh.cells)
    if dim == 3:  # two hexahedra, to have an interior face
        shifted = mesh.cells[0].data + len(mesh.points)
        mesh = meshlane.Mesh(
            np.concatenate([mesh.points, mesh.points + [0, 0, 1]]),
            [("hexahedron", np.concatenate([mesh.cells[0].data, shifted]))],
        )
    path = tmp_path / "mesh.msh"
    meshlane.ansys.write(path, mesh, binary=False)

    parser = _parse(path)
    points = np.concatenate([p for _, p in parser.nodes])
    cells = [b.data for b in mesh.cells if b.dim == dim]
    centers = np.concatenate([points[c].mean(axis=1) for c in cells])
    checked = 0
    for _, _, ptr, nodes, c0, c1 in parser.face_zones:
        for k in range(len(c0)):
            f = nodes[ptr[k] : ptr[k + 1]] - 1
            p = points[f]
            if dim == 2:
                d = p[1] - p[0]
                normal = np.array([-d[1], d[0]])
            else:
                normal = np.cross(p[1] - p[0], p[2] - p[0])
            towards_c0 = np.dot(normal, centers[c0[k] - 1] - p.mean(axis=0))
            assert towards_c0 > 0
            checked += 1
    assert checked > 0


def test_write_zones(tmp_path):
    # a hexahedron (solid) on a tetrahedron (fluid): the shared face is in a
    # group ("interface", a two-sided wall), the bottom quad is the inlet
    mesh = meshlane.Mesh(
        _POINTS_3D,
        [
            ("hexahedron", [_CELLS_3D[0][1]]),
            ("pyramid", [_CELLS_3D[1][1]]),
            ("quad", [[4, 5, 6, 7], [0, 3, 2, 1]]),
        ],
        cell_sets={
            "block": [np.array([0]), np.array([], int), np.array([], int)],
            "roof air": [np.array([], int), np.array([0]), np.array([], int)],
            "interface": [np.array([], int), np.array([], int), np.array([0])],
            "inlet": [np.array([], int), np.array([], int), np.array([1])],
        },
    )
    path = tmp_path / "mesh.msh"
    meshlane.ansys.write(
        path, mesh, zone_types={"block": "solid", "inlet": "velocity-inlet"}
    )

    zones = _parse(path).zones
    by_name = {name: kind for kind, name in zones.values()}
    assert by_name == {
        "block": "solid",
        "roof_air": "fluid",  # blanks are not allowed in Fluent zone names
        "interface": "wall",
        "inlet": "velocity-inlet",
        "wall": "wall",
    }
    out = meshlane.ansys.read(path)
    sizes = {k: sum(len(x) for x in v) for k, v in out.cell_sets.items()}
    assert sizes == {"block": 1, "roof_air": 1, "interface": 1, "inlet": 1, "wall": 8}


def test_write_unknown_zone_type(tmp_path):
    with pytest.raises(meshlane.WriteError, match="zone type"):
        meshlane.ansys.write(
            tmp_path / "mesh.msh", helpers.tet_mesh, zone_types={"x": "nozzle"}
        )


def test_write_quadratic_and_surface(tmp_path):
    path = tmp_path / "quadratic.msh"
    meshlane.ansys.write(path, helpers.hex20_mesh)
    assert _volume_counts(meshlane.ansys.read(path)) == {"hexahedron": 1}

    # a surface in space is written as faces without cells
    surface = meshlane.Mesh(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [("triangle", [[0, 1, 2], [0, 1, 3]])],
    )
    path = tmp_path / "surface.msh"
    meshlane.ansys.write(path, surface)
    out = meshlane.ansys.read(path)
    assert [(b.type, len(b.data)) for b in out.cells] == [("triangle", 2)]


@pytest.mark.parametrize(
    "filename", ["hex.msh", "solid.msh", "disk.msh", "flap.msh", "simple-duct.msh.h5"]
)
def test_write_fluent_files(tmp_path, filename):
    if filename.endswith(".h5"):
        pytest.importorskip("h5py")
    mesh = meshlane.read(MESH_DIR / filename)
    path = tmp_path / "mesh.msh"
    meshlane.ansys.write(path, mesh)

    out = meshlane.ansys.read(path)

    assert _counts(out) == _counts(mesh)

    def sizes(m):
        return sorted(sum(len(x) for x in v) for v in m.cell_sets.values())

    assert sizes(out) == sizes(mesh)


def _counts(mesh):
    out = {}
    for b in mesh.cells:
        out[b.type] = out.get(b.type, 0) + len(b.data)
    return out

from pathlib import Path

import pytest

import numpy as np

import meshio

OBJ_PATH = Path(__file__).resolve().parent / "meshes" / "ply" / "bun_zipper_res4.ply"


def test_read_str():
    meshio.read(str(OBJ_PATH))


def test_read_pathlike():
    meshio.read(OBJ_PATH)


@pytest.mark.skip
def test_read_buffer():
    with open(str(OBJ_PATH)) as f:
        meshio.read(f, "ply")


@pytest.fixture
def mesh():
    return meshio.read(OBJ_PATH)


def test_write_str(mesh, tmpdir):
    tmp_path = str(tmpdir.join("tmp.ply"))
    meshio.write(tmp_path, mesh)
    assert Path(tmp_path).is_file()


def test_write_pathlike(mesh, tmpdir):
    tmp_path = Path(tmpdir.join("tmp.ply"))
    meshio.write(tmp_path, mesh)
    assert Path(tmp_path).is_file()


@pytest.mark.skip
def test_write_buffer(mesh, tmpdir):
    tmp_path = str(tmpdir.join("tmp.ply"))
    with open(tmp_path, "w") as f:
        meshio.write(f, mesh, "ply")
    assert Path(tmp_path).is_file()

def test_msh_format_selection_for_med_data():
    from meshio._helpers import _pick_best_format
    from meshio._mesh import CellBlock

    # Points et cells minimaux pour instancier un Mesh valide
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    cells = [CellBlock("triangle", np.array([[0, 1, 2]]))]

    # Cas 1 : mesh MED (cell_tags) → gmsh
    mesh = meshio.Mesh(
        points, cells,
        cell_data={"cell_tags": [np.array([-1])]}
    )
    assert _pick_best_format(["ansys", "gmsh"], mesh) == "gmsh"

    # Cas 2 : mesh Gmsh (gmsh:physical) → gmsh
    mesh2 = meshio.Mesh(
        points, cells,
        cell_data={"gmsh:physical": [np.array([1])]}
    )
    assert _pick_best_format(["ansys", "gmsh"], mesh2) == "gmsh"

    # Cas 3 : mesh nu → défaut (ansys)
    mesh3 = meshio.Mesh(points, cells)
    assert _pick_best_format(["ansys", "gmsh"], mesh3) == "ansys"

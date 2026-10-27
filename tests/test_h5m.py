import numpy as np
import pytest

import meshlane

from . import helpers

h5py = pytest.importorskip("h5py")


@pytest.mark.parametrize("shape", [(1,), (1, 1), (1, 3)])
def test_write_cell_data(tmp_path, shape):
    data = np.arange(np.prod(shape), dtype=float).reshape(shape)
    mesh = meshlane.Mesh(
        helpers.tri_mesh_one_cell.points,
        helpers.tri_mesh_one_cell.cells,
        cell_data={"zero": [data], "material": [np.array([7])]},
    )
    path = tmp_path / "mesh.h5m"

    meshlane.write(path, mesh)

    with h5py.File(path, "r") as f:
        tags = f["tstt/elements/Tri3/tags"]
        np.testing.assert_array_equal(tags["zero"][()], data)
        np.testing.assert_array_equal(tags["material"][()], [7])


@pytest.mark.parametrize("skip_unsupported", [False, True])
def test_write_cell_data_multiple_blocks(tmp_path, skip_unsupported):
    cells = [
        ("line", [[0, 1], [1, 2]]),
        ("triangle", [[0, 1, 2]]),
        ("tetra", [[0, 1, 2, 3]]),
    ]
    values = [np.array([10, 11]), np.array([20]), np.array([30])]
    if skip_unsupported:
        cells.insert(1, ("quad", [[0, 1, 2, 3]]))
        values.insert(1, np.array([99]))
    mesh = meshlane.Mesh(
        helpers.tet_mesh.points,
        cells,
        cell_data={"material": values, "density": [v * 0.5 for v in values]},
    )
    path = tmp_path / "mesh.h5m"

    meshlane.write(path, mesh)

    with h5py.File(path, "r") as f:
        elements = f["tstt/elements"]
        assert set(elements) == {"Edge2", "Tri3", "Tet4"}
        for name, expected in [("Edge2", [10, 11]), ("Tri3", [20]), ("Tet4", [30])]:
            tags = elements[name]["tags"]
            np.testing.assert_array_equal(tags["material"][()], expected)
            np.testing.assert_array_equal(tags["density"][()], np.array(expected) * 0.5)


@pytest.mark.parametrize("missing_block", [0, 1])
def test_write_missing_cell_data(tmp_path, missing_block):
    values = [np.array([10, 11]), np.array([20, 21])]
    values[missing_block] = None
    mesh = meshlane.Mesh(
        helpers.line_tri_mesh.points,
        [("line", [[0, 1], [1, 2]]), ("triangle", [[0, 1, 2], [0, 2, 3]])],
        cell_data={"material": values},
    )
    path = tmp_path / "mesh.h5m"

    meshlane.write(path, mesh)

    with h5py.File(path, "r") as f:
        for index, name in enumerate(["Edge2", "Tri3"]):
            group = f["tstt/elements"][name]
            if index == missing_block:
                assert "tags" not in group
            else:
                np.testing.assert_array_equal(group["tags/material"][()], values[index])


@pytest.mark.parametrize(
    "mesh", [helpers.line_mesh, helpers.tri_mesh, helpers.tet_mesh, helpers.empty_mesh]
)
def test_write_without_cell_data(tmp_path, mesh):
    path = tmp_path / "mesh.h5m"

    meshlane.write(path, mesh)

    with h5py.File(path, "r") as f:
        for group in f["tstt/elements"].values():
            assert "tags" not in group
    helpers.write_read(tmp_path, meshlane.h5m.write, meshlane.h5m.read, mesh, 1.0e-15)

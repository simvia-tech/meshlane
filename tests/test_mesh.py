import copy

import numpy as np
import pytest
from numpy.testing import assert_equal

import meshlane

from . import helpers


def test_cells_dict():
    mesh = copy.deepcopy(helpers.tri_mesh)
    assert len(mesh.cells_dict) == 1
    assert np.array_equal(mesh.cells_dict["triangle"], [[0, 1, 2], [0, 2, 3]])

    # two cells groups
    mesh = meshlane.Mesh(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        [("triangle", [[0, 1, 2]]), ("triangle", [[0, 2, 3]])],
        cell_data={"a": [[0.5], [1.3]]},
    )
    assert len(mesh.cells_dict) == 1
    assert_equal(mesh.cells_dict, {"triangle": [[0, 1, 2], [0, 2, 3]]})
    assert_equal(mesh.cell_data_dict, {"a": {"triangle": [0.5, 1.3]}})


def test_sets_to_int_data():
    mesh = helpers.tri_mesh_5
    mesh = helpers.add_point_sets(mesh)
    mesh = helpers.add_cell_sets(mesh)

    mesh.point_sets_to_data()
    mesh.cell_sets_to_data()

    assert mesh.cell_sets == {}
    assert_equal(mesh.cell_data, {"grain0-grain1": [[0, 0, 1, 1, 1]]})

    assert mesh.point_sets == {}
    assert_equal(mesh.point_data, {"fixed-loose": [0, 0, 0, 1, 1, 1, 1]})

    # now back to set data
    mesh.cell_data_to_sets("grain0-grain1")
    mesh.point_data_to_sets("fixed-loose")

    assert mesh.cell_data == {}
    assert_equal(mesh.cell_sets, {"grain0": [[0, 1]], "grain1": [[2, 3, 4]]})

    assert mesh.point_data == {}
    assert_equal(mesh.point_sets, {"fixed": [0, 1, 2], "loose": [3, 4, 5, 6]})


@pytest.mark.skip
def test_sets_to_int_data_warning():
    mesh = meshlane.Mesh(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        {"triangle": [[0, 1, 2], [1, 2, 3]]},
        cell_sets={"tag": [[0]]},
    )
    with pytest.warns(UserWarning):
        mesh.cell_sets_to_data()
    assert np.all(mesh.cell_data["tag"] == np.array([[0, -1]]))

    mesh = meshlane.Mesh(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        {"triangle": [[0, 1, 2], [1, 2, 3]]},
        point_sets={"tag": [[0, 1, 3]]},
    )
    with pytest.warns(UserWarning):
        mesh.point_sets_to_data()

    assert np.all(mesh.point_data["tag"] == np.array([[0, 0, -1, 0]]))


def test_int_data_to_sets():
    mesh = helpers.tri_mesh
    mesh.cell_data = {"grain0-grain1": [np.array([0, 1])]}

    mesh.cell_data_to_sets("grain0-grain1")

    assert_equal(mesh.cell_sets, {"grain0": [[0]], "grain1": [[1]]})


def test_gh_1165():
    mesh = meshlane.Mesh(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        {
            "triangle": [[0, 1, 2], [1, 2, 3]],
            "line": [[0, 1], [0, 2], [1, 3], [2, 3]],
        },
        cell_sets={
            "test": [[], [1]],
            "sets": [[0, 1], [0, 2, 3]],
        },
    )

    mesh.cell_sets_to_data()
    mesh.cell_data_to_sets("test-sets")

    assert_equal(mesh.cell_sets, {"test": [[], [1]], "sets": [[0, 1], [0, 2, 3]]})


def test_copy():
    mesh = helpers.tri_mesh
    mesh2 = mesh.copy()

    assert np.all(mesh.points == mesh2.points)
    assert not np.may_share_memory(mesh.points, mesh2.points)


def test_remove_duplicate_cells():
    # cell 2 duplicates cell 0 (same node set, different node order)
    tri = np.array([[0, 1, 2], [1, 2, 3], [0, 2, 1]])
    pts = np.random.rand(4, 3)
    mesh = meshlane.Mesh(
        pts,
        [("triangle", tri)],
        cell_data={"val": [np.array([10, 20, 30])]},
        cell_sets={"A": [np.array([0, 2])], "B": [np.array([1])]},
    )
    n = mesh.remove_duplicate_cells()
    assert n == 1
    assert_equal(mesh.cells[0].data, np.array([[0, 1, 2], [1, 2, 3]]))
    # cell_data of the dropped cell is removed
    assert_equal(mesh.cell_data["val"][0], np.array([10, 20]))
    # set A loses the dropped duplicate (kept cell 0 -> new index 0)
    assert_equal(mesh.cell_sets["A"][0], np.array([0]))
    # set B: cell 1 -> new index 1
    assert_equal(mesh.cell_sets["B"][0], np.array([1]))


def test_remove_duplicate_cells_none():
    mesh = copy.deepcopy(helpers.tri_mesh)
    assert mesh.remove_duplicate_cells() == 0

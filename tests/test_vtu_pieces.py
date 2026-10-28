from xml.etree import ElementTree as ET

import numpy as np
import pytest

import meshlane


@pytest.mark.parametrize("piece_count", [1, 2, 3])
@pytest.mark.parametrize("with_data", [False, True])
@pytest.mark.parametrize("binary", [False, True])
def test_multiple_pieces(tmp_path, piece_count, with_data, binary):
    definitions = [
        (3, [("triangle", [[0, 1, 2]]), ("line", [[0, 2]])]),
        (4, [("quad", [[0, 1, 2, 3]]), ("line", [[1, 3]])]),
        (3, [("triangle", [[2, 1, 0]])]),
    ]
    expected_points = []
    expected_cells = []
    expected_scalar = []
    expected_vector = []
    root = None
    point_offset = 0
    for index, (point_count, cells) in enumerate(definitions[:piece_count]):
        points = np.arange(point_count * 3, dtype=float).reshape(-1, 3) + index * 100
        scalar = [np.array([index * 10 + block]) for block in range(len(cells))]
        vector = [np.array([[index, block, 7]]) for block in range(len(cells))]
        mesh = meshlane.Mesh(
            points,
            cells,
            point_data={"point_id": np.arange(point_count) + index * 10},
            cell_data={"scalar": scalar, "vector": vector} if with_data else {},
        )
        filename = tmp_path / f"piece-{index}.vtu"
        meshlane.write(filename, mesh, binary=binary, compression=None)
        piece_root = ET.parse(filename).getroot()
        if root is None:
            root = piece_root
        else:
            root.find("UnstructuredGrid").append(
                piece_root.find("UnstructuredGrid/Piece")
            )
        expected_points.append(points)
        expected_cells.extend(
            (name, np.array(nodes) + point_offset) for name, nodes in cells
        )
        expected_scalar.extend(scalar)
        expected_vector.extend(vector)
        point_offset += point_count

    filename = tmp_path / "combined.vtu"
    ET.ElementTree(root).write(filename)
    actual = meshlane.read(filename)
    np.testing.assert_array_equal(actual.points, np.concatenate(expected_points))
    np.testing.assert_array_equal(
        actual.point_data["point_id"],
        np.concatenate(
            [
                np.arange(count) + index * 10
                for index, (count, _) in enumerate(definitions[:piece_count])
            ]
        ),
    )
    assert [cell.type for cell in actual.cells] == [name for name, _ in expected_cells]
    for cell, (_, expected) in zip(actual.cells, expected_cells):
        np.testing.assert_array_equal(cell.data, expected)
    if with_data:
        for name, expected in [
            ("scalar", expected_scalar),
            ("vector", expected_vector),
        ]:
            assert len(actual.cell_data[name]) == len(expected)
            for values, reference in zip(actual.cell_data[name], expected):
                np.testing.assert_array_equal(values, reference)
    else:
        assert actual.cell_data == {}

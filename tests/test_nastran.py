import io
import pathlib

import numpy as np
import pytest

import meshlane

from . import helpers


@pytest.mark.parametrize(
    "mesh",
    [
        helpers.empty_mesh,
        helpers.tri_mesh,
        helpers.tri_mesh_2d,
        helpers.triangle6_mesh,
        helpers.quad_mesh,
        helpers.quad8_mesh,
        helpers.tri_quad_mesh,
        helpers.tet_mesh,
        helpers.tet10_mesh,
        helpers.hex_mesh,
        helpers.hex20_mesh,
    ],
)
def test(mesh, tmp_path):
    helpers.write_read(
        tmp_path, meshlane.nastran.write, meshlane.nastran.read, mesh, 1.0e-13
    )


@pytest.mark.parametrize("filename", ["cylinder.fem", "cylinder_cells_first.fem"])
def test_reference_file(filename, tmp_path):
    this_dir = pathlib.Path(__file__).resolve().parent
    filename = this_dir / "meshes" / "nastran" / filename

    mesh = meshlane.read(filename)

    # points
    assert np.isclose(mesh.points.sum(), 16.5316866)

    # cells
    ref_num_cells = {
        "line": 241,
        "triangle": 171,
        "quad": 721,
        "pyramid": 1180,
        "tetra": 5309,
    }
    assert {
        cell_block.type: cell_block.data.sum() for cell_block in mesh.cells
    } == ref_num_cells

    # References must stay aligned even when a whole block has blank PIDs (#43).
    assert [len(refs) for refs in mesh.cell_data["nastran:ref"]] == [
        len(block) for block in mesh.cells
    ]
    converted_path = tmp_path / "cylinder.med"
    meshlane.write(converted_path, mesh)
    converted = meshlane.read(converted_path)
    np.testing.assert_allclose(converted.points, mesh.points)
    assert converted.cells_dict.keys() == mesh.cells_dict.keys()
    for cell_type, connectivity in mesh.cells_dict.items():
        np.testing.assert_array_equal(converted.cells_dict[cell_type], connectivity)
    expected_refs = dict(
        zip([cell.type for cell in mesh.cells], mesh.cell_data["nastran:ref"])
    )
    # MED exposes these as per-block fields with generated names.
    for index, cell in enumerate(converted.cells):
        refs = [
            values[index]
            for name, values in converted.cell_data.items()
            if name.startswith("nastran:ref") and values[index] is not None
        ]
        assert len(refs) == 1
        np.testing.assert_array_equal(np.atleast_1d(refs[0]), expected_refs[cell.type])


@pytest.mark.parametrize("field_format", ["free", "small", "large"])
@pytest.mark.parametrize("references", ["blank", "explicit", "mixed"])
def test_blank_references(field_format, references):
    if references == "blank":
        point_refs, cell_refs = [""] * 4, [""] * 5
    elif references == "explicit":
        point_refs, cell_refs = [0, 3, 4, 0], [3, 4, 6, 6, 5]
    else:
        point_refs, cell_refs = ["", 7, 0, ""], ["", 9, "", "", 7]

    def card(fields):
        if field_format == "free":
            return ",".join(map(str, fields)) + "\n"
        return "".join(f"{value!s:<8}" for value in fields) + "\n"

    lines = ["BEGIN BULK\n"]
    points = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    for node, pref, (x, y, z) in zip([10, 20, 30, 40], point_refs, points):
        if field_format == "large":
            lines.append(f"{'GRID*':<8}{node:<16}{pref!s:<16}{x:<16}{y:<16}\n")
            lines.append(f"{'*':<8}{z:<16}\n")
        else:
            lines.append(card(["GRID", node, pref, x, y, z]))
    for index, node_ids in enumerate(
        [[10, 20], [20, 30], [10, 20, 30], [20, 30, 40], [30, 40]]
    ):
        keyword = "CBAR" if len(node_ids) == 2 else "CTRIA3"
        orientation = [0, 0, 1] if keyword == "CBAR" else []
        lines.append(
            card([keyword, 101 + index, cell_refs[index], *node_ids, *orientation])
        )
    lines.append("ENDDATA\n")
    mesh = meshlane.read(io.StringIO("".join(lines)), "nastran")

    np.testing.assert_array_equal(
        mesh.point_data["nastran:ref"], [int(p or 0) for p in point_refs]
    )
    expected = np.array(
        [int(p) if p != "" else 101 + i for i, p in enumerate(cell_refs)]
    )
    assert [cell.type for cell in mesh.cells] == ["line", "triangle", "line"]
    for actual, wanted in zip(
        mesh.cell_data["nastran:ref"], np.split(expected, [2, 4])
    ):
        np.testing.assert_array_equal(actual, wanted)
    # Rebuilding invokes the same metadata length checks used by converters.
    meshlane.Mesh(
        mesh.points, mesh.cells, point_data=mesh.point_data, cell_data=mesh.cell_data
    )


def test_long_format():
    filename = io.StringIO(
        "BEGIN BULK\n"
        "GRID*    43                             1.50000000000000 0.0\n"
        "*        0.\n"
        "ENDDATA\n"
    )

    mesh = meshlane.read(filename, "nastran")

    # points
    assert len(mesh.points) == 1
    assert np.isclose(mesh.points.sum(), 1.5)

import pathlib

import numpy as np
import pytest

import meshlane

from . import helpers


@pytest.mark.parametrize(
    "mesh",
    [
        helpers.empty_mesh,
        helpers.line_mesh,
        helpers.tri_mesh,
        helpers.tri_mesh_2d,
        helpers.quad_mesh,
        helpers.tri_quad_mesh,
        helpers.tet_mesh,
        helpers.hex_mesh,
        helpers.add_cell_data(helpers.tri_mesh, [("medit:ref", (), int)]),
    ],
)
def test_io(mesh, tmp_path):
    helpers.write_read(
        tmp_path, meshlane.medit.write, meshlane.medit.read, mesh, 1.0e-15
    )


def test_generic_io(tmp_path):
    helpers.generic_io(tmp_path / "test.mesh")
    # With additional, insignificant suffix:
    helpers.generic_io(tmp_path / "test.0.mesh")
    # same for binary files
    helpers.generic_io(tmp_path / "test.meshb")
    helpers.generic_io(tmp_path / "test.0.meshb")


@pytest.mark.parametrize("extension", [".mesh", ".meshb"])
@pytest.mark.parametrize("column", [False, True])
@pytest.mark.parametrize("single_cell", [False, True])
@pytest.mark.parametrize("location", ["point", "cell", "both"])
def test_scalar_reference_labels(tmp_path, extension, column, single_cell, location):
    base = helpers.tri_mesh_one_cell if single_cell else helpers.line_tri_mesh
    point_labels = np.arange(len(base.points), dtype=np.int32)
    cell_labels = [np.arange(len(block), dtype=np.int32) + 10 for block in base.cells]
    mesh = meshlane.Mesh(
        base.points,
        base.cells,
        point_data={
            "medit:ref": (
                point_labels[:, None]
                if column and location in ("point", "both")
                else point_labels
            )
        },
        cell_data={
            "medit:ref": [
                values[:, None] if column and location in ("cell", "both") else values
                for values in cell_labels
            ]
        },
    )
    path = tmp_path / ("labels" + extension)

    meshlane.write(path, mesh)
    result = meshlane.read(path)

    np.testing.assert_array_equal(result.point_data["medit:ref"], point_labels)
    for actual, expected in zip(result.cell_data["medit:ref"], cell_labels):
        np.testing.assert_array_equal(actual, expected)
    assert mesh.point_data["medit:ref"].shape == (
        (len(point_labels), 1)
        if column and location in ("point", "both")
        else (len(point_labels),)
    )
    for actual, expected in zip(mesh.cell_data["medit:ref"], cell_labels):
        assert actual.shape == (
            (len(expected), 1)
            if column and location in ("cell", "both")
            else (len(expected),)
        )


@pytest.mark.parametrize("extension", [".mesh", ".meshb"])
@pytest.mark.parametrize("location", ["point", "cell"])
def test_reject_multicomponent_reference_labels(tmp_path, extension, location):
    mesh = meshlane.Mesh(helpers.tri_mesh.points, helpers.tri_mesh.cells)
    if location == "point":
        mesh.point_data["medit:ref"] = np.ones((len(mesh.points), 2), dtype=int)
    else:
        mesh.cell_data["medit:ref"] = [
            np.ones((len(block), 2), dtype=int) for block in mesh.cells
        ]

    with pytest.raises(meshlane.WriteError, match="reference labels must be scalar"):
        meshlane.write(tmp_path / ("labels" + extension), mesh)


# same tests with ugrid format files converted with UGC from
# https://www.simcenter.msstate.edu/


@pytest.mark.parametrize(
    "filename, ref_num_points, ref_num_triangle, ref_num_quad, ref_num_wedge, ref_num_tet, ref_num_hex, ref_tag_counts",
    [
        (
            "sphere_mixed.1.meshb",
            3270,
            864,
            0,
            3024,
            9072,
            0,
            {1: 432, 2: 216, 3: 216},
        ),
        ("hch_strct.4.meshb", 306, 12, 178, 96, 0, 144, {1: 15, 2: 15, 3: 160}),
        ("hch_strct.4.be.meshb", 306, 12, 178, 96, 0, 144, {1: 15, 2: 15, 3: 160}),
        ("cube86.mesh", 39, 72, 0, 0, 86, 0, {1: 14, 2: 14, 3: 14, 4: 8, 5: 14, 6: 8}),
    ],
)
def test_reference_file(
    filename,
    ref_num_points,
    ref_num_triangle,
    ref_num_quad,
    ref_num_wedge,
    ref_num_tet,
    ref_num_hex,
    ref_tag_counts,
):
    this_dir = pathlib.Path(__file__).resolve().parent
    filename = this_dir / "meshes" / "medit" / filename

    mesh = meshlane.read(filename)
    assert mesh.points.shape[0] == ref_num_points
    assert mesh.points.shape[1] == 3

    medit_meshio_id = {
        "triangle": None,
        "quad": None,
        "tetra": None,
        "pyramid": None,
        "wedge": None,
        "hexahedron": None,
    }

    for i, cell_block in enumerate(mesh.cells):
        if cell_block.type in medit_meshio_id:
            medit_meshio_id[cell_block.type] = i

    # validate element counts
    if ref_num_triangle > 0:
        c = mesh.cells[medit_meshio_id["triangle"]]
        assert c.data.shape == (ref_num_triangle, 3)
    else:
        assert medit_meshio_id["triangle"] is None

    if ref_num_quad > 0:
        c = mesh.cells[medit_meshio_id["quad"]]
        assert c.data.shape == (ref_num_quad, 4)
    else:
        assert medit_meshio_id["quad"] is None

    if ref_num_tet > 0:
        c = mesh.cells[medit_meshio_id["tetra"]]
        assert c.data.shape == (ref_num_tet, 4)
    else:
        assert medit_meshio_id["tetra"] is None

    if ref_num_wedge > 0:
        c = mesh.cells[medit_meshio_id["wedge"]]
        assert c.data.shape == (ref_num_wedge, 6)
    else:
        assert medit_meshio_id["wedge"] is None

    if ref_num_hex > 0:
        c = mesh.cells[medit_meshio_id["hexahedron"]]
        assert c.data.shape == (ref_num_hex, 8)
    else:
        assert medit_meshio_id["hexahedron"] is None

    # validate boundary tags

    # gather tags
    all_tags = []
    for k, c in enumerate(mesh.cells):
        if c.type not in ["triangle", "quad"]:
            continue
        all_tags.append(mesh.cell_data["medit:ref"][k])

    all_tags = np.concatenate(all_tags)

    # validate against known values
    unique, counts = np.unique(all_tags, return_counts=True)
    tags = dict(zip(unique, counts))
    assert tags.keys() == ref_tag_counts.keys()
    for key in tags.keys():
        assert tags[key] == ref_tag_counts[key]

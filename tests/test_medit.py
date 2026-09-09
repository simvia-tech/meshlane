import copy
import pathlib
import textwrap

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
        # higher-order cells
        helpers.line3_mesh,
        helpers.line4_mesh,
        helpers.line5_mesh,
        helpers.triangle6_mesh,
        helpers.triangle10_mesh,
        helpers.triangle15_mesh,
        helpers.quad9_mesh,
        helpers.tet10_mesh,
        helpers.wedge18_mesh,
        helpers.hex27_mesh,
        helpers.pyramid14_mesh,
        helpers.add_cell_data(helpers.tri_mesh, [("medit:ref", (), int)]),
    ],
)
@pytest.mark.parametrize("extension", [".mesh", ".meshb"])
def test_io(mesh, extension, tmp_path):
    helpers.write_read(
        tmp_path,
        meshlane.medit.write,
        meshlane.medit.read,
        mesh,
        1.0e-15,
        extension=extension,
    )


def test_read_ascii_variants(tmp_path):
    """MeshVersionFormatted 3, SubDomainFromGeom and an uppercase END used to raise; Hexaedra (Dobrzynski's spelling) already worked."""
    path = tmp_path / "variants.mesh"
    path.write_text(textwrap.dedent("""\
        MeshVersionFormatted 3
        Dimension
        3
        Vertices
        8
        0 0 0 1
        1 0 0 1
        1 1 0 1
        0 1 0 1
        0 0 1 1
        1 0 1 1
        1 1 1 1
        0 1 1 1
        Hexaedra
        1
        1 2 3 4 5 6 7 8 4
        SubDomainFromGeom
        1
        3 1 0 0
        END
        """))

    mesh = meshlane.medit.read(path)

    assert mesh.points.shape == (8, 3)
    assert len(mesh.cells) == 1
    assert mesh.cells[0].type == "hexahedron"
    assert np.array_equal(mesh.cells[0].data, [np.arange(8)])
    assert np.array_equal(mesh.cell_data["medit:ref"][0], [4])


def test_read_ascii_skips_ordering_sections(tmp_path):
    """<Keyword>Ordering sections are skipped metadata; TrianglesP2Ordering also checks the keyword-with-digits guard doesn't misfire."""
    path = tmp_path / "ordering.mesh"
    path.write_text(textwrap.dedent("""\
        MeshVersionFormatted 2
        Dimension
        2
        Vertices
        6
        0 0 1
        1 0 1
        0 1 1
        0.5 0 1
        0.5 0.5 1
        0 0.5 1
        TrianglesP2
        1
        1 2 3 4 5 6 1
        TrianglesP2Ordering
        6
        2 0 0
        0 2 0
        0 0 2
        1 1 0
        0 1 1
        1 0 1
        End
        """))

    mesh = meshlane.medit.read(path)

    assert len(mesh.cells) == 1
    assert mesh.cells[0].type == "triangle6"
    assert np.array_equal(mesh.cells[0].data, [np.arange(6)])


def test_read_ascii_rejects_desynchronised_stream(tmp_path):
    """A line that is not a keyword means a section was read with the wrong cell size."""
    path = tmp_path / "broken.mesh"
    path.write_text(textwrap.dedent("""\
        MeshVersionFormatted 2
        Dimension
        2
        Vertices
        1
        0 0 1
        1 2 3
        End
        """))

    with pytest.raises(meshlane.ReadError, match="Expected a keyword"):
        meshlane.medit.read(path)


@pytest.mark.parametrize("mesh", [helpers.quad8_mesh, helpers.hex20_mesh])
@pytest.mark.parametrize("extension", [".mesh", ".meshb"])
def test_cells_without_medit_equivalent_are_skipped(mesh, extension, tmp_path, capsys):
    """quad8, wedge15 and hexahedron20 have no Medit keyword."""
    path = tmp_path / f"test{extension}"
    meshlane.medit.write(path, mesh)
    assert "doesn't know" in capsys.readouterr().err
    assert len(meshlane.medit.read(path).cells) == 0


@pytest.mark.parametrize("extension", [".mesh", ".meshb"])
def test_write_does_not_modify_input_mesh(extension, tmp_path):
    """Writing must not reorder the caller's cells (hexahedron27 is permuted on write)."""
    mesh = copy.deepcopy(helpers.hex27_mesh)
    before = mesh.cells[0].data.copy()
    meshlane.medit.write(tmp_path / f"test{extension}", mesh)
    assert np.array_equal(before, mesh.cells[0].data)


def test_generic_io(tmp_path):
    helpers.generic_io(tmp_path / "test.mesh")
    # With additional, insignificant suffix:
    helpers.generic_io(tmp_path / "test.0.mesh")
    # same for binary files
    helpers.generic_io(tmp_path / "test.meshb")
    helpers.generic_io(tmp_path / "test.0.meshb")


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


@pytest.mark.parametrize(
    "filename, cell_type, num_nodes_per_cell",
    [
        ("triangle_p1.mesh", "triangle", 3),
        ("triangle_p2.mesh", "triangle6", 6),
        ("triangle_p3.mesh", "triangle10", 10),
        ("triangle_p4.mesh", "triangle15", 15),
        ("quad_q1.mesh", "quad", 4),
        ("quad_q2.mesh", "quad9", 9),
        ("tetra_p1.mesh", "tetra", 4),
        ("tetra_p2.mesh", "tetra10", 10),
        ("prism_p1.mesh", "wedge", 6),
        ("prism_p2.mesh", "wedge18", 18),
        ("hex_q1.mesh", "hexahedron", 8),
        ("hex_q2.mesh", "hexahedron27", 27),
        ("pyramid_p1.mesh", "pyramid", 5),
        ("pyramid_p2.mesh", "pyramid14", 14),
    ],
)
@pytest.mark.parametrize("extension", [".mesh", ".meshb"])
def test_vizir_reference_element(filename, cell_type, num_nodes_per_cell, extension, tmp_path):
    """ViZiR4 reference elements (https://pyamg.saclay.inria.fr/vizir4examples.html, Inria); round-tripping catches writer permutation bugs like hexahedron27's that a shape-only check would miss."""
    this_dir = pathlib.Path(__file__).resolve().parent
    mesh = meshlane.read(this_dir / "meshes" / "medit" / filename)

    matches = [c for c in mesh.cells if c.type == cell_type]
    assert len(matches) == 1
    assert matches[0].data.shape == (1, num_nodes_per_cell)

    roundtrip_path = tmp_path / f"roundtrip{extension}"
    meshlane.medit.write(roundtrip_path, mesh)
    roundtrip_mesh = meshlane.medit.read(roundtrip_path)

    roundtrip_matches = [c for c in roundtrip_mesh.cells if c.type == cell_type]
    assert len(roundtrip_matches) == 1
    assert np.array_equal(roundtrip_matches[0].data, matches[0].data)
    assert np.allclose(roundtrip_mesh.points, mesh.points)

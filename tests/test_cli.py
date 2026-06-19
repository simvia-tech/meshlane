import numpy as np

import meshlane

from . import helpers


def is_same_mesh(mesh0, mesh1, atol):
    if not np.allclose(mesh0.points, mesh1.points, atol=atol, rtol=0.0):
        return False
    for cells0, cells1 in zip(mesh0.cells, mesh1.cells):
        if cells0.type != cells1.type or not np.allclose(cells0.data, cells1.data):
            return False
    return True


def test_info(tmp_path):
    infile = tmp_path / "out.msh"
    meshlane.write(infile, helpers.tri_mesh, file_format="gmsh")
    meshlane._cli.main(["info", str(infile), "--input-format", "gmsh"])


def test_convert(tmp_path):
    input_mesh = helpers.tri_mesh

    infile = tmp_path / "in.msh"
    meshlane.write(infile, helpers.tri_mesh, file_format="gmsh")

    outfile = tmp_path / "out.msh"
    meshlane._cli.main(
        [
            "convert",
            str(infile),
            str(outfile),
            "--input-format",
            "gmsh",
            "--output-format",
            "vtk",
            "--sets-to-int-data",
        ]
    )

    mesh = meshlane.read(outfile, file_format="vtk")

    atol = 1.0e-15
    assert np.allclose(input_mesh.points, mesh.points, atol=atol, rtol=0.0)

    for cells0, cells1 in zip(input_mesh.cells, mesh.cells):
        assert cells0.type == cells1.type
        assert np.allclose(cells0.data, cells1.data)


def test_compress(tmp_path):
    input_mesh = helpers.tri_mesh

    infile = tmp_path / "in.vtu"
    meshlane.write(infile, input_mesh)

    meshlane._cli.main(["decompress", str(infile)])
    mesh = meshlane.read(infile)
    assert is_same_mesh(input_mesh, mesh, atol=1.0e-15)

    meshlane._cli.main(["compress", str(infile)])
    mesh = meshlane.read(infile)
    assert is_same_mesh(input_mesh, mesh, atol=1.0e-15)


def test_ascii_binary(tmp_path):
    input_mesh = helpers.tri_mesh

    infile = tmp_path / "in.vtu"
    meshlane.write(infile, input_mesh)

    meshlane._cli.main(["ascii", str(infile)])
    mesh = meshlane.read(infile)
    assert is_same_mesh(input_mesh, mesh, atol=1.0e-12)

    meshlane._cli.main(["binary", str(infile)])
    mesh = meshlane.read(infile)
    assert is_same_mesh(input_mesh, mesh, atol=1.0e-12)

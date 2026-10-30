import gzip
import io

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
        # helpers.triangle6_mesh,
        helpers.quad_mesh,
        helpers.quad8_mesh,
        helpers.tri_quad_mesh,
        helpers.tet_mesh,
        # helpers.tet10_mesh,
        helpers.hex_mesh,
        helpers.hex20_mesh,
    ],
)
def test_io(mesh, tmp_path):
    helpers.write_read(
        tmp_path, meshlane.permas.write, meshlane.permas.read, mesh, 1.0e-15
    )


def test_generic_io(tmp_path):
    helpers.generic_io(tmp_path / "test.post")
    helpers.generic_io(tmp_path / "test.post.gz")
    # With additional, insignificant suffix:
    helpers.generic_io(tmp_path / "test.0.post")
    helpers.generic_io(tmp_path / "test.0.post.gz")


@pytest.mark.parametrize("extension", [".post.gz", ".dato.gz", ".0.post.gz"])
@pytest.mark.parametrize("as_string", [False, True])
def test_gzip_write(tmp_path, extension, as_string):
    filename = tmp_path / ("mesh" + extension)
    plain = io.StringIO()
    meshlane.permas.write(plain, helpers.tri_mesh)

    meshlane.write(str(filename) if as_string else filename, helpers.tri_mesh)

    data = filename.read_bytes()
    assert data[:2] == b"\x1f\x8b"
    assert gzip.decompress(data).decode() == plain.getvalue()


@pytest.mark.parametrize("extension", [".post.gz", ".dato.gz", ".0.post.gz"])
@pytest.mark.parametrize("as_string", [False, True])
def test_gzip_read(tmp_path, extension, as_string):
    # Compress independently so the reader cannot hide an uncompressed writer.
    plain = io.StringIO()
    meshlane.permas.write(plain, helpers.tri_mesh)
    filename = tmp_path / ("mesh" + extension)
    filename.write_bytes(gzip.compress(plain.getvalue().encode()))

    mesh = meshlane.read(str(filename) if as_string else filename)

    np.testing.assert_allclose(mesh.points, helpers.tri_mesh.points)
    assert len(mesh.cells) == 1
    assert mesh.cells[0].type == "triangle"
    np.testing.assert_array_equal(mesh.cells[0].data, helpers.tri_mesh.cells[0].data)


def test_text_buffer():
    buffer = io.StringIO()
    meshlane.write(buffer, helpers.tri_mesh, file_format="permas")
    assert not buffer.closed
    buffer.seek(0)
    mesh = meshlane.read(buffer, file_format="permas")
    assert not buffer.closed
    np.testing.assert_allclose(mesh.points, helpers.tri_mesh.points)
    np.testing.assert_array_equal(mesh.cells[0].data, helpers.tri_mesh.cells[0].data)

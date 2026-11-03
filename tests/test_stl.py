import gzip
import io
import struct
from contextlib import nullcontext

import pytest

import meshlane

from . import helpers


@pytest.mark.parametrize(
    "mesh",
    [helpers.empty_mesh, helpers.tri_mesh],
)
@pytest.mark.parametrize(
    "binary, tol",
    [
        (False, 1.0e-15),
        # binary STL only operates in single precision
        (True, 1.0e-8),
    ],
)
def test_stl(mesh, binary, tol, tmp_path):
    def writer(*args, **kwargs):
        return meshlane.stl.write(*args, binary=binary, **kwargs)

    helpers.write_read(tmp_path, writer, meshlane.stl.read, mesh, tol)


@pytest.mark.parametrize("mesh", [helpers.empty_mesh, helpers.tri_mesh])
@pytest.mark.parametrize("compressed", [False, True])
def test_binary_stream(mesh, compressed):
    buffer = io.BytesIO()
    context = (
        gzip.GzipFile(fileobj=buffer, mode="wb") if compressed else nullcontext(buffer)
    )
    with context as stream:
        meshlane.write(stream, mesh, file_format="stl", binary=True)
        assert not stream.closed
    assert not buffer.closed
    data = buffer.getvalue()
    if compressed:
        data = gzip.decompress(data)

    triangles = mesh.get_cells_type("triangle")
    assert struct.unpack_from("<I", data, 80)[0] == len(triangles)
    assert len(data) == 84 + 50 * len(triangles)
    for i, triangle in enumerate(triangles):
        record = struct.unpack_from("<12fH", data, 84 + 50 * i)
        assert record[:3] == (0.0, 0.0, 1.0)
        assert record[3:12] == pytest.approx(mesh.points[triangle].ravel())
        assert record[12] == 0


def test_binary_gzip_file(tmp_path):
    plain = tmp_path / "mesh.stl"
    compressed = tmp_path / "mesh.stl.gz"
    meshlane.write(plain, helpers.tri_mesh, binary=True)
    with gzip.open(compressed, "wb") as stream:
        meshlane.write(stream, helpers.tri_mesh, file_format="stl", binary=True)
        assert not stream.closed
    # A real gzip file has fileno(), but writes must still pass through gzip.
    assert gzip.decompress(compressed.read_bytes()) == plain.read_bytes()

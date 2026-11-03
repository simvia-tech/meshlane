import base64
import pathlib
import struct

import numpy as np
import pytest

import meshlane
from meshlane.vtu._vtu import VtuReader

from . import helpers

test_set = [
    # helpers.empty_mesh,
    helpers.line_mesh,
    helpers.tri_mesh,
    helpers.tri_mesh_one_cell,
    helpers.triangle6_mesh,
    helpers.quad_mesh,
    helpers.quad8_mesh,
    helpers.tri_quad_mesh,
    helpers.polygon_mesh,
    helpers.polygon_mesh_one_cell,
    helpers.polygon2_mesh,
    helpers.tet_mesh,
    helpers.tet10_mesh,
    helpers.hex_mesh,
    helpers.hex20_mesh,
    helpers.pyramid_mesh,
    helpers.wedge_mesh,
    helpers.polyhedron_mesh,
    helpers.lagrange_high_order_mesh,
    helpers.add_point_data(helpers.tri_mesh, 1),
    helpers.add_point_data(helpers.tri_mesh, 2),
    helpers.add_point_data(helpers.tri_mesh, 3),
    helpers.add_cell_data(helpers.tri_mesh, [("a", (), np.float64)]),
    helpers.add_cell_data(helpers.tri_quad_mesh, [("a", (), np.float64)]),
    helpers.add_cell_data(helpers.tri_mesh, [("a", (2,), np.float32)]),
    helpers.add_cell_data(helpers.tri_mesh, [("b", (3,), np.float64)]),
    helpers.add_cell_data(helpers.polygon_mesh, [("a", (), np.float32)]),
    helpers.add_cell_data(helpers.polyhedron_mesh, [("a", (2,), np.float32)]),
]


@pytest.mark.parametrize("mesh", test_set)
@pytest.mark.parametrize(
    "data_type", [(False, None), (True, None), (True, "lzma"), (True, "zlib")]
)
def test(mesh, data_type, tmp_path):
    binary, compression = data_type

    def writer(*args, **kwargs):
        return meshlane.vtu.write(
            *args, binary=binary, compression=compression, **kwargs
        )

    # ASCII files are only meant for debugging, VTK stores only 11 digits
    # <https://gitlab.kitware.com/vtk/vtk/-/issues/17038#note_264052>
    tol = 1.0e-15 if binary else 1.0e-10
    helpers.write_read(tmp_path, writer, meshlane.vtu.read, mesh, tol)


def test_generic_io(tmp_path):
    helpers.generic_io(tmp_path / "test.vtu")
    # With additional, insignificant suffix:
    helpers.generic_io(tmp_path / "test.0.vtu")


@pytest.mark.parametrize(
    "filename, ref_cells, ref_num_cells, ref_num_pnt",
    [
        ("00_raw_binary.vtu", "tetra", 162, 64),
        ("01_raw_binary_int64.vtu", "tetra", 162, 64),
        ("02_raw_compressed.vtu", "tetra", 162, 64),
    ],
)
def test_read_from_file(filename, ref_cells, ref_num_cells, ref_num_pnt):
    this_dir = pathlib.Path(__file__).resolve().parent
    filename = this_dir / "meshes" / "vtu" / filename

    mesh = meshlane.read(filename)
    assert len(mesh.cells) == 1
    assert ref_cells == mesh.cells[0].type
    assert len(mesh.cells[0].data) == ref_num_cells
    assert len(mesh.points) == ref_num_pnt


@pytest.mark.parametrize("header_type, code", [("UInt32", "I"), ("UInt64", "Q")])
@pytest.mark.parametrize(
    "byte_order, prefix", [("LittleEndian", "<"), ("BigEndian", ">")]
)
@pytest.mark.parametrize("separate_header", [False, True])
@pytest.mark.parametrize("count", [0, 1, 2, 3, 4, 5])
def test_appended_array_boundaries(
    header_type, code, byte_order, prefix, separate_header, count
):
    reader = VtuReader.__new__(VtuReader)
    reader.header_type = header_type
    reader.byte_order = byte_order
    expected = np.arange(count, dtype=np.uint8)
    header = struct.pack(prefix + code, count)
    if separate_header:
        encoded = base64.b64encode(header) + base64.b64encode(expected.tobytes())
    else:
        encoded = base64.b64encode(header + expected.tobytes())
    # A later independently encoded array must not be decoded with this one.
    following = base64.b64encode(struct.pack(prefix + code, 1) + b"\xff")
    data = (encoded + following).decode()
    if count == 5:
        data = " \n".join(data[index : index + 5] for index in range(0, len(data), 5))
    actual = reader.read_uncompressed_binary(data, np.dtype(np.uint8))
    np.testing.assert_array_equal(actual, expected)

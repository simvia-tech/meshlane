import xml.etree.ElementTree as ET

import numpy as np
import pytest

import meshlane

from . import helpers


@pytest.mark.parametrize(
    "binary, compression", [(False, None), (True, None), (True, "lzma"), (True, "zlib")]
)
@pytest.mark.parametrize("header_type", ["UInt32", "UInt64"])
def test_field_data(tmp_path, binary, compression, header_type):
    field_data = {
        "physical group": np.array([42, 2], dtype=">i4"),
        "time": np.array([1.25], dtype=np.float32),
        "vectors": np.arange(15000, dtype=np.float64).reshape(-1, 3) / 7,
    }
    mesh = meshlane.Mesh(
        helpers.tri_mesh.points,
        helpers.tri_mesh.cells,
        field_data=field_data.copy(),
        point_data={"time": np.arange(4, dtype=np.float32)},
        cell_data={"time": [np.arange(2, dtype=np.float32)]},
    )
    filename = tmp_path / "field-data.vtu"
    meshlane.vtu.write(
        filename, mesh, binary=binary, compression=compression, header_type=header_type
    )

    grid = ET.parse(filename).getroot().find("UnstructuredGrid")
    arrays = grid.findall("FieldData/DataArray")
    assert len(arrays) == len(field_data)
    assert grid.find("Piece/FieldData") is None
    for array in arrays:
        data = field_data[array.attrib["Name"]]
        assert int(array.attrib["NumberOfTuples"]) == len(data)
        components = data.shape[1] if data.ndim == 2 else 1
        assert int(array.attrib.get("NumberOfComponents", "1")) == components
        assert array.attrib["format"] == ("binary" if binary else "ascii")

    loaded = meshlane.vtu.read(filename)
    assert loaded.field_data.keys() == field_data.keys()
    for name, data in field_data.items():
        actual = loaded.field_data[name]
        np.testing.assert_allclose(actual, data, rtol=1e-10)
        assert actual.shape == data.shape
        assert actual.dtype == data.dtype.newbyteorder("=")
    np.testing.assert_array_equal(loaded.points, mesh.points)
    np.testing.assert_array_equal(loaded.cells[0].data, mesh.cells[0].data)
    np.testing.assert_array_equal(loaded.point_data["time"], mesh.point_data["time"])
    np.testing.assert_array_equal(
        loaded.cell_data["time"][0], mesh.cell_data["time"][0]
    )

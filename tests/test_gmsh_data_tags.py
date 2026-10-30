import numpy as np
import pytest

import meshlane

POINTS = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
CELLS = [(2, [0, 1, 2]), (1, [2, 3]), (2, [1, 3, 2])]


def write_tagged_mesh(path, version, binary, sparse, data_first=False, bad_tags=None):
    point_tags = np.array([30, 10, 40, 20] if sparse else [1, 2, 3, 4])
    element_tags = np.array([70, 50, 90] if sparse else [1, 2, 3])
    with path.open("wb") as f:

        def text(s):
            f.write(s.encode())

        def numbers(values, dtype):
            values = np.asarray(values, dtype=dtype)
            if binary:
                f.write(values.tobytes())
            else:
                text(" ".join(map(str, values.ravel())) + "\n")

        def data():
            for kind, tags in [("NodeData", point_tags), ("ElementData", element_tags)]:
                for components in [1, 3, 9]:
                    missing = bad_tags == (kind, "missing")
                    count = len(tags) - int(missing)
                    text(
                        f'${kind}\n1\n"value{components}"\n1\n0.0\n3\n0\n{components}\n{count}\n'
                    )
                    # The independently defined value is tied to the mesh index,
                    # while the records deliberately arrive in reverse order.
                    for i in reversed(range(len(tags))):
                        if missing and i == 0:
                            continue
                        tag = tags[i]
                        if bad_tags and kind == bad_tags[0] and i == 0:
                            tag = tags[1] if bad_tags[1] == "duplicate" else 999
                        values = [10 * (i + 1) + j for j in range(components)]
                        if binary:
                            numbers([tag], "i4")
                            numbers(values, "f8")
                        else:
                            text(" ".join(map(str, [tag, *values])) + "\n")
                    text(f"\n$End{kind}\n")

        text(f"$MeshFormat\n{version} {int(binary)} 8\n")
        if binary:
            numbers([1], "i4")
            text("\n")
        text("$EndMeshFormat\n")
        if data_first:
            data()
        text("$Nodes\n")
        if version == "2.2":
            text("4\n")
        else:
            header = (
                [1, 4] if version == "4.0" else [1, 4, min(point_tags), max(point_tags)]
            )
            numbers(header, "L" if version == "4.0" else "u8")
            if version == "4.0" and not binary:
                text("1 2 0 4\n")
            else:
                numbers([1, 2, 0] if version == "4.0" else [2, 1, 0], "i4")
                numbers([4], "L" if version == "4.0" else "u8")
        if version == "4.1":
            numbers(point_tags, "u8")
            numbers(POINTS, "f8")
        else:
            for tag, point in zip(point_tags, POINTS):
                if binary:
                    numbers([tag], "i4")
                    numbers(point, "f8")
                else:
                    text(" ".join(map(str, [tag, *point])) + "\n")
        text("\n$EndNodes\n$Elements\n")
        if version == "2.2":
            text("3\n")
        else:
            header = (
                [3, 3]
                if version == "4.0"
                else [3, 3, min(element_tags), max(element_tags)]
            )
            numbers(header, "L" if version == "4.0" else "u8")
        for tag, (element_type, indices) in zip(element_tags, CELLS):
            nodes = point_tags[indices]
            if version == "2.2":
                if binary:
                    numbers([element_type, 1, 0], "i4")
                    numbers([tag, *nodes], "i4")
                else:
                    numbers([tag, element_type, 0, *nodes], "i4")
            else:
                dim = 1 if element_type == 1 else 2
                numbers(
                    (
                        [1, dim, element_type]
                        if version == "4.0"
                        else [dim, 1, element_type]
                    ),
                    "i4",
                )
                numbers([1], "L" if version == "4.0" else "u8")
                numbers([tag, *nodes], "i4" if version == "4.0" else "u8")
        text("\n$EndElements\n")
        if not data_first:
            data()


@pytest.mark.parametrize("version", ["2.2", "4.0", "4.1"])
@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("data_first", [False, True])
def test_data_follows_node_and_element_tags(
    tmp_path, version, binary, sparse, data_first
):
    path = tmp_path / "tagged.msh"
    write_tagged_mesh(path, version, binary, sparse, data_first)
    mesh = meshlane.read(path)
    np.testing.assert_array_equal(mesh.points, POINTS)
    for block, (_, indices) in zip(mesh.cells, CELLS):
        np.testing.assert_array_equal(block.data, [indices])
    for components in [1, 3, 9]:
        expected = 10 * np.arange(1, 5)[:, None] + np.arange(components)
        np.testing.assert_array_equal(
            mesh.point_data[f"value{components}"],
            expected[:, 0] if components == 1 else expected,
        )
        for i, block in enumerate(mesh.cell_data[f"value{components}"]):
            expected = 10 * (i + 1) + np.arange(components)
            np.testing.assert_array_equal(
                block, expected if components == 1 else [expected]
            )


@pytest.mark.parametrize("version", ["2.2", "4.0", "4.1"])
@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("kind", ["NodeData", "ElementData"])
@pytest.mark.parametrize("bad", ["duplicate", "unknown", "missing"])
def test_invalid_data_tags_are_rejected(tmp_path, version, binary, kind, bad):
    path = tmp_path / "invalid.msh"
    write_tagged_mesh(path, version, binary, False, bad_tags=(kind, bad))
    with pytest.raises(meshlane.ReadError, match="tags"):
        meshlane.gmsh.read(path)


@pytest.mark.parametrize("binary", [False, True])
def test_gmsh40_writer_tags_match_element_data(tmp_path, binary):
    cells = [("triangle", [[0, 1, 2]]), ("line", [[2, 3]]), ("triangle", [[1, 3, 2]])]
    mesh = meshlane.Mesh(POINTS, cells, cell_data={"value": [[10.0], [20.0], [30.0]]})
    path = tmp_path / "written.msh"
    meshlane.gmsh.write(path, mesh, fmt_version="4.0", binary=binary)
    # Inspect the on-disk element IDs independently of the meshlane reader.
    raw = path.read_bytes().split(b"$Elements\n", 1)[1]
    if binary:
        offset = 2 * np.dtype("L").itemsize
        ids = []
        for _, indices in CELLS:
            offset += 3 * 4 + np.dtype("L").itemsize
            ids.append(int(np.frombuffer(raw, dtype="i4", count=1, offset=offset)[0]))
            offset += 4 * (1 + len(indices))
    else:
        lines = raw.splitlines()
        ids = [int(lines[i].split()[0]) for i in (2, 4, 6)]
    assert ids == [1, 2, 3]
    loaded = meshlane.gmsh.read(path)
    for actual, expected in zip(loaded.cell_data["value"], mesh.cell_data["value"]):
        np.testing.assert_array_equal(actual, expected)

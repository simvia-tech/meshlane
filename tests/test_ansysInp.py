"""Tests for the meshlane.ansysInp module."""
import io
import textwrap
import tempfile
import os

import numpy as np
import pytest

from meshlane import Mesh, CellBlock
# Direct imports from the internal module for in-memory tests
from meshlane.ansysInp._ansysInp import (
    _read_lines,
    _slice_ints,
    _int_width,
    _real_width,
    _is_data_line,
    read,
    write,
)


# Helper: write to a StringIO buffer via a temporary file

def _write_to_str(mesh: Mesh) -> str:
    """Writes mesh to a temporary file and returns the content."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".inp", delete=False, encoding="utf-8"
    ) as tmp:
        tmp_name = tmp.name
    try:
        write(tmp_name, mesh)
        with open(tmp_name, encoding="utf-8") as f:
            return f.read()
    finally:
        os.unlink(tmp_name)


def _read_from_str(content: str) -> Mesh:
    """Reads a Mesh from a string (without a disk file)."""
    return _read_lines(content.splitlines())


# Test data

CUBE_TETRA_INP = textwrap.dedent("""\
    /PREP7
    ET,1,285
    NBLOCK,6,SOLID,       8,       8
    (3i9,6e21.13e3)
            1        0        0 0.0000000000000E+000 0.0000000000000E+000 0.0000000000000E+000
            2        0        0 1.0000000000000E+000 0.0000000000000E+000 0.0000000000000E+000
            3        0        0 1.0000000000000E+000 1.0000000000000E+000 0.0000000000000E+000
            4        0        0 0.0000000000000E+000 1.0000000000000E+000 0.0000000000000E+000
            5        0        0 0.0000000000000E+000 0.0000000000000E+000 1.0000000000000E+000
            6        0        0 1.0000000000000E+000 0.0000000000000E+000 1.0000000000000E+000
            7        0        0 1.0000000000000E+000 1.0000000000000E+000 1.0000000000000E+000
            8        0        0 0.0000000000000E+000 1.0000000000000E+000 1.0000000000000E+000
    N,R5.3,LOC,      -1,
    EBLOCK,19,SOLID,       2,       2
    (19i9)
            1        1        1        1        0        0        0        0        4        0        1        1        2        3        5
            1        1        1        1        0        0        0        0        4        0        2        3        4        5        8
           -1
    CMBLOCK,BOTTOM,NODE,        4
    (8i10)
             1         2         3         4
    CMBLOCK,ALL_ELEMS,ELEMENT,        2
    (8i10)
             1        -2
    FINISH
""")

RANGE_CMBLOCK_INP = textwrap.dedent("""\
    /PREP7
    ET,1,285
    NBLOCK,6,SOLID,4,4
    (3i9,6e21.13e3)
            1        0        0 0.0000000000000E+000 0.0000000000000E+000 0.0000000000000E+000
            2        0        0 1.0000000000000E+000 0.0000000000000E+000 0.0000000000000E+000
            3        0        0 0.0000000000000E+000 1.0000000000000E+000 0.0000000000000E+000
            4        0        0 0.0000000000000E+000 0.0000000000000E+000 1.0000000000000E+000
    N,R5.3,LOC,      -1,
    EBLOCK,19,SOLID,1,1
    (19i9)
            1        1        1        1        0        0        0        0        4        0        1        1        2        3        4
           -1
    CMBLOCK,ALL_NODES,NODE,        2
    (8i10)
             1        -4
    FINISH
""")

ETBLOCK_INP = textwrap.dedent("""\
    /PREP7
    ETBLOCK,1,1
    (2i9,19a9)
            1      285
           -1
    NBLOCK,6,SOLID,4,4
    (3i9,6e21.13e3)
            1        0        0 0.0000000000000E+000 0.0000000000000E+000 0.0000000000000E+000
            2        0        0 1.0000000000000E+000 0.0000000000000E+000 0.0000000000000E+000
            3        0        0 0.0000000000000E+000 1.0000000000000E+000 0.0000000000000E+000
            4        0        0 0.0000000000000E+000 0.0000000000000E+000 1.0000000000000E+000
    N,R5.3,LOC,      -1,
    EBLOCK,19,SOLID,1,1
    (19i9)
            1        1        1        1        0        0        0        0        4        0        1        1        2        3        4
           -1
    FINISH
""")


# A 10-node tetra (TET187). Its connectivity does not fit on the EBLOCK
# first line (which holds at most 8 node IDs after the 11-field header), so
# parsing exercises the continuation-line path.
TETRA10_INP = textwrap.dedent("""\
    /PREP7
    ET,1,187
    NBLOCK,6,SOLID,      10,      10
    (3i9,6e21.13e3)
            1        0        0 0.0000000000000E+000 0.0000000000000E+000 0.0000000000000E+000
            2        0        0 1.0000000000000E+000 0.0000000000000E+000 0.0000000000000E+000
            3        0        0 0.0000000000000E+000 1.0000000000000E+000 0.0000000000000E+000
            4        0        0 0.0000000000000E+000 0.0000000000000E+000 1.0000000000000E+000
            5        0        0 0.5000000000000E+000 0.0000000000000E+000 0.0000000000000E+000
            6        0        0 0.5000000000000E+000 0.5000000000000E+000 0.0000000000000E+000
            7        0        0 0.0000000000000E+000 0.5000000000000E+000 0.0000000000000E+000
            8        0        0 0.0000000000000E+000 0.0000000000000E+000 0.5000000000000E+000
            9        0        0 0.5000000000000E+000 0.0000000000000E+000 0.5000000000000E+000
           10        0        0 0.0000000000000E+000 0.5000000000000E+000 0.5000000000000E+000
    N,R5.3,LOC,      -1,
    EBLOCK,19,SOLID,       1,       1
    (19i9)
            1        1        1        1        0        0        0        0       10        0        1        1        2        3        4        5        6        7        8
            9       10
           -1
    FINISH
""")

# CMBLOCK whose first item is a negative range marker (no preceding base
# value). Such a file is malformed and must raise a clean ReadError.
BAD_CMBLOCK_INP = textwrap.dedent("""\
    /PREP7
    ET,1,285
    NBLOCK,6,SOLID,4,4
    (3i9,6e21.13e3)
            1        0        0 0.0000000000000E+000 0.0000000000000E+000 0.0000000000000E+000
            2        0        0 1.0000000000000E+000 0.0000000000000E+000 0.0000000000000E+000
            3        0        0 0.0000000000000E+000 1.0000000000000E+000 0.0000000000000E+000
            4        0        0 0.0000000000000E+000 0.0000000000000E+000 1.0000000000000E+000
    N,R5.3,LOC,      -1,
    EBLOCK,19,SOLID,1,1
    (19i9)
            1        1        1        1        0        0        0        0        4        0        1        1        2        3        4
           -1
    CMBLOCK,BAD,NODE,1
    (8i10)
            -4
    FINISH
""")

# CMBLOCK with two consecutive ranges: 1..4 then 6..8.
MULTI_RANGE_CMBLOCK_INP = textwrap.dedent("""\
    /PREP7
    ET,1,285
    NBLOCK,6,SOLID,8,8
    (3i9,6e21.13e3)
            1        0        0 0.0000000000000E+000 0.0000000000000E+000 0.0000000000000E+000
            2        0        0 1.0000000000000E+000 0.0000000000000E+000 0.0000000000000E+000
            3        0        0 0.0000000000000E+000 1.0000000000000E+000 0.0000000000000E+000
            4        0        0 0.0000000000000E+000 0.0000000000000E+000 1.0000000000000E+000
            5        0        0 1.0000000000000E+000 1.0000000000000E+000 0.0000000000000E+000
            6        0        0 1.0000000000000E+000 0.0000000000000E+000 1.0000000000000E+000
            7        0        0 0.0000000000000E+000 1.0000000000000E+000 1.0000000000000E+000
            8        0        0 1.0000000000000E+000 1.0000000000000E+000 1.0000000000000E+000
    N,R5.3,LOC,      -1,
    EBLOCK,19,SOLID,1,1
    (19i9)
            1        1        1        1        0        0        0        0        4        0        1        1        2        3        4
           -1
    CMBLOCK,TWO_RANGES,NODE,        6
    (8i10)
             1        -4         6        -8
    FINISH
""")


def _make_tetra_mesh() -> Mesh:
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)
    cells = [CellBlock("tetra", np.array([[0, 1, 2, 3]], dtype=np.int64))]
    return Mesh(points=points, cells=cells)


# Tests: low-level helpers

class TestHelpers:

    def test_slice_ints_normal(self):
        line = "        1        0        0"
        assert _slice_ints(line, 9) == [1, 0, 0]

    def test_slice_ints_negative(self):
        assert _slice_ints("       -1", 9) == [-1]

    def test_slice_ints_stops_on_text(self):
        # "FINISH" must not raise an exception - we stop
        result = _slice_ints("FINISH", 9)
        assert result == []

    def test_slice_ints_mixed_stops_at_text(self):
        # If an alphabetic chunk appears, we stop cleanly
        line = "        1        2FINISH  "
        result = _slice_ints(line, 9)
        # We get at least the first two integers
        assert result[:2] == [1, 2]

    def test_int_width_standard(self):
        assert _int_width("(3i9,6e21.13e3)") == 9

    def test_int_width_8i10(self):
        assert _int_width("(8i10)") == 10

    def test_real_width_standard(self):
        assert _real_width("(3i9,6e21.13e3)") == 21

    def test_real_width_e20(self):
        assert _real_width("(3i9,6e20.13)") == 20

    def test_is_data_line_numeric(self):
        assert _is_data_line("        1        0        0") is True

    def test_is_data_line_finish(self):
        assert _is_data_line("FINISH") is False

    def test_is_data_line_nblock(self):
        assert _is_data_line("NBLOCK,6,SOLID,8,8") is False

    def test_is_data_line_cmblock(self):
        assert _is_data_line("CMBLOCK,MY_SET,NODE,4") is False

    def test_is_data_line_comment(self):
        assert _is_data_line("! comment") is False

    def test_is_data_line_empty(self):
        assert _is_data_line("") is False

    def test_is_data_line_et_command(self):
        assert _is_data_line("ET,1,285") is False

    def test_is_data_line_n_terminator(self):
        assert _is_data_line("N,R5.3,LOC,      -1,") is False


# Tests: reading

class TestRead:

    def test_points_shape(self):
        mesh = _read_from_str(CUBE_TETRA_INP)
        assert mesh.points.shape == (8, 3)

    def test_point_first_coord(self):
        mesh = _read_from_str(CUBE_TETRA_INP)
        np.testing.assert_allclose(mesh.points[0], [0.0, 0.0, 0.0])

    def test_point_second_coord(self):
        mesh = _read_from_str(CUBE_TETRA_INP)
        np.testing.assert_allclose(mesh.points[1], [1.0, 0.0, 0.0])

    def test_point_last_coord(self):
        mesh = _read_from_str(CUBE_TETRA_INP)
        np.testing.assert_allclose(mesh.points[7], [0.0, 1.0, 1.0])

    def test_cells_type_is_tetra(self):
        mesh = _read_from_str(CUBE_TETRA_INP)
        assert len(mesh.cells) == 1
        assert mesh.cells[0].type == "tetra"

    def test_cells_count(self):
        mesh = _read_from_str(CUBE_TETRA_INP)
        assert len(mesh.cells[0].data) == 2

    def test_cell_connectivity_contains_node0(self):
        mesh = _read_from_str(CUBE_TETRA_INP)
        # Ansys node 1 → 0-based index 0
        assert 0 in mesh.cells[0].data[0]

    def test_point_set_bottom_exists(self):
        mesh = _read_from_str(CUBE_TETRA_INP)
        assert "BOTTOM" in mesh.point_sets

    def test_point_set_bottom_indices(self):
        mesh = _read_from_str(CUBE_TETRA_INP)
        assert set(mesh.point_sets["BOTTOM"].tolist()) == {0, 1, 2, 3}

    def test_cell_set_all_elems_exists(self):
        mesh = _read_from_str(CUBE_TETRA_INP)
        assert "ALL_ELEMS" in mesh.cell_sets

    def test_cell_set_all_elems_covers_both(self):
        mesh = _read_from_str(CUBE_TETRA_INP)
        flat = [i for block in mesh.cell_sets["ALL_ELEMS"] for i in block]
        assert set(flat) == {0, 1}

    def test_cmblock_range_decode(self):
        """Range notation '1 -4' must yield nodes {0,1,2,3}."""
        mesh = _read_from_str(RANGE_CMBLOCK_INP)
        assert "ALL_NODES" in mesh.point_sets
        assert set(mesh.point_sets["ALL_NODES"].tolist()) == {0, 1, 2, 3}

    def test_etblock_parsing(self):
        """ETBLOCK must be recognized as ET,1,285."""
        mesh = _read_from_str(ETBLOCK_INP)
        assert mesh.cells[0].type == "tetra"

    def test_no_block_raises(self):
        from meshlane._exceptions import ReadError
        with pytest.raises(ReadError):
            _read_from_str("/PREP7\nFINISH\n")

    def test_finish_does_not_crash_eblock(self):
        """FINISH after -1 must not crash (main bug regression)."""
        mesh = _read_from_str(CUBE_TETRA_INP)
        # If we reach this point without ValueError, the bug is fixed
        assert mesh is not None


# Tests: writing

class TestWrite:

    def test_output_contains_prep7(self):
        assert "/PREP7" in _write_to_str(_make_tetra_mesh())

    def test_output_contains_finish(self):
        assert "FINISH" in _write_to_str(_make_tetra_mesh())

    def test_output_contains_nblock(self):
        assert "NBLOCK" in _write_to_str(_make_tetra_mesh())

    def test_output_contains_eblock(self):
        assert "EBLOCK" in _write_to_str(_make_tetra_mesh())

    def test_output_contains_et(self):
        content = _write_to_str(_make_tetra_mesh())
        assert "ET," in content

    def test_nblock_node_count(self):
        content = _write_to_str(_make_tetra_mesh())
        nblock_line = next(l for l in content.splitlines() if l.startswith("NBLOCK"))
        assert "4" in nblock_line

    def test_eblock_element_count(self):
        content = _write_to_str(_make_tetra_mesh())
        eblock_line = next(l for l in content.splitlines() if l.startswith("EBLOCK"))
        assert "1" in eblock_line

    def test_unknown_type_raises(self):
        from meshlane._exceptions import WriteError
        mesh = Mesh(
            points=np.zeros((3, 3)),
            cells=[CellBlock("polygon", np.array([[0, 1, 2]]))],
        )
        with pytest.raises(WriteError):
            _write_to_str(mesh)

    def test_2d_points_extended_to_3d(self):
        """2D points must be extended with z=0 without error."""
        mesh = Mesh(
            points=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float),
            cells=[CellBlock("triangle", np.array([[0, 1, 2]]))],
        )
        content = _write_to_str(mesh)
        assert "NBLOCK" in content


# Tests: roundtrip read -> write -> read

class TestRoundtrip:

    def _roundtrip(self, content: str) -> tuple[Mesh, Mesh]:
        original = _read_from_str(content)
        written  = _write_to_str(original)
        restored = _read_from_str(written)
        return original, restored

    def test_points_preserved(self):
        orig, rt = self._roundtrip(CUBE_TETRA_INP)
        np.testing.assert_allclose(orig.points, rt.points, atol=1e-10)

    def test_cell_type_preserved(self):
        orig, rt = self._roundtrip(CUBE_TETRA_INP)
        assert orig.cells[0].type == rt.cells[0].type

    def test_cell_count_preserved(self):
        orig, rt = self._roundtrip(CUBE_TETRA_INP)
        assert len(orig.cells[0].data) == len(rt.cells[0].data)

    def test_point_set_names_preserved(self):
        orig, rt = self._roundtrip(CUBE_TETRA_INP)
        assert set(orig.point_sets.keys()) == set(rt.point_sets.keys())

    def test_point_set_content_preserved(self):
        orig, rt = self._roundtrip(CUBE_TETRA_INP)
        assert (
            set(orig.point_sets["BOTTOM"].tolist())
            == set(rt.point_sets["BOTTOM"].tolist())
        )

    def test_cell_set_names_preserved(self):
        orig, rt = self._roundtrip(CUBE_TETRA_INP)
        assert set(orig.cell_sets.keys()) == set(rt.cell_sets.keys())


# Tests: higher-order elements (connectivity spans EBLOCK continuation lines)

class TestHigherOrderElements:

    def test_tetra10_reads_without_crash(self):
        """A 10-node tetra needs a continuation line in EBLOCK; the reader
        must parse it instead of crashing on the continuation path.
        """
        mesh = _read_from_str(TETRA10_INP)
        assert len(mesh.cells) == 1
        assert mesh.cells[0].type == "tetra10"

    def test_tetra10_node_count(self):
        mesh = _read_from_str(TETRA10_INP)
        assert mesh.cells[0].data.shape == (1, 10)

    def test_tetra10_points(self):
        mesh = _read_from_str(TETRA10_INP)
        assert len(mesh.points) == 10


# Tests: CMBLOCK component decoding edge cases

class TestCMBlockEdgeCases:

    def test_multi_range_expands_correctly(self):
        """Two consecutive ranges (1..4 and 6..8) must both expand."""
        mesh = _read_from_str(MULTI_RANGE_CMBLOCK_INP)
        # 1-based node IDs 1,2,3,4,6,7,8 -> 0-based indices 0,1,2,3,5,6,7
        assert sorted(mesh.point_sets["TWO_RANGES"].tolist()) == [0, 1, 2, 3, 5, 6, 7]

    def test_leading_negative_raises_readerror(self):
        """A CMBLOCK whose first item is a range marker (negative) must raise
        a clean ReadError instead of a TypeError.
        """
        from meshlane._exceptions import ReadError
        with pytest.raises(ReadError, match="range marker"):
            _read_from_str(BAD_CMBLOCK_INP)


# Tests: 1-integer NBLOCK header + COMPACT EBLOCK (single- and multi-line)
def _fmt_1i9_nblock(coords):
    lines = [f"nblock,3,,{len(coords)}", "(1i9,3e20.9e3)"]
    for i, (x, y, z) in enumerate(coords, 1):
        lines.append(f"{i:9d}{x:20.9E}{y:20.9E}{z:20.9E}")
    lines.append("N,R5.3,LOC,      -1,")
    return "\n".join(lines)


def _fmt_compact_eblock(elems):
    lines = [f"eblock,11,COMPACT,,{len(elems)}", "(11i9)"]
    for eid, nodes in enumerate(elems, 1):
        vals = [eid] + list(nodes)
        for k in range(0, len(vals), 11):  # 11 fields per line, wrap
            lines.append("".join(f"{v:9d}" for v in vals[k:k + 11]))
    lines.append("       -1")
    return "\n".join(lines)


class TestCompactEblock:
    def test_1i9_nblock_and_compact_hex(self):
        # unit cube, SOLID185 (linear hex), COMPACT eblock, 1-integer NBLOCK
        coords = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                  (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
        inp = (
            "/PREP7\net,1,185\nMAT,1 $ TYPE,1 $ REAL,1\n"
            + _fmt_1i9_nblock(coords) + "\n"
            + _fmt_compact_eblock([list(range(1, 9))])
        )
        mesh = _read_from_str(inp)
        assert len(mesh.points) == 8
        # coordinates parsed correctly from the 1-integer NBLOCK
        assert np.allclose(mesh.points[0], [0.0, 0.0, 0.0])
        assert np.allclose(mesh.points[6], [1.0, 1.0, 1.0])
        assert len(mesh.cells) == 1
        assert mesh.cells[0].type == "hexahedron"
        assert mesh.cells[0].data.shape == (1, 8)

    def test_compact_multiline_hex20(self):
        # SOLID186 (hex20): 20 nodes span two lines in COMPACT (11i9)
        coords = [(float(i), 0.0, 0.0) for i in range(20)]
        inp = (
            "/PREP7\net,1,186\nMAT,1 $ TYPE,1 $ REAL,1\n"
            + _fmt_1i9_nblock(coords) + "\n"
            + _fmt_compact_eblock([list(range(1, 21))])
        )
        mesh = _read_from_str(inp)
        assert mesh.cells[0].type == "hexahedron20"
        assert mesh.cells[0].data.shape == (1, 20)
        # elem_id stripped; 20 nodes in order (0-based)
        assert mesh.cells[0].data[0].tolist() == list(range(20))


# Tests: ANSYS degenerate solids (tet/wedge/pyramid stored as hexes) and
# contact/target element skipping.

class TestDegenerateAndContact:
    def _deck(self, coords, ansys_num, conn):
        return (
            f"/PREP7\net,1,{ansys_num}\nMAT,1 $ TYPE,1 $ REAL,1\n"
            + _fmt_1i9_nblock(coords) + "\n"
            + _fmt_compact_eblock(conn)
        )

    def test_degenerate_hex_to_tetra(self):
        # SOLID185 tet stored as "I J K K M M M M"
        coords = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
        mesh = _read_from_str(self._deck(coords, 185, [[1, 2, 3, 3, 4, 4, 4, 4]]))
        assert len(mesh.cells) == 1
        assert mesh.cells[0].type == "tetra"
        assert mesh.cells[0].data[0].tolist() == [0, 1, 2, 3]

    def test_degenerate_hex_to_wedge(self):
        # SOLID185 wedge stored as "I J K K M N O O" (K=L and O=P)
        coords = [(0, 0, 0), (1, 0, 0), (0, 1, 0),
                  (0, 0, 1), (1, 0, 1), (0, 1, 1)]
        mesh = _read_from_str(self._deck(coords, 185, [[1, 2, 3, 3, 4, 5, 6, 6]]))
        assert mesh.cells[0].type == "wedge"
        assert mesh.cells[0].data[0].tolist() == [0, 1, 2, 3, 4, 5]

    def test_degenerate_hex_to_pyramid(self):
        # SOLID185 pyramid stored as "I J K L M M M M" (M=N=O=P)
        coords = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0.5, 0.5, 1)]
        mesh = _read_from_str(self._deck(coords, 185, [[1, 2, 3, 4, 5, 5, 5, 5]]))
        assert mesh.cells[0].type == "pyramid"
        assert mesh.cells[0].data[0].tolist() == [0, 1, 2, 3, 4]

    def test_genuine_hex_stays_hex(self):
        # 8 distinct corner nodes must remain a hexahedron (no false collapse)
        coords = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                  (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
        mesh = _read_from_str(self._deck(coords, 185, [list(range(1, 9))]))
        assert mesh.cells[0].type == "hexahedron"

    def test_contact_elements_skipped(self):
        # A hex plus a CONTA174 contact patch, where the contact type is declared
        # via an APDL parameter ("*set,cid,3" + "et,cid,174"). The patch must be
        # dropped, and *set/label resolution must work (otherwise the patch would
        # be misread as a solid tetra).
        coords = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                  (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
        inp = "\n".join([
            "/PREP7",
            "*set,cid,3",
            "et,1,185",
            "et,cid,174",
            _fmt_1i9_nblock(coords),
            "MAT,1 $ TYPE,1 $ REAL,1",
            _fmt_compact_eblock([list(range(1, 9))]),   # solid hex
            "MAT,3 $ TYPE,3 $ REAL,3",
            _fmt_compact_eblock([[1, 2, 3, 4]]),        # CONTA174 patch -> skipped
        ])
        mesh = _read_from_str(inp)
        assert len(mesh.cells) == 1
        assert mesh.cells[0].type == "hexahedron"
        assert "tetra" not in {c.type for c in mesh.cells}

    def test_degenerate_hex20_to_tetra10_midpoints(self):
        # A curved SOLID186 tet stored as a degenerate 20-node hex. The recovered
        # tetra10 must place each midside node at the midpoint of its edge, which
        # validates the ANSYS hex20 -> tetra10 node mapping geometrically.
        coords = [
            (0, 0, 0), (2, 0, 0), (0, 2, 0), (0, 0, 2),    # 1..4 corners I,J,K,M
            (1, 0, 0), (1, 1, 0), (0, 1, 0),               # 5,6,7 = IJ, JK, KI
            (0, 0, 1), (1, 0, 1), (0, 1, 1),               # 8,9,10 = IM, JM, KM
        ]
        conn = [[1, 2, 3, 3, 4, 4, 4, 4,
                 5, 6, 3, 7, 4, 4, 4, 4, 8, 9, 10, 10]]
        mesh = _read_from_str(self._deck(coords, 186, conn))
        assert mesh.cells[0].type == "tetra10"
        assert mesh.cells[0].data.shape == (1, 10)
        c = mesh.points[mesh.cells[0].data[0]]
        assert np.allclose(c[4], (c[0] + c[1]) / 2)   # edge (0,1)
        assert np.allclose(c[5], (c[1] + c[2]) / 2)   # edge (1,2)
        assert np.allclose(c[6], (c[2] + c[0]) / 2)   # edge (2,0)
        assert np.allclose(c[7], (c[0] + c[3]) / 2)   # edge (0,3)
        assert np.allclose(c[8], (c[1] + c[3]) / 2)   # edge (1,3)
        assert np.allclose(c[9], (c[2] + c[3]) / 2)   # edge (2,3)

"""
I/O for Ansys Fluent's msh format.

<https://romeo.univ-reims.fr/documents/fluent/tgrid/ug/appb.pdf>

A Fluent mesh describes the volume cells only through their faces: each face lists
its nodes and the cells on either side, and the cell sections only give the cell
types. The reader rebuilds the cells from the faces (see ``_cells.py``) and returns
them with the boundary faces; the named zones (sections 39/45) become cell_sets.
Interior faces are not returned.

Files written by older meshlane versions (and by meshio) list the cell nodes in the
cell sections instead; they are read as such.
"""

import re

import numpy as np

from ..__about__ import __version__
from .._exceptions import ReadError, WriteError
from .._files import open_file
from .._helpers import register_format
from .._mesh import CellBlock, Mesh
from ._cells import Faces, build_cells

_SECTION = re.compile(rb"\(\s*([0-9]+)")
_SPACE = re.compile(rb"\s*")
_DELIMITER = re.compile(rb'[()"]')
_BINARY_END = b"End of Binary Section"

# Face zone types (bc-type of section 13) that are not boundaries: interior faces,
# and the parent faces of hanging-node refinement (their children are stored too).
_BC_INTERIOR = 2
_BC_PARENT = 31

# Element types of section 12 and their node count, for the files that list the
# cell nodes in it.
_CELL_TYPES = {
    1: ("triangle", 3),
    2: ("tetra", 4),
    3: ("quad", 4),
    4: ("hexahedron", 8),
    5: ("pyramid", 5),
    6: ("wedge", 6),
}


def _close(buf, pos):
    """Position after the ``)`` matching an already open bracket, from pos. Nested
    brackets and quoted strings are skipped."""
    depth = 1
    while True:
        m = _DELIMITER.search(buf, pos)
        if m is None:
            raise ReadError("Unbalanced brackets.")
        pos = m.end()
        char = m.group()
        if char == b'"':
            pos = buf.index(b'"', pos) + 1
        elif char == b"(":
            depth += 1
        else:
            depth -= 1
            if depth == 0:
                return pos


def _binary_end(buf, pos):
    """Position after a binary section, from anywhere inside it. Binary data may
    contain any byte, so brackets cannot be counted: the section ends with the
    ``End of Binary Section <index>)`` marker."""
    marker = buf.find(_BINARY_END, pos)
    if marker < 0:
        raise ReadError("Missing 'End of Binary Section'.")
    return buf.index(b")", marker) + 1


def _hex_ints(text):
    tokens = text.split()
    return np.fromiter((int(t, 16) for t in tokens), dtype=np.int64, count=len(tokens))


def _split_faces(ints, num_faces, face_type):
    """CSR nodes and the two cells of each face from the body of section 13.

    Each face is ``n0 n1 ... c0 c1``; in mixed (0) and polygonal (5) zones it is
    preceded by its number of nodes."""
    ints = np.asarray(ints, dtype=np.int64)
    if face_type in (2, 3, 4):
        k = face_type
        if ints.size != num_faces * (k + 2):
            raise ReadError("Face section has the wrong size.")
        a = ints.reshape(num_faces, k + 2)
        return np.arange(num_faces + 1) * k, a[:, :k].ravel(), a[:, k], a[:, k + 1]
    values = ints.tolist()
    starts = np.empty(num_faces, dtype=np.int64)
    sizes = np.empty(num_faces, dtype=np.int64)
    i = 0
    for j in range(num_faces):
        x = values[i]
        starts[j] = i + 1
        sizes[j] = x
        i += x + 3
    if i > len(values) or (sizes < 2).any():
        raise ReadError("Mixed face section is inconsistent.")
    ptr = np.concatenate([[0], np.cumsum(sizes)])
    idx = np.repeat(starts - ptr[:-1], sizes) + np.arange(ptr[-1])
    return ptr, ints[idx], ints[starts + sizes], ints[starts + sizes + 1]


class _FluentMesh:
    """The content of a Fluent mesh, as read from a .msh or a .msh.h5 file, with
    1-based node and cell indices (0: no cell)."""

    def __init__(self):
        self.dim = 3
        self.nodes = []  # (first node index, points)
        self.num_cells = 0
        self.cell_zones = []  # (zone id, first cell index, last cell index)
        self.listed_cells = []  # (cell type, nodes), cells listed in section 12
        self.face_zones = []  # (zone id, bc type, ptr, nodes, c0, c1)
        self.zones = {}  # zone id -> (zone type, name)

    def _zone_name(self, zone):
        return self.zones[zone][1] if zone in self.zones else f"zone{zone}"

    def mesh(self):
        if self.nodes:
            offset = min(first for first, _ in self.nodes)
            num_points = max(first + len(p) for first, p in self.nodes) - offset
            points = np.zeros((num_points, self.nodes[0][1].shape[1]))
            for first, p in self.nodes:
                points[first - offset : first - offset + len(p)] = p
        else:
            offset = 1
            points = np.zeros((0, self.dim))

        if not self.face_zones:
            cells = [(t, nodes - offset) for t, nodes in self.listed_cells]
            return Mesh(points, cells)

        zones = [z for z in self.face_zones if z[1] != _BC_PARENT]
        sizes = [len(z[4]) for z in zones]
        ptr = [np.array([0])]
        shift = 0
        for z in zones:
            ptr.append(z[2][1:] + shift)
            shift += z[2][-1]
        face_nodes = np.concatenate([z[3] for z in zones])
        face_nodes -= offset
        c0 = np.concatenate([z[4] for z in zones])
        c0 -= 1
        c1 = np.concatenate([z[5] for z in zones])
        c1 -= 1
        faces = Faces(np.concatenate(ptr), face_nodes, c0, c1)
        face_zone = np.repeat([z[0] for z in zones], sizes)
        face_bc = np.repeat([z[1] for z in zones], sizes)
        # the per-zone arrays are copied in faces: free them (large meshes)
        del zones, face_nodes, c0, c1
        self.face_zones = []
        if len(faces.nodes) and (
            faces.nodes.min() < 0 or faces.nodes.max() >= len(points)
        ):
            raise ReadError("Face node index out of range.")
        num_cells = max(
            self.num_cells,
            int(faces.c0.max(initial=-1)) + 1,
            int(faces.c1.max(initial=-1)) + 1,
        )

        blocks, out_of_c0 = build_cells(points, faces, num_cells)

        # boundary faces, oriented out of the domain
        num_volume = len(blocks)
        boundary = np.flatnonzero(face_bc != _BC_INTERIOR)
        # (faces without cells, as in surface meshes, keep their node order)
        has_c0 = faces.c0[boundary] >= 0
        has_c1 = faces.c1[boundary] >= 0
        reverse = np.where(has_c0, not out_of_c0, has_c1 & out_of_c0)
        size = faces.size[boundary]
        for k in np.unique(size):
            sel = size == k
            f = boundary[sel]
            nodes = faces.nodes[faces.ptr[f][:, None] + np.arange(k)]
            nodes = np.where(reverse[sel][:, None], nodes[:, ::-1], nodes)
            kind = {2: "line", 3: "triangle", 4: "quad"}.get(k, f"polygon{k}")
            blocks.append((kind, nodes, f))

        # zones -> cell_sets; volume blocks hold cell ids, face blocks face ids
        cell_sets = {}
        for zone, first, last in self.cell_zones:
            name = self._zone_name(zone)
            cell_sets[name] = [
                (
                    np.flatnonzero((ids >= first - 1) & (ids <= last - 1))
                    if i < num_volume
                    else np.array([], dtype=int)
                )
                for i, (_, _, ids) in enumerate(blocks)
            ]
        for zone in np.unique(face_zone[boundary]):
            name = self._zone_name(int(zone))
            cell_sets[name] = [
                (
                    np.flatnonzero(face_zone[ids] == zone)
                    if i >= num_volume
                    else np.array([], dtype=int)
                )
                for i, (_, _, ids) in enumerate(blocks)
            ]
        cell_sets = {
            name: sets for name, sets in cell_sets.items() if any(len(s) for s in sets)
        }

        cells = [CellBlock(kind, data) for kind, data, _ in blocks]
        return Mesh(points, cells, cell_sets=cell_sets)


class _Parser(_FluentMesh):
    def __init__(self, buf):
        super().__init__()
        self.buf = buf
        # meshlane and meshio write a newline between "(" and binary data
        self.written_by_meshlane = False

    def parse(self):
        buf = self.buf
        handlers = {
            "1": self._title,
            "2": self._dimension,
            "10": self._nodes,
            "12": self._cells,
            "13": self._faces,
            "39": self._zone,
            "45": self._zone,
        }
        pos = 0
        while True:
            pos = buf.find(b"(", pos)
            if pos < 0:
                return
            m = _SECTION.match(buf, pos)
            if m is None:
                raise ReadError(f"Expected a section index at byte {pos}.")
            index = m.group(1).decode()
            # binary sections are prefixed with 20 (single precision) or 30 (double)
            binary = len(index) == 4 and index[:2] in ("20", "30")
            code = index[2:] if binary else index
            handler = handlers.get(code)
            if handler is None:
                pos = self._skip(m.end(), binary)
            else:
                pos = handler(m.end(), index, binary)

    # -- section structure ------------------------------------------------------

    def _header(self, pos):
        """Tokens of the ``(...)`` header that follows the section index."""
        pos = _SPACE.match(self.buf, pos).end()
        if self.buf[pos : pos + 1] != b"(":
            raise ReadError(f"Expected a section header at byte {pos}.")
        end = self.buf.index(b")", pos)
        return self.buf[pos + 1 : end].split(), end + 1

    def _int_header(self, pos):
        tokens, pos = self._header(pos)
        if len(tokens) < 3:
            raise ReadError(f"Section header too short at byte {pos}.")
        return [int(t, 16) for t in tokens], pos

    def _body(self, pos):
        """Position of the first byte of the body after a header, or None."""
        pos = _SPACE.match(self.buf, pos).end()
        return pos + 1 if self.buf[pos : pos + 1] == b"(" else None

    def _end(self, pos, binary):
        """Skip the rest of a section, from after its header."""
        pos = _SPACE.match(self.buf, pos).end()
        if self.buf[pos : pos + 1] == b")":
            return pos + 1
        if binary:
            return _binary_end(self.buf, pos)
        return _close(self.buf, pos)

    def _skip(self, pos, binary):
        if binary:
            return self._end(self._header(pos)[1], binary)
        return _close(self.buf, pos)

    def _ascii_body(self, body):
        """Text of an ASCII body, and the position after its section."""
        close = self.buf.index(b")", body)
        return self.buf[body:close], _close(self.buf, close + 1)

    # -- sections ---------------------------------------------------------------

    def _binary_body(self, body):
        if self.written_by_meshlane and self.buf[body : body + 1] == b"\n":
            return body + 1
        return body

    def _title(self, pos, index, binary):
        # (1 "meshlane 5.5.0")
        end = _close(self.buf, pos)
        if re.match(rb'\s*"mesh(io|lane) ', self.buf[pos:end]):
            self.written_by_meshlane = True
        return end

    def _dimension(self, pos, index, binary):
        # (2 3)
        m = re.compile(rb"\s*([0-9]+)").match(self.buf, pos)
        if m is None:
            raise ReadError("Invalid dimension section.")
        self.dim = int(m.group(1))
        return _close(self.buf, m.end())

    def _nodes(self, pos, index, binary):
        # (10 (zone-id first-index last-index type [ND]) (x y z ...))
        a, pos = self._int_header(pos)
        if a[0] == 0:
            # declaration of the total number of nodes
            return self._end(pos, binary)
        first, last = a[1], a[2]
        num_nodes = last - first + 1
        dim = a[4] if len(a) > 4 else self.dim
        body = self._body(pos)
        if body is None:
            return self._end(pos, binary)
        count = num_nodes * dim
        if binary:
            dtype = np.dtype("<f8" if index.startswith("30") else "<f4")
            body = self._binary_body(body)
            pts = np.frombuffer(self.buf, dtype, count=count, offset=body)
            end = _binary_end(self.buf, body + count * dtype.itemsize)
        else:
            text, end = self._ascii_body(body)
            pts = np.array(text.split()).astype(float)
        if pts.size != count:
            raise ReadError("Node section has the wrong size.")
        self.nodes.append((first, pts.reshape(num_nodes, dim).astype(float)))
        return end

    def _cells(self, pos, index, binary):
        # (12 (zone-id first-index last-index type element-type) [(types)])
        a, pos = self._int_header(pos)
        zone, first, last = a[:3]
        self.num_cells = max(self.num_cells, last)
        if zone == 0:
            return self._end(pos, binary)
        self.cell_zones.append((zone, first, last))
        element_type = a[4] if len(a) > 4 else 0
        body = self._body(pos)
        if body is None or element_type not in _CELL_TYPES:
            # Fluent: no body, or the type of each cell of a mixed zone; the cells
            # are rebuilt from the faces.
            return self._end(pos, binary)

        # cell nodes, as written by older meshlane versions
        cell_type, k = _CELL_TYPES[element_type]
        count = (last - first + 1) * k
        if binary:
            # 2012 sections hold 32-bit integers; 3012 ones 32 or 64-bit.
            body = self._binary_body(body)
            stop = self.buf.find(_BINARY_END, body)
            itemsize = 8 if stop - body >= 8 * count else 4
            nodes = np.frombuffer(self.buf, f"<i{itemsize}", count=count, offset=body)
            end = _binary_end(self.buf, body + count * itemsize)
        else:
            text, end = self._ascii_body(body)
            nodes = _hex_ints(text)
        if nodes.size != count:
            raise ReadError("Cell section has the wrong size.")
        self.listed_cells.append((cell_type, nodes.reshape(-1, k).astype(np.int64)))
        return end

    def _faces(self, pos, index, binary):
        # (13 (zone-id first-index last-index bc-type face-type) (n0 n1 ... c0 c1))
        a, pos = self._int_header(pos)
        zone, first, last = a[:3]
        if zone == 0:
            return self._end(pos, binary)
        bc_type = a[3] if len(a) > 3 else 0
        face_type = a[4] if len(a) > 4 else 0
        num_faces = last - first + 1
        body = self._body(pos)
        if body is None:
            return self._end(pos, binary)
        if binary:
            if face_type in (2, 3, 4):
                count = num_faces * (face_type + 2)
            else:
                count = (self.buf.find(_BINARY_END, body) - body) // 4
            ints = np.frombuffer(self.buf, "<i4", count=count, offset=body)
            end = _binary_end(self.buf, body)
        else:
            text, end = self._ascii_body(body)
            ints = _hex_ints(text)
        ptr, nodes, c0, c1 = _split_faces(ints, num_faces, face_type)
        self.face_zones.append((zone, bc_type, ptr, nodes, c0, c1))
        return end

    def _zone(self, pos, index, binary):
        # (39 (zone-id zone-type zone-name [domain-id]) (scheme properties...))
        tokens, pos = self._header(pos)
        if len(tokens) >= 3:
            try:
                zone = int(tokens[0])
            except ValueError:
                zone = int(tokens[0], 16)
            self.zones[zone] = (tokens[1].decode(), tokens[2].decode())
        return _close(self.buf, pos)


_HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"


def read(filename):
    with open_file(filename, "rb") as f:
        head = f.read(1024)
        if head.startswith(_HDF5_SIGNATURE):
            # Fluent HDF5 mesh saved with a plain .msh extension
            from ._ansys_h5 import read as read_h5

            return read_h5(filename)
        # Fluent files start with a section; fail fast on other .msh files
        if not head.lstrip().startswith(b"("):
            raise ReadError("Not a Fluent mesh file.")
        buf = head + f.read()
    parser = _Parser(buf)
    try:
        parser.parse()
        return parser.mesh()
    except (ValueError, IndexError, KeyError) as e:
        raise ReadError(f"Invalid Fluent mesh file: {e}") from e


def write(filename, mesh, binary=True):
    with open_file(filename, "wb") as fh:
        # header
        fh.write(f'(1 "meshlane {__version__}")\n'.encode())

        # dimension
        num_points, dim = mesh.points.shape
        if dim not in [2, 3]:
            raise WriteError(f"Can only write dimension 2, 3, got {dim}.")
        fh.write((f"(2 {dim})\n").encode())

        # total number of nodes
        first_node_index = 1
        fh.write((f"(10 (0 {first_node_index:x} {num_points:x} 0))\n").encode())

        # total number of cells
        total_num_cells = sum(len(c) for c in mesh.cells)
        fh.write((f"(12 (0 1 {total_num_cells:x} 0))\n").encode())

        # Write nodes
        key = "3010" if binary else "10"
        fh.write(
            f"({key} (1 {first_node_index:x} {num_points:x} 1 {dim:x})(\n".encode()
        )
        if binary:
            mesh.points.tofile(fh)
            fh.write(b"\n)")
            fh.write(b"End of Binary Section 3010)\n")
        else:
            np.savetxt(fh, mesh.points, fmt="%.16e")
            fh.write(b"))\n")

        # Write cells
        meshio_to_ansys_type = {
            # "mixed": 0,
            "triangle": 1,
            "tetra": 2,
            "quad": 3,
            "hexahedron": 4,
            "pyramid": 5,
            "wedge": 6,
            # "polyhedral": 7,
        }
        first_index = 0
        binary_dtypes = {
            # np.int16 is not allowed
            np.dtype("int32"): "2012",
            np.dtype("int64"): "3012",
        }
        for cell_block in mesh.cells:
            cell_type = cell_block.type
            values = cell_block.data
            key = binary_dtypes[values.dtype] if binary else "12"
            last_index = first_index + len(values) - 1
            try:
                ansys_cell_type = meshio_to_ansys_type[cell_type]
            except KeyError:
                legal_keys = ", ".join(meshio_to_ansys_type.keys())
                raise KeyError(
                    f"Illegal ANSYS cell type '{cell_type}'. (legal: {legal_keys})"
                )
            fh.write(
                f"({key} (1 {first_index:x} {last_index:x} 1 {ansys_cell_type})(\n".encode()
            )
            if binary:
                (values + first_node_index).tofile(fh)
                fh.write(b"\n)")
                fh.write((f"End of Binary Section {key})\n").encode())
            else:
                np.savetxt(fh, values + first_node_index, fmt="%x")
                fh.write(b"))\n")
            first_index = last_index + 1


register_format("ansys", [".msh"], read, {"ansys": write})

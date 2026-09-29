"""
I/O for the native ASCII format of OpenVolumeMesh (.ovm), cf.
<https://www.graphics.rwth-aachen.de/media/openvolumemesh_static/Documentation/OpenVolumeMesh-Doc-Latest/file_format.html>.

Based on the meshio implementation by Martin Heistermann
<https://github.com/mheistermann/meshio/commit/a9219d00894ffbd2c7f3c04ea1cad1bae65439b0>.

OVM does not describe a cell by its nodes but by its topology: vertices, edges
(pairs of vertices), faces (loops of half-edges) and polyhedra (sets of
half-faces). Half-edge 2e runs along edge e from source to target and 2e + 1 the
other way round; half-faces 2f and 2f + 1 are the two sides of face f. The
OpenVolumeMesh convention is that the half-face normals of a cell point inwards.

Consequences for the conversion:

  * on writing, edges and faces are built from the cells and shared ones are
    written once;
  * on reading, cell types are deduced from the face valences and the node order
    is rebuilt; only the cells of the highest dimension present are returned, so
    the faces and edges bounding volume cells do not become extra cell blocks;
  * properties (the ``*Prop`` sections after ``Polyhedra``) are skipped.
"""

from __future__ import annotations

import numpy as np

from .._common import warn
from .._exceptions import ReadError, WriteError
from .._files import open_file
from .._helpers import register_format
from .._mesh import CellBlock, Mesh

# Faces of each cell type, in meshio node ordering, with their normals pointing
# into the cell. The first entry is the base the reader rebuilds the cell from.
_cell_faces = {
    "tetra": [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]],
    "pyramid": [[0, 1, 2, 3], [0, 4, 1], [1, 4, 2], [2, 4, 3], [3, 4, 0]],
    "wedge": [[0, 1, 2], [0, 3, 4, 1], [1, 4, 5, 2], [2, 5, 3, 0], [3, 5, 4]],
    "hexahedron": [
        [0, 1, 2, 3],
        [0, 4, 5, 1],
        [1, 5, 6, 2],
        [2, 6, 7, 3],
        [3, 7, 4, 0],
        [4, 7, 6, 5],
    ],
}

# Number of values of each property kind, as a function of the entity counts
_property_sizes = {
    "vprop": lambda n: n["vertices"],
    "eprop": lambda n: n["edges"],
    "heprop": lambda n: 2 * n["edges"],
    "fprop": lambda n: n["faces"],
    "hfprop": lambda n: 2 * n["faces"],
    "cprop": lambda n: n["polyhedra"],
    "mprop": lambda _: 1,
}


def read(filename):
    with open_file(filename) as f:
        lines = f.read().splitlines()
    return _Reader(lines).read()


class _Reader:
    def __init__(self, lines):
        # OpenVolumeMesh skips blank lines and lines starting with '#'
        self.lines = [
            line.strip()
            for line in lines
            if line.strip() and not line.strip().startswith("#")
        ]
        self.k = 0

    def next_line(self, what):
        if self.k >= len(self.lines):
            raise ReadError(f"OVM: unexpected end of file while reading {what}.")
        line = self.lines[self.k]
        self.k += 1
        return line

    def section(self, name):
        line = self.next_line(f"the {name} section")
        if line.split()[0].lower() != name:
            raise ReadError(
                f"OVM: expected section '{name.capitalize()}', found '{line}'."
            )
        try:
            count = int(self.next_line(f"the {name} count"))
        except ValueError as e:
            raise ReadError(f"OVM: invalid {name} count.") from e
        if count < 0:
            raise ReadError(f"OVM: negative {name} count.")
        return [self.next_line(name).split() for _ in range(count)]

    def read(self):
        if not self.lines:
            raise ReadError("OVM: empty file.")
        header = self.lines[0].split()
        if header[0].upper() == "OVM":
            if len(header) > 1 and header[1].upper() == "BINARY":
                raise ReadError("OVM: binary files are not supported.")
            self.k = 1

        try:
            vertices = self.section("vertices")
            points = np.array([v[:3] for v in vertices], dtype=float).reshape(-1, 3)
            edges = np.array([e[:2] for e in self.section("edges")], dtype=int).reshape(
                -1, 2
            )
            faces = [_valence_list(f, "face") for f in self.section("faces")]
            polyhedra = [
                _valence_list(c, "polyhedron") for c in self.section("polyhedra")
            ]
        except ValueError as e:
            raise ReadError(f"OVM: invalid number ({e}).") from e

        counts = {
            "vertices": len(points),
            "edges": len(edges),
            "faces": len(faces),
            "polyhedra": len(polyhedra),
        }
        self.skip_properties(counts)

        topology = _Topology(points, edges, faces)
        if polyhedra:
            cells = topology.volume_cells(polyhedra)
        elif faces:
            cells = topology.surface_cells()
        elif len(edges):
            cells = [CellBlock("line", edges.copy())]
        else:
            cells = []
        return Mesh(points, cells)

    def skip_properties(self, counts):
        while self.k < len(self.lines):
            line = self.next_line("a property")
            kind = line.split()[0].lower()
            if kind not in _property_sizes:
                raise ReadError(f"OVM: unexpected line '{line}' after Polyhedra.")
            for _ in range(_property_sizes[kind](counts)):
                self.next_line(f"the values of property {line}")


def _valence_list(tokens, what):
    valence = int(tokens[0])
    indices = [int(t) for t in tokens[1:]]
    if len(indices) != valence:
        raise ReadError(
            f"OVM: {what} of valence {valence} lists {len(indices)} indices."
        )
    return indices


class _Topology:
    def __init__(self, points, edges, faces):
        self.points = points
        if edges.size and (edges.min() < 0 or edges.max() >= len(points)):
            raise ReadError("OVM: edge with an invalid vertex index.")
        self.edges = edges
        n_halfedges = 2 * len(edges)
        # vertex loop of each face, in the order of its half-edges
        self.faces = []
        for face in faces:
            if min(face, default=0) < 0 or max(face, default=0) >= n_halfedges:
                raise ReadError("OVM: face with an invalid half-edge index.")
            sources = [edges[he // 2, he % 2] for he in face]
            targets = [edges[he // 2, 1 - he % 2] for he in face]
            if targets != sources[1:] + sources[:1]:
                raise ReadError("OVM: face whose half-edges do not form a loop.")
            self.faces.append([int(v) for v in sources])

    def halfface(self, hf):
        if hf < 0 or hf >= 2 * len(self.faces):
            raise ReadError("OVM: polyhedron with an invalid half-face index.")
        verts = self.faces[hf // 2]
        # the opposite side runs the same loop backwards
        return verts[:1] + verts[:0:-1] if hf % 2 else list(verts)

    def surface_cells(self):
        blocks = {}
        for verts in self.faces:
            cell_type = {3: "triangle", 4: "quad"}.get(len(verts), "polygon")
            blocks.setdefault(cell_type, []).append(verts)
        return [CellBlock(cell_type, data) for cell_type, data in blocks.items()]

    def volume_cells(self, polyhedra):
        builders = {
            (3, 3, 3, 3): ("tetra", self._tetra),
            (3, 3, 3, 3, 4): ("pyramid", self._pyramid),
            (3, 3, 4, 4, 4): ("wedge", self._wedge),
            (4, 4, 4, 4, 4, 4): ("hexahedron", self._hexahedron),
        }
        blocks = {}
        n_skipped = 0
        for polyhedron in polyhedra:
            loops = [self.halfface(hf) for hf in polyhedron]
            signature = tuple(sorted(len(loop) for loop in loops))
            if signature not in builders:
                n_skipped += 1
                continue
            cell_type, build = builders[signature]
            blocks.setdefault(cell_type, []).append(build(loops))
        if n_skipped:
            warn(f"OVM: skipped {n_skipped} polyhedra of unsupported type.")
        return [
            CellBlock(cell_type, np.array(data, dtype=int))
            for cell_type, data in blocks.items()
        ]

    def _oriented_base(self, base, others):
        # The base loop must turn counterclockwise seen from the rest of the
        # cell. That holds for inward half-face normals (the OpenVolumeMesh
        # convention); files with outward normals get their base reversed.
        p = self.points[base]
        normal = np.cross(p[1] - p[0], p[2] - p[0])
        if len(base) == 4:
            normal += np.cross(p[2] - p[0], p[3] - p[0])
        if np.dot(normal, self.points[others].mean(axis=0) - p.mean(axis=0)) < 0:
            return base[:1] + base[:0:-1]
        return base

    @staticmethod
    def _check(loops, nodes, n_nodes):
        if len(set(nodes)) != n_nodes or {v for loop in loops for v in loop} != set(
            nodes
        ):
            raise ReadError("OVM: polyhedron whose faces do not bound a cell.")

    @staticmethod
    def _edges(loops):
        edges = set()
        for loop in loops:
            for a, b in zip(loop, loop[1:] + loop[:1]):
                edges.add((a, b))
                edges.add((b, a))
        return edges

    def _lift(self, loops, base, n_nodes):
        """The node above each base node: its neighbour off the base."""
        edges = self._edges(loops)
        others = {v for loop in loops for v in loop} - set(base)
        top = []
        for v in base:
            above = [w for w in others if (v, w) in edges]
            if len(above) != 1:
                raise ReadError("OVM: polyhedron whose faces do not bound a cell.")
            top.append(above[0])
        nodes = base + top
        self._check(loops, nodes, n_nodes)
        return nodes

    def _tetra(self, loops):
        base = loops[0]
        others = [v for loop in loops for v in loop if v not in base]
        nodes = self._oriented_base(base, others) + others[:1]
        self._check(loops, nodes, 4)
        return nodes

    def _pyramid(self, loops):
        base = next(loop for loop in loops if len(loop) == 4)
        others = [v for loop in loops for v in loop if v not in base]
        nodes = self._oriented_base(base, others) + others[:1]
        self._check(loops, nodes, 5)
        return nodes

    def _wedge(self, loops):
        base = next(loop for loop in loops if len(loop) == 3)
        others = [v for loop in loops for v in loop if v not in base]
        return self._lift(loops, self._oriented_base(base, others), 6)

    def _hexahedron(self, loops):
        base = loops[0]
        others = [v for loop in loops for v in loop if v not in base]
        return self._lift(loops, self._oriented_base(base, others), 8)


def write(filename, mesh):
    points = mesh.points
    if points.ndim != 2 or points.shape[1] not in (2, 3):
        raise WriteError("OVM requires 2D or 3D points.")
    if points.shape[1] == 2:
        points = np.column_stack([points, np.zeros(len(points))])

    topology = _TopologyBuilder()
    polyhedra = []
    skipped = set()
    for cell_block in mesh.cells:
        if cell_block.type in _cell_faces:
            local_faces = _cell_faces[cell_block.type]
            for cell in cell_block.data:
                polyhedra.append(
                    [topology.halfface([cell[i] for i in f]) for f in local_faces]
                )
        elif cell_block.type in ("triangle", "quad") or cell_block.type.startswith(
            "polygon"
        ):
            for cell in cell_block.data:
                topology.halfface(cell)
        elif cell_block.type == "line":
            for src, dst in cell_block.data:
                topology.halfedge(src, dst)
        elif cell_block.type != "vertex":
            skipped.add(cell_block.type)
    if skipped:
        warn(f"OVM: cell types not supported, skipped: {', '.join(sorted(skipped))}.")
    if mesh.point_data or mesh.cell_data:
        warn("OVM: point and cell data are not written.")

    with open_file(filename, "w") as f:
        f.write("OVM ASCII\n")
        f.write(f"Vertices\n{len(points)}\n")
        # 17 significant digits make every float64 round-trip exactly
        f.writelines(f"{x:.17g} {y:.17g} {z:.17g}\n" for x, y, z in points)
        f.write(f"Edges\n{len(topology.edges)}\n")
        f.writelines(f"{a} {b}\n" for a, b in topology.edges)
        f.write(f"Faces\n{len(topology.faces)}\n")
        f.writelines(_valence_line(hes) for hes in topology.faces)
        f.write(f"Polyhedra\n{len(polyhedra)}\n")
        f.writelines(_valence_line(hfs) for hfs in polyhedra)


def _valence_line(indices):
    return f"{len(indices)} " + " ".join(str(i) for i in indices) + "\n"


class _TopologyBuilder:
    """Edges and faces built from cells, each shared entity stored once."""

    def __init__(self):
        self.edges = []
        self.faces = []
        self._edge_index = {}
        # face key (sorted vertices) -> (face index, vertex loop as first stored)
        self._face_index = {}

    def halfedge(self, src, dst):
        src, dst = int(src), int(dst)
        key = (src, dst) if src < dst else (dst, src)
        e = self._edge_index.get(key)
        if e is None:
            e = len(self.edges)
            self._edge_index[key] = e
            self.edges.append((src, dst))
        return 2 * e if self.edges[e] == (src, dst) else 2 * e + 1

    def halfface(self, verts):
        verts = [int(v) for v in verts]
        key = tuple(sorted(verts))
        entry = self._face_index.get(key)
        if entry is None:
            f = len(self.faces)
            hes = [self.halfedge(a, b) for a, b in zip(verts, verts[1:] + verts[:1])]
            self.faces.append(hes)
            self._face_index[key] = (f, verts)
            return 2 * f
        f, stored = entry
        # same loop direction as the stored face: this side, otherwise the other
        i = stored.index(verts[0])
        return 2 * f if stored[(i + 1) % len(stored)] == verts[1] else 2 * f + 1


register_format("ovm", [".ovm"], read, {"ovm": write})

import numpy as np

from .._common import warn
from .._helpers import read, reader_map
from .._exceptions import ReadError
from ..med import read_med_multi


def add_args(parser):
    parser.add_argument("infile", type=str, help="mesh file to be read from")
    parser.add_argument(
        "--input-format",
        "-i",
        type=str,
        choices=sorted(list(reader_map.keys())),
        help="input file format",
        default=None,
    )


def info(args):
    # read mesh data
    is_med = False
    if args.input_format == "med":
        is_med = True
    else:
        lower = args.infile.lower()
        if lower.endswith(".med"):
            is_med = True

    if is_med:
        meshes, names = read_med_multi(args.infile)
        for name, mesh in zip(names, meshes):
            print(f"--- Mesh: {name} ---")
            print(mesh)

            # perform the same consistency checks per mesh
            is_consistent = True
            for cells in mesh.cells:
                if np.any(cells.data > mesh.points.shape[0]):
                    warn("Inconsistent mesh. Cells refer to nonexistent points.")
                    is_consistent = False
                    break

            if is_consistent:
                point_is_used = np.zeros(mesh.points.shape[0], dtype=bool)
                for cells in mesh.cells:
                    point_is_used[cells.data] = True
                if np.any(~point_is_used):
                    warn("Some points are not part of any cell.")
        return 0

    mesh = read(args.infile, file_format=args.input_format)
    print(mesh)

    # check if the cell arrays are consistent with the points
    is_consistent = True
    for cells in mesh.cells:
        if np.any(cells.data > mesh.points.shape[0]):
            warn("Inconsistent mesh. Cells refer to nonexistent points.")
            is_consistent = False
            break

    # check if there are redundant points
    if is_consistent:
        point_is_used = np.zeros(mesh.points.shape[0], dtype=bool)
        for cells in mesh.cells:
            point_is_used[cells.data] = True
        if np.any(~point_is_used):
            warn("Some points are not part of any cell.")

    return 0

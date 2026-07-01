import numpy as np

from .._common import warn
from .._helpers import read, reader_map


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

    # Detect multi-mesh MED files 
    if args.infile.endswith(".med"):
        import h5py
        with h5py.File(args.infile, "r") as f:
            mesh_names = list(f["ENS_MAA"].keys())
        
        if len(mesh_names) > 1:
            from meshlane.med._medmulti import read_med_multi
            meshes, names = read_med_multi(args.infile)
            for mesh, name in zip(meshes, names):
                print(f"\n{'='*40}")
                print(f"  Maillage : '{name}'")
                print(f"{'='*40}")
                print(mesh)
            return

    # read mesh data
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

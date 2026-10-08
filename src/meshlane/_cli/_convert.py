import sys

import numpy as np

from .._common import error, warn
from .._helpers import _writer_map, read, reader_map, write
from ._multi import multi_format, read_multi, write_multi


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
    parser.add_argument(
        "--output-format",
        "-o",
        type=str,
        choices=sorted(list(_writer_map.keys())),
        help="output file format",
        default=None,
    )
    parser.add_argument(
        "--ascii",
        "-a",
        action="store_true",
        help="write in ASCII format variant (where applicable, default: binary)",
    )
    parser.add_argument("outfile", type=str, help="mesh file to be written to")
    parser.add_argument(
        "--float-format",
        "-f",
        type=str,
        help="float format used in output ASCII files (default: .16e)",
    )
    parser.add_argument(
        "--sets-to-int-data",
        "-s",
        action="store_true",
        help="if possible, convert sets to integer data (useful if the output type does not support sets)",
    )
    parser.add_argument(
        "--int-data-to-sets",
        "-d",
        action="store_true",
        help="if possible, convert integer data to sets (useful if the output type does not support integer data)",
    )
    parser.add_argument(
        "--remove-duplicates",
        action="store_true",
        help="remove duplicate cells (cells sharing the same node set); off by default",
    )


def _process(mesh, args):
    # Some converters (like VTK) require `points` to be contiguous.
    mesh.points = np.ascontiguousarray(mesh.points)

    # Duplicate (coincident) cells: remove on request, otherwise just warn.
    if args.remove_duplicates:
        n_dup = mesh.remove_duplicate_cells()
        if n_dup:
            print(f"Removed {n_dup} duplicate cell(s) with identical node sets.")
    else:
        n_dup = mesh.remove_duplicate_cells(dry_run=True)
        if n_dup:
            warn(
                f"{n_dup} duplicate cell(s) with identical node sets were detected. "
                "Coincident cells may cause connectivity errors in some solvers. "
                "To remove them, rerun the conversion with the --remove-duplicates "
                "option."
            )

    if args.sets_to_int_data:
        mesh.point_sets_to_data()
        mesh.cell_sets_to_data()

    if args.int_data_to_sets:
        for key in mesh.point_data:
            mesh.point_data_to_sets(key)
        for key in mesh.cell_data:
            mesh.cell_data_to_sets(key)


def convert(args):
    # read mesh data
    print(f"Reading '{args.infile}' (large meshes may take a while)...", flush=True)
    if multi_format(args.infile, args.input_format):
        # MED files may hold several meshes and SU2 files several zones: read
        # them all (also fine for one)
        meshes, mesh_names = read_multi(args.infile, args.input_format)
    else:
        meshes, mesh_names = [read(args.infile, file_format=args.input_format)], None

    for mesh in meshes:
        _process(mesh, args)

    # write it out
    print(f"Writing '{args.outfile}'...", flush=True)
    if len(meshes) > 1:
        # more than one mesh: only MED (meshes) and SU2 (zones) can store them all
        if not multi_format(args.outfile, args.output_format):
            error(
                f"Input has {len(meshes)} meshes, but the output format cannot hold "
                "more than one. Convert to MED (.med) or SU2 (.su2, one zone per "
                "mesh) to keep all of them."
            )
            sys.exit(1)
        write_multi(args.outfile, meshes, mesh_names, args.output_format)
    else:
        kwargs = {"file_format": args.output_format}
        if args.float_format is not None:
            kwargs["float_fmt"] = args.float_format
        if args.ascii:
            kwargs["binary"] = False
        write(args.outfile, meshes[0], **kwargs)
    print("Done.")

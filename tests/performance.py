import os
import pathlib
import tempfile
import time
import tracemalloc

import dufte
import matplotlib.pyplot as plt
import meshzoo
import numpy as np

import meshlane


def generate_triangular_mesh():
    p = pathlib.Path("sphere.xdmf")
    if pathlib.Path.is_file(p):
        mesh = meshlane.read(p)
    else:
        points, cells = meshzoo.icosa_sphere(300)
        mesh = meshlane.Mesh(points, {"triangle": cells})
        mesh.write(p)
    return mesh


def generate_tetrahedral_mesh():
    """Generates a fairly large mesh."""
    if pathlib.Path.is_file("cache.xdmf"):
        mesh = meshlane.read("cache.xdmf")
    else:
        import pygalmesh

        s = pygalmesh.Ball([0, 0, 0], 1.0)
        mesh = pygalmesh.generate_mesh(s, cell_size=2.0e-2, verbose=True)
        # mesh = pygalmesh.generate_mesh(s, cell_size=1.0e-1, verbose=True)
        mesh.cells = {"tetra": mesh.cells["tetra"]}
        mesh.point_data = []
        mesh.cell_data = {"tetra": {}}
        mesh.write("cache.xdmf")
    return mesh


def plot_speed(names, elapsed_write, elapsed_read):
    plt.style.use(dufte.style)

    names = np.asarray(names)
    elapsed_write = np.asarray(elapsed_write)
    elapsed_read = np.asarray(elapsed_read)

    fig, ax = plt.subplots(1, 2, figsize=(12, 8))

    idx = np.argsort(elapsed_write)[::-1]
    ax[0].barh(range(len(names)), elapsed_write[idx], align="center")
    ax[0].set_yticks(range(len(names)))
    ax[0].set_yticklabels(names[idx])
    ax[0].set_xlabel("time (s)")
    ax[0].set_title("write")
    ax[0].grid()

    idx = np.argsort(elapsed_read)[::-1]
    ax[1].barh(range(len(names)), elapsed_read[idx], align="center")
    ax[1].set_yticks(range(len(names)))
    ax[1].set_yticklabels(names[idx])
    ax[1].set_xlabel("time (s)")
    ax[1].set_title("read")
    ax[1].grid()

    fig.tight_layout()
    # plt.show()
    fig.savefig("performance.svg", transparent=True, bbox_inches="tight")
    plt.close()


def plot_file_sizes(names, file_sizes, mem_size):
    idx = np.argsort(file_sizes)
    file_sizes = [file_sizes[i] for i in idx]
    names = [names[i] for i in idx]

    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    y_pos = np.arange(len(file_sizes))
    ax.barh(y_pos, file_sizes, align="center")
    #
    ylim = ax.get_ylim()
    plt.plot(
        [mem_size, mem_size], [-2, len(file_sizes) + 2], "C3", linewidth=2.0, zorder=0
    )
    ax.set_ylim(ylim)
    #
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel("file size [MB]")
    ax.set_title("file sizes")
    plt.grid()
    # plt.show()
    plt.savefig("filesizes.svg", transparent=True, bbox_inches="tight")
    plt.close()


def plot_memory_usage(names, peak_memory_write, peak_memory_read, mem_size):
    names = np.asarray(names)
    peak_memory_write = np.asarray(peak_memory_write)
    peak_memory_read = np.asarray(peak_memory_read)

    fig, ax = plt.subplots(1, 2, figsize=(12, 8))

    idx = np.argsort(peak_memory_write)[::-1]
    ax[0].barh(range(len(names)), peak_memory_write[idx], align="center")
    ax[0].set_yticks(range(len(names)))
    ax[0].set_yticklabels(names[idx])
    ax[0].set_xlabel("peak memory [MB]")
    ax[0].set_title("write")
    ax[0].grid()
    # plot memsize of mesh
    ylim = ax[0].get_ylim()
    ax[0].plot(
        [mem_size, mem_size], [-2, len(names) + 2], "C3", linewidth=2.0, zorder=0
    )
    ax[0].set_ylim(ylim)

    idx = np.argsort(peak_memory_read)[::-1]
    ax[1].barh(range(len(names)), peak_memory_read[idx], align="center")
    ax[1].set_yticks(range(len(names)))
    ax[1].set_yticklabels(names[idx])
    ax[1].set_xlabel("peak memory [MB]")
    ax[1].set_title("read")
    ax[1].grid()
    # plot memsize of mesh
    ylim = ax[1].get_ylim()
    ax[1].plot(
        [mem_size, mem_size], [-2, len(names) + 2], "C3", linewidth=2.0, zorder=0
    )
    ax[1].set_ylim(ylim)

    fig.tight_layout()
    # plt.show()
    fig.savefig("memory.svg", transparent=True, bbox_inches="tight")
    plt.close()


def read_write(plot=False):
    # mesh = generate_tetrahedral_mesh()
    mesh = generate_triangular_mesh()
    print(mesh)
    mem_size = mesh.points.nbytes + mesh.cells[0].data.nbytes
    mem_size /= 1024.0**2
    print(f"mem_size: {mem_size:.2f} MB")

    formats = {
        "Abaqus": (meshlane.abaqus.write, meshlane.abaqus.read, ["out.inp"]),
        "Ansys (ASCII)": (
            lambda f, m: meshlane.ansys.write(f, m, binary=False),
            meshlane.ansys.read,
            ["out.ans"],
        ),
        # "Ansys (binary)": (
        #     lambda f, m: meshlane.ansys.write(f, m, binary=True),
        #     meshlane.ansys.read,
        #     ["out.ans"],
        # ),
        "AVS-UCD": (meshlane.avsucd.write, meshlane.avsucd.read, ["out.ucd"]),
        # "CGNS": (meshlane.cgns.write, meshlane.cgns.read, ["out.cgns"]),
        "Dolfin-XML": (meshlane.dolfin.write, meshlane.dolfin.read, ["out.xml"]),
        "Exodus": (meshlane.exodus.write, meshlane.exodus.read, ["out.e"]),
        # "FLAC3D": (meshlane.flac3d.write, meshlane.flac3d.read, ["out.f3grid"]),
        "Gmsh 4.1 (ASCII)": (
            lambda f, m: meshlane.gmsh.write(f, m, binary=False),
            meshlane.gmsh.read,
            ["out.msh"],
        ),
        "Gmsh 4.1 (binary)": (
            lambda f, m: meshlane.gmsh.write(f, m, binary=True),
            meshlane.gmsh.read,
            ["out.msh"],
        ),
        "MDPA": (meshlane.mdpa.write, meshlane.mdpa.read, ["out.mdpa"]),
        "MED": (meshlane.med.write, meshlane.med.read, ["out.med"]),
        "Medit": (meshlane.medit.write, meshlane.medit.read, ["out.mesh"]),
        "MOAB": (meshlane.h5m.write, meshlane.h5m.read, ["out.h5m"]),
        "Nastran": (meshlane.nastran.write, meshlane.nastran.read, ["out.bdf"]),
        "Netgen": (meshlane.netgen.write, meshlane.netgen.read, ["out.vol"]),
        "OFF": (meshlane.off.write, meshlane.off.read, ["out.off"]),
        "Permas": (meshlane.permas.write, meshlane.permas.read, ["out.dato"]),
        "PLY (binary)": (
            lambda f, m: meshlane.ply.write(f, m, binary=True),
            meshlane.ply.read,
            ["out.ply"],
        ),
        "PLY (ASCII)": (
            lambda f, m: meshlane.ply.write(f, m, binary=False),
            meshlane.ply.read,
            ["out.ply"],
        ),
        "STL (binary)": (
            lambda f, m: meshlane.stl.write(f, m, binary=True),
            meshlane.stl.read,
            ["out.stl"],
        ),
        "STL (ASCII)": (
            lambda f, m: meshlane.stl.write(f, m, binary=False),
            meshlane.stl.read,
            ["out.stl"],
        ),
        # "TetGen": (meshlane.tetgen.write, meshlane.tetgen.read, ["out.node", "out.ele"],),
        "VTK (binary)": (
            lambda f, m: meshlane.vtk.write(f, m, binary=True),
            meshlane.vtk.read,
            ["out.vtk"],
        ),
        "VTK (ASCII)": (
            lambda f, m: meshlane.vtk.write(f, m, binary=False),
            meshlane.vtk.read,
            ["out.vtk"],
        ),
        "VTU (binary, uncompressed)": (
            lambda f, m: meshlane.vtu.write(f, m, binary=True, compression=None),
            meshlane.vtu.read,
            ["out.vtu"],
        ),
        "VTU (binary, zlib)": (
            lambda f, m: meshlane.vtu.write(f, m, binary=True, compression="zlib"),
            meshlane.vtu.read,
            ["out.vtu"],
        ),
        "VTU (binary, LZMA)": (
            lambda f, m: meshlane.vtu.write(f, m, binary=True, compression="lzma"),
            meshlane.vtu.read,
            ["out.vtu"],
        ),
        "VTU (ASCII)": (
            lambda f, m: meshlane.vtu.write(f, m, binary=False),
            meshlane.vtu.read,
            ["out.vtu"],
        ),
        "Wavefront .obj": (meshlane.obj.write, meshlane.obj.read, ["out.obj"]),
        # "wkt": ".wkt",
        "XDMF (binary)": (
            lambda f, m: meshlane.xdmf.write(f, m, data_format="Binary"),
            meshlane.xdmf.read,
            ["out.xdmf", "out0.bin", "out1.bin"],
        ),
        "XDMF (HDF, GZIP)": (
            lambda f, m: meshlane.xdmf.write(f, m, data_format="HDF", compression="gzip"),
            meshlane.xdmf.read,
            ["out.xdmf", "out.h5"],
        ),
        "XDMF (HDF, uncompressed)": (
            lambda f, m: meshlane.xdmf.write(f, m, data_format="HDF", compression=None),
            meshlane.xdmf.read,
            ["out.xdmf", "out.h5"],
        ),
        "XDMF (XML)": (
            lambda f, m: meshlane.xdmf.write(f, m, data_format="XML"),
            meshlane.xdmf.read,
            ["out.xdmf"],
        ),
    }

    # formats = {
    #     # "VTK (ASCII)": formats["VTK (ASCII)"],
    #     # "VTK (binary)": formats["VTK (binary)"],
    #     # "VTU (ASCII)": formats["VTU (ASCII)"],
    #     # "VTU (binary)": formats["VTU (binary)"],
    #     # "Gmsh 4.1 (binary)": formats["Gmsh 4.1 (binary)"],
    #     # "FLAC3D": formats["FLAC3D"],
    #     "MDPA": formats["MDPA"],
    # }

    # max_key_length = max(len(key) for key in formats)

    elapsed_write = []
    elapsed_read = []
    file_sizes = []
    peak_memory_write = []
    peak_memory_read = []

    print()
    print(
        "format                      "
        + "write (s)    "
        + "read(s)      "
        + "file size    "
        + "write mem    "
        + "read mem "
    )
    print()
    with tempfile.TemporaryDirectory() as directory:
        directory = pathlib.Path(directory)
        for name, (writer, reader, filenames) in formats.items():
            filename = directory / filenames[0]

            tracemalloc.start()
            t = time.time()
            writer(filename, mesh)
            # snapshot = tracemalloc.take_snapshot()
            elapsed_write.append(time.time() - t)
            peak_memory_write.append(tracemalloc.get_traced_memory()[1])
            tracemalloc.stop()

            file_sizes.append(sum(os.stat(directory / f).st_size for f in filenames))

            tracemalloc.start()
            t = time.time()
            reader(filename)
            elapsed_read.append(time.time() - t)
            peak_memory_read.append(tracemalloc.get_traced_memory()[1])
            tracemalloc.stop()
            print(
                "{:<26}  {:e} {:e} {:e} {:e} {:e}".format(
                    name,
                    elapsed_write[-1],
                    elapsed_read[-1],
                    file_sizes[-1] / 1024.0**2,
                    peak_memory_write[-1] / 1024.0**2,
                    peak_memory_read[-1] / 1024.0**2,
                )
            )

    names = list(formats.keys())
    # convert to MB
    file_sizes = np.array(file_sizes)
    file_sizes = file_sizes / 1024.0**2
    peak_memory_write = np.array(peak_memory_write)
    peak_memory_write = peak_memory_write / 1024.0**2
    peak_memory_read = np.array(peak_memory_read)
    peak_memory_read = peak_memory_read / 1024.0**2

    if plot:
        plot_speed(names, elapsed_write, elapsed_read)
        plot_file_sizes(names, file_sizes, mem_size)
        plot_memory_usage(names, peak_memory_write, peak_memory_read, mem_size)


if __name__ == "__main__":
    read_write(plot=True)

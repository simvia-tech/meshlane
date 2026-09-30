"""Formats whose files can hold several meshes: MED (meshes), SU2 (zones)."""

from pathlib import Path

from .. import su2
from ..med import read_med_multi, write_med_multi

_readers = {"med": read_med_multi, "su2": su2.read_multi}
_writers = {"med": write_med_multi, "su2": su2.write_multi}
_extensions = {".med": "med", ".su2": "su2"}


def multi_format(path, file_format=None):
    """The multi-mesh format of ``path`` ("med", "su2"), or None."""
    if file_format is not None:
        return file_format if file_format in _readers else None
    return _extensions.get(Path(path).suffix.lower())


def read_multi(path, file_format=None):
    """``(meshes, names)`` of a MED or SU2 file."""
    return _readers[multi_format(path, file_format)](path)


def write_multi(path, meshes, mesh_names=None, file_format=None):
    _writers[multi_format(path, file_format)](path, meshes, mesh_names=mesh_names)

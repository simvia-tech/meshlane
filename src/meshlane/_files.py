import io
from contextlib import contextmanager

def is_buffer(obj, mode):
    return ("r" in mode and hasattr(obj, "read")) or (
        "w" in mode and hasattr(obj, "write")
    )


@contextmanager
def open_file(path_or_buf, mode="r"):
    if is_buffer(path_or_buf, mode):
        yield path_or_buf
    elif "r" in mode and "b" not in mode:
        with open(path_or_buf,"rb") as fb:
            raw = fb.read()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("cp1252", errors="replace")
        f = io.StringIO(text)
        f.name = str(path_or_buf)
        yield f
    else:
        with open(path_or_buf, mode) as f:
            yield f

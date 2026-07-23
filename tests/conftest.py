import os
import tempfile
from pathlib import Path


if "MPLCONFIGDIR" not in os.environ:
    cache_name = f"scampi-matplotlib-{os.getuid()}-{os.getpid()}"
    os.environ["MPLCONFIGDIR"] = str(Path(tempfile.gettempdir()) / cache_name)

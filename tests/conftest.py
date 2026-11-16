"""Settings shared by every test (ADR-061).

The suite runs in parallel, one pytest-xdist worker per CPU core (`addopts` in `pyproject.toml`).
NumPy's linear-algebra library (OpenBLAS) would otherwise start its own pool of threads in every
worker, so 12 workers would run about 12 x 12 threads on 12 hardware threads and wait on each
other. One thread per worker made the whole suite about 1.7 times faster (4 min 36 s -> 2 min
40 s) with the same results. These lines must run before NumPy is first imported, which is why
they live here: pytest imports this file before any test module. Processes that tests start
(`nd train` launched from the API) inherit the setting.
"""

import os

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

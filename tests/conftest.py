import uuid
from pathlib import Path

import pytest


@pytest.fixture()
def local_tmp_path ():
    base = Path.cwd() / "tests" / "tmp_work"
    base.mkdir(parents=True, exist_ok=True)
    tmp = base / uuid.uuid4().hex
    tmp.mkdir(parents=True, exist_ok=True)
    yield tmp

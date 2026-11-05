from hashlib import sha256
import json
from pathlib import Path

def digest(path):
    h = sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(2**20),b""):
            h.update(block)
    return h.hexdigest()


def write_json(path,value):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding="utf8")
    tmp.replace(path)


def identity(paths, config):
    return dict(config=config,files={str(Path(p).name):digest(p) for p in paths})




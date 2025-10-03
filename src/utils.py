from contextlib import contextmanager
import time
import json

@contextmanager
def timer(msg):
    start = time.time()
    print(f'[{msg}] start.')
    yield
    elapsed_time = time.time() - start
    print(f'[{msg}] done in {elapsed_time / 60:.2f} min.')

def load_config(config_path: str) :
    with open(config_path) as f:
        config = json.load(f)
    return config
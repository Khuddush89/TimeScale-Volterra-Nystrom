import os, sys, json, time, platform
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)
import numpy as np


def save_json(name, obj):
    def conv(o):
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, np.ndarray): return o.tolist()
        raise TypeError(type(o))
    with open(os.path.join(RES, name + ".json"), "w") as fh:
        json.dump(obj, fh, default=conv, indent=1)


def load_json(name):
    with open(os.path.join(RES, name + ".json")) as fh:
        return json.load(fh)


def env_info():
    import scipy, jax, matplotlib
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                jax=jax.__version__, matplotlib=matplotlib.__version__, cpu=platform.processor() or platform.machine(),
                cores=os.cpu_count())

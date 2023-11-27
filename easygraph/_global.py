from pathlib import Path


def get_eg_cache_root():
    root = Path.home() / Path(".easygraph/")
    root.mkdir(parents=True, exist_ok=True)
    return root


AUTHOR_EMAIL = "bdye22@m.fudan.edu.cn"
# global paths
CACHE_ROOT = get_eg_cache_root()
DATASETS_ROOT = CACHE_ROOT / "datasets"
REMOTE_ROOT = "https://download.moon-lab.tech:28501/"
REMOTE_DATASETS_ROOT = REMOTE_ROOT + "datasets/"

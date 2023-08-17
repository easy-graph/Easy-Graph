try:
    from .common import compose_pipes
    from .common import to_bool_tensor
    from .common import to_long_tensor
    from .common import to_tensor
    from .normalize import min_max_scaler
    from .normalize import norm_ft
except:
    print(
        "Warning raise in module:datapipe. Please install Pytorch before you use"
        " functions related to nueral network"
    )

from .loader import load_from_json
from .loader import load_from_pickle
from .loader import load_from_txt


# __all__ = [
#     "compose_pipes",
#     "norm_ft",
#     "min_max_scaler",
#     "to_tensor",
#     "to_bool_tensor",
#     "to_long_tensor",
#     "load_from_pickle",
#     "load_from_json",
#     "load_from_txt",
# ]

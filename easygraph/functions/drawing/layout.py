from typing import List

from .simulator import Simulator
from .utils import edge_list_to_incidence_matrix
from .utils import init_pos


def force_layout(
    num_v: int,
    e_list: List[tuple],
    push_v_strength: float,
    push_e_strength: float,
    pull_e_strength: float,
    pull_center_strength: float,
):
    import numpy as np

    v_coor = init_pos(num_v, scale=5)
    assert v_coor.max() <= 5.0 and v_coor.min() >= -5.0
    centers = [np.array([0, 0])]
    sim = Simulator(
        nums=num_v,
        forces={
            Simulator.NODE_ATTRACTION: pull_e_strength,
            Simulator.NODE_REPULSION: push_v_strength,
            Simulator.EDGE_REPULSION: push_e_strength,
            Simulator.CENTER_GRAVITY: pull_center_strength,
        },
        centers=centers,
    )
    v_coor = sim.simulate(v_coor, edge_list_to_incidence_matrix(num_v, e_list))
    v_coor = (v_coor - v_coor.min(0)) / (v_coor.max(0) - v_coor.min(0)) * 0.8 + 0.1
    return v_coor

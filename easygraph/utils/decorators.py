__all__ = [
    "only_implemented_for_UnDirected_graph",
    "only_implemented_for_Directed_graph"
]

def only_implemented_for_UnDirected_graph(func):
    # print("--------{:<40}: Only Implemented For UnDirected Graph--------".format(func.__name__))
    return func

def only_implemented_for_Directed_graph(func):
    # print("--------{:<40}: Only Implemented For Directed Graph--------".format(func.__name__))
    return func

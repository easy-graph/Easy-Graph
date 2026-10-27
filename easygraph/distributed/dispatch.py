

def dispatch(cpp_method_name, *args, **kwargs):
    if cpp_method_name == "cpp_pagerank":
        from .pagerank import pagerank

        return pagerank(*args, **kwargs)
    raise NotImplementedError(
        f"{cpp_method_name or 'This algorithm'} has no distributed implementation. "
        "The initial distributed backend supports PageRank."
    )

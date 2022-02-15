__all__ = [
    "split_len",
    "split"
]

def split_len(nodes, step=30000):
    ret = []
    length = len(nodes)
    for i in range(0, length, step):
        ret.append(nodes[i: i + step])
    if len(ret[-1]) * 3 < step:
        ret[-2] = ret[-2] + ret[-1]
        ret = ret[:-1]
    return ret

def split(nodes, n):
    ret = []
    length = len(nodes)  # 总长
    step = int(length / n) + 1  # 每份的长度
    for i in range(0, length, step):
        ret.append(nodes[i:i + step])
    return ret
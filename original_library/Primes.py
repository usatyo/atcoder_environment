from collections import defaultdict
from math import sqrt


def primes(n):
    """素数のリスト生成

    Args:
        n (int): 調べる最大整数

    Returns:
        list<int>: n 以下の素数のリスト
    """
    ret = []
    dic = defaultdict(bool)
    root = int(sqrt(n)) + 1
    for i in range(2, root):
        if dic[i]:
            continue
        ret.append(i)
        for j in range(2, int(n / i) + 1):
            dic[j * i] = True
    for i in range(root, n):
        if dic[i]:
            continue
        ret.append(i)

    return ret

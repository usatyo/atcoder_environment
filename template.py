from itertools import combinations, permutations, product
from math import ceil, gcd, lcm
from string import ascii_lowercase, ascii_uppercase
from sys import setrecursionlimit, stdin

# UnionFind
from atcoder.dsu import DSU

# crt: 中国剰余定理
from atcoder.math import crt, floor_sum, inv_mod

# object であるため、sum や [Modint(0)] * n などの初期化がバグる
# 遅いので使わない
from atcoder.modint import ModContext, Modint

# 強連結成分分解
from atcoder.scc import SCCGraph

# セグ木
from atcoder.segtree import SegTree
from atcoder.string import suffix_array, lcp_array

# 遅延セグ木
# op: 区間取得の演算方法
# e: opの単位元
# mapping: 各ノードでの値に対する操作の解決方法 (渡される操作, ノードでの計算途中の値) => 計算済み値
#     区間加算取得など、区間の幅の情報が必要になる場合は値の方に情報を持たせる、名前 map は被るからダメ！
# composition: ひとつのノードに複数の操作が来た場合の解決方法 (2つ目に渡される操作, 1つ目に渡される操作) => 合成操作
#     可換でない場合に注意
# id: mapping に対する恒等写像(ほぼ単位元)
from atcoder.lazysegtree import LazySegTree

# binary index tree
# 転倒数を O(N) で求める場合など
# for i in range(n):
#     fa.add(a[i], 1)
#     inva += i - fa.sum(0, a[i])
from atcoder.fenwicktree import FenwickTree

# 最大フロー、最小カット
# https://github.com/shakayami/ACL-for-python/wiki/maxflow
from atcoder.maxflow import MFGraph


# 最小費用流
from atcoder.mincostflow import MCFGraph


# 論理式の充足可能問題（到達不能かどうか）
from atcoder.twosat import TwoSAT


from sortedcontainers import sortedset, SortedList, SortedDict


setrecursionlimit(10**7)
standard_prime = 998244353


def main(input):
    ans = 0
    return ans


def _format(ans):
    if type(ans) == str or type(ans) == int or type(ans) == float:
        return str(ans)
    elif type(ans) == list or type(ans) == tuple:
        if type(ans[0]) == list or type(ans[0]) == tuple:
            return "\n".join([(" ".join([str(x) for x in a])) for a in ans])
        else:
            return "\n".join(ans)
    else:
        raise TypeError("Return type is not supported.")


if __name__ == "__main__":
    ans = main(lambda: stdin.readline().rstrip())
    ans = _format(ans)
    print(ans)

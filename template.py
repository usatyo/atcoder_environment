from itertools import combinations, permutations, product
from math import ceil, gcd, lcm
from string import ascii_lowercase, ascii_uppercase
from sys import setrecursionlimit

# UnionFind
from atcoder.dsu import DSU

# crt: 中国剰余定理
from atcoder.math import crt, floor_sum, inv_mod
from atcoder.modint import ModContext, Modint

# 強連結成分分解
from atcoder.scc import SCCGraph

# セグ木
from atcoder.segtree import SegTree
from atcoder.string import suffix_array

setrecursionlimit(10**7)
standard_prime = 998244353

# with ModContext(standard_prime):

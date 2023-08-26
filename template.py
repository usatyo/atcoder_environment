import math
from bisect import bisect_left, bisect_right
from collections import defaultdict, deque
from copy import deepcopy
from heapq import heapify, heappop, heappush
from itertools import combinations, permutations, product
from math import ceil, gcd
from string import ascii_lowercase, ascii_uppercase
from sys import exit, setrecursionlimit

# UnionFind
from atcoder.dsu import DSU

# crt: 中国剰余定理
from atcoder.math import crt, floor_sum, inv_mod

# 強連結成分分解
from atcoder.scc import SCCGraph

# セグ木
from atcoder.segtree import SegTree
from atcoder.string import suffix_array

setrecursionlimit(10**7)

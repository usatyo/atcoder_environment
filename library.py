from collections import defaultdict
from heapq import heapify, heappop, heappush


class Dijkstra:
    def __init__(self) -> None:
        """初期化"""
        self.e = defaultdict(list)

    def add(self, u, v, d):
        """辺を追加

        Args:
            u (int): 頂点 1
            v (int): 頂点 2
            d (int): コスト
        """
        self.e[u].append([v, d])
        self.e[v].append([u, d])

    def delete(self, u, v):
        """辺を削除

        Args:
            u (int): 頂点 1
            v (int): 頂点 2
        """
        self.e[u] = [_ for _ in self.e[u] if _[0] != v]
        self.e[v] = [_ for _ in self.e[v] if _[0] != u]

    def search(self, s):
        """最短距離探索 O((E + V)logV)

        Args:
            s (int): 始点

        Returns:
            int[]: 始点から各点までの最短距離
        """
        d = defaultdict(lambda: float("inf"))
        d[s] = 0
        q = []
        heappush(q, (0, s))
        v = defaultdict(bool)
        while len(q):
            k, u = heappop(q)
            if v[u]:
                continue
            v[u] = True

            for uv, ud in self.e[u]:
                if v[uv]:
                    continue
                vd = k + ud
                if d[uv] > vd:
                    d[uv] = vd
                    heappush(q, (vd, uv))

        return d


class UnionFind:
    def __init__(self, n):
        """初期化

        Args:
            n (int): 頂点数
        """
        self.par = [-1] * n
        self.rank = [0] * n
        self.siz = [1] * n

    def root(self, x):
        """根を求める

        Args:
            x (int): 頂点

        Returns:
            int: 根の番号
        """
        if self.par[x] == -1:
            return x
        else:
            self.par[x] = self.root(self.par[x])
            return self.par[x]

    def issame(self, x, y):
        """同じグループかどうか

        Args:
            x (int): 頂点 1
            y (int): 頂点 2

        Returns:
            bool: グループ一致なら True
        """
        return self.root(x) == self.root(y)

    def unite(self, x, y):
        """グループの併合

        Args:
            x (int): 頂点 1
            y (int): 頂点 2

        Returns:
            bool: すでに同じグループならFalse
        """
        rx = self.root(x)
        ry = self.root(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.par[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        self.siz[rx] += self.siz[ry]
        return True

    def size(self, x):
        """サイズを求める

        Args:
            x (int): 頂点

        Returns:
            int: グループのサイズ
        """
        return self.siz[self.root(x)]


class WarshallFloyd:
    """ワーシャルフロイド法 O(V^3)"""

    def __init__(self, N):
        """初期化

        Args:
            N (int): 頂点数
        """
        self.N = N
        self.d = [
            [float("inf") for i in range(N)] for i in range(N)
        ]  # d[u][v] : 辺uvのコスト(存在しないときはinf)

    def add(self, u, v, c, directed=False):
        """辺を追加

        Args:
            u (int): 始点
            v (int): 終点
            c (int): コスト
            directed (bool, optional): 有向グラフであれば True に. Defaults to False.
        """
        if directed is False:
            self.d[u][v] = c
            self.d[v][u] = c
        else:
            self.d[u][v] = c

    def WarshallFloyd_search(self):
        """最短距離検索

        Returns:
            int[][]: d[i][j] で i から j への最短距離. d[i][i] < 0 なら、グラフは負のサイクルを持つ
        """
        for k in range(self.N):
            for i in range(self.N):
                for j in range(self.N):
                    self.d[i][j] = min(self.d[i][j], self.d[i][k] + self.d[k][j])
        hasNegativeCycle = False
        for i in range(self.N):
            if self.d[i][i] < 0:
                hasNegativeCycle = True
                break
        for i in range(self.N):
            self.d[i][i] = 0
        return hasNegativeCycle, self.d


def scc(N, edges):
    """強連結成分分解 O(E + V)

    Args:
        N (int): 頂点数
        edges (int[][]): edges[i][0] が始点、edges[i][1] が終点

    Returns:
        int[][]: 強連結成分ごとに分割された頂点の配列
    """
    M = len(edges)
    start = [0] * (N + 1)
    elist = [0] * M
    for e in edges:
        start[e[0] + 1] += 1
    for i in range(1, N + 1):
        start[i] += start[i - 1]
    counter = start[:]
    for e in edges:
        elist[counter[e[0]]] = e[1]
        counter[e[0]] += 1
    visited = []
    low = [0] * N
    Ord = [-1] * N
    ids = [0] * N
    NG = [0, 0]

    def dfs(v):
        stack = [(v, -1, 0), (v, -1, 1)]
        while stack:
            v, bef, t = stack.pop()
            if t:
                if bef != -1 and Ord[v] != -1:
                    low[bef] = min(low[bef], Ord[v])
                    stack.pop()
                    continue
                low[v] = NG[0]
                Ord[v] = NG[0]
                NG[0] += 1
                visited.append(v)
                for i in range(start[v], start[v + 1]):
                    to = elist[i]
                    if Ord[to] == -1:
                        stack.append((to, v, 0))
                        stack.append((to, v, 1))
                    else:
                        low[v] = min(low[v], Ord[to])
            else:
                if low[v] == Ord[v]:
                    while True:
                        u = visited.pop()
                        Ord[u] = N
                        ids[u] = NG[1]
                        if u == v:
                            break
                    NG[1] += 1
                low[bef] = min(low[bef], low[v])

    for i in range(N):
        if Ord[i] == -1:
            dfs(i)
    for i in range(N):
        ids[i] = NG[1] - 1 - ids[i]
    group_num = NG[1]
    counts = [0] * group_num
    for x in ids:
        counts[x] += 1
    groups = [[] for i in range(group_num)]
    for i in range(N):
        groups[ids[i]].append(i)
    return groups


class segtree:
    """セグメント木"""

    n = 1
    size = 1
    log = 2
    d = [0]
    op = None
    e = 10**15

    def __init__(self, V, OP, E):
        """初期化

        Args:
            V (int[]): 配列の初期値
            OP (function): 配列に加える操作(二項演算)
            E (any)): 操作の単位元
        """
        self.n = len(V)
        self.op = OP
        self.e = E
        self.log = (self.n - 1).bit_length()
        self.size = 1 << self.log
        self.d = [E for _ in range(2 * self.size)]
        for i in range(self.n):
            self.d[self.size + i] = V[i]
        for i in range(self.size - 1, 0, -1):
            self.update(i)

    def set(self, p, x):
        """配列の要素を更新

        Args:
            p (int): 更新する位置
            x (any): 更新する値
        """
        assert 0 <= p and p < self.n
        p += self.size
        self.d[p] = x
        for i in range(1, self.log + 1):
            self.update(p >> i)

    def get(self, p):
        """配列の要素を取得

        Args:
            p (int): 取得する位置

        Returns:
            any: 取得した結果の値
        """
        assert 0 <= p and p < self.n
        return self.d[p + self.size]

    def prod(self, l, r):
        """操作を実行

        Args:
            l (int): 適用範囲はこの値以上
            r (int): 適用範囲はこの値未満

        Returns:
            any: 操作の結果
        """
        assert 0 <= l and l <= r and r <= self.n
        sml = self.e
        smr = self.e
        l += self.size
        r += self.size
        while l < r:
            if l & 1:
                sml = self.op(sml, self.d[l])
                l += 1
            if r & 1:
                smr = self.op(self.d[r - 1], smr)
                r -= 1
            l >>= 1
            r >>= 1
        return self.op(sml, smr)

    def max_right(self, l, f):
        assert 0 <= l and l <= self.n
        assert f(self.e)
        if l == self.n:
            return self.n
        l += self.size
        sm = self.e
        while 1:
            while l % 2 == 0:
                l >>= 1
            if not (f(self.op(sm, self.d[l]))):
                while l < self.size:
                    l = 2 * l
                    if f(self.op(sm, self.d[l])):
                        sm = self.op(sm, self.d[l])
                        l += 1
                return l - self.size
            sm = self.op(sm, self.d[l])
            l += 1
            if (l & -l) == l:
                break
        return self.n

    def min_left(self, r, f):
        assert 0 <= r and r < self.n
        assert f(self.e)
        if r == 0:
            return 0
        r += self.size
        sm = self.e
        while 1:
            r -= 1
            while r > 1 & (r % 2):
                r >>= 1
            if not (f(self.op(self.d[r], sm))):
                while r < self.size:
                    r = 2 * r + 1
                    if f(self.op(self.d[r], sm)):
                        sm = self.op(self.d[r], sm)
                        r -= 1
                return r + 1 - self.size
            sm = self.op(self.d[r], sm)
            if (r & -r) == r:
                break
        return 0

    def update(self, k):
        """値の更新には update ではなく set を使う"""
        self.d[k] = self.op(self.d[2 * k], self.d[2 * k + 1])

    def __str__(self):
        return str([self.get(i) for i in range(self.n)])


class LazySegmentTree:
    """
    init(init_val, ide_ele): 配列init_valで初期化 O(N)
    update(l, r, x): 区間[l, r)をxに更新 O(logN)
    query(l, r): 区間[l, r)をsegfuncしたものを返す O(logN)
    """

    def __init__(self, init_val, segfunc, ide_ele):
        """
        init_val: 配列の初期値
        segfunc: 区間にしたい操作
        ide_ele: 単位元
        num: n以上の最小の2のべき乗
        data: 値配列(1-index)
        lazy: 遅延配列(1-index)
        """
        n = len(init_val)
        self.segfunc = segfunc
        self.ide_ele = ide_ele
        self.num = 1 << (n - 1).bit_length()
        self.data = [ide_ele] * 2 * self.num
        self.lazy = [None] * 2 * self.num
        # 配列の値を葉にセット
        for i in range(n):
            self.data[self.num + i] = init_val[i]
        # 構築していく
        for i in range(self.num - 1, 0, -1):
            self.data[i] = self.segfunc(self.data[2 * i], self.data[2 * i + 1])

    def gindex(self, l, r):
        """
        伝搬する対象の区間を求める
        lm: 伝搬する必要のある最大の左閉区間
        rm: 伝搬する必要のある最大の右開区間
        """
        l += self.num
        r += self.num
        lm = l >> (l & -l).bit_length()
        rm = r >> (r & -r).bit_length()

        while r > l:
            if l <= lm:
                yield l
            if r <= rm:
                yield r
            r >>= 1
            l >>= 1
        while l:
            yield l
            l >>= 1

    def propagates(self, *ids):
        """
        遅延伝搬処理
        ids: 伝搬する対象の区間
        """
        for i in reversed(ids):
            v = self.lazy[i]
            if v is None:
                continue
            self.lazy[2 * i] = v
            self.lazy[2 * i + 1] = v
            self.data[2 * i] = v
            self.data[2 * i + 1] = v
            self.lazy[i] = None

    def update(self, l, r, x):
        """
        区間[l, r)の値をxに更新
        l, r: index(0-index)
        x: update value
        """
        (*ids,) = self.gindex(l, r)
        self.propagates(*ids)
        l += self.num
        r += self.num
        while l < r:
            if l & 1:
                self.lazy[l] = x
                self.data[l] = x
                l += 1
            if r & 1:
                self.lazy[r - 1] = x
                self.data[r - 1] = x
            r >>= 1
            l >>= 1
        for i in ids:
            self.data[i] = self.segfunc(self.data[2 * i], self.data[2 * i + 1])

    def query(self, l, r):
        """
        [l, r)のsegfuncしたものを得る
        l: index(0-index)
        r: index(0-index)
        """
        (*ids,) = self.gindex(l, r)
        self.propagates(*ids)

        res = self.ide_ele

        l += self.num
        r += self.num
        while l < r:
            if l & 1:
                res = self.segfunc(res, self.data[l])
                l += 1
            if r & 1:
                res = self.segfunc(res, self.data[r - 1])
            l >>= 1
            r >>= 1
        return res


def factorization(n):
    """n を素因数分解

    Args:
        n (int): 素因数分解の対象

    Returns:
        int[][]: [[素因数, 指数], ...]の2次元リスト
    """
    arr = []
    temp = n
    for i in range(2, int(-(-(n**0.5) // 1)) + 1):
        if temp % i == 0:
            cnt = 0
            while temp % i == 0:
                cnt += 1
                temp //= i
            arr.append([i, cnt])

    if temp != 1:
        arr.append([temp, 1])

    if arr == []:
        arr.append([n, 1])

    return arr


import math
from bisect import bisect_left, bisect_right
from typing import Generic, Iterable, Iterator, List, Tuple, TypeVar, Optional
T = TypeVar('T')

class SortedSet(Generic[T]):
    BUCKET_RATIO = 50
    REBUILD_RATIO = 170

    def _build(self, a: Optional[List[T]] = None) -> None:
        "Evenly divide `a` into buckets."
        if a is None: a = list(self)
        size = len(a)
        bucket_size = int(math.ceil(math.sqrt(size / self.BUCKET_RATIO)))
        self.a = [a[size * i // bucket_size : size * (i + 1) // bucket_size] for i in range(bucket_size)]
    
    def __init__(self, a: Iterable[T] = []) -> None:
        "Make a new SortedSet from iterable. / O(N) if sorted and unique / O(N log N)"
        a = list(a)
        self.size = len(a)
        if not all(a[i] < a[i + 1] for i in range(len(a) - 1)):
            a = sorted(set(a))
        self._build(a)

    def __iter__(self) -> Iterator[T]:
        for i in self.a:
            for j in i: yield j

    def __reversed__(self) -> Iterator[T]:
        for i in reversed(self.a):
            for j in reversed(i): yield j
    
    def __eq__(self, other) -> bool:
        return list(self) == list(other)
    
    def __len__(self) -> int:
        return self.size
    
    def __repr__(self) -> str:
        return "SortedSet" + str(self.a)
    
    def __str__(self) -> str:
        s = str(list(self))
        return "{" + s[1 : len(s) - 1] + "}"

    def _position(self, x: T) -> Tuple[List[T], int]:
        "Find the bucket and position which x should be inserted. self must not be empty."
        for a in self.a:
            if x <= a[-1]: break
        return (a, bisect_left(a, x))

    def __contains__(self, x: T) -> bool:
        if self.size == 0: return False
        a, i = self._position(x)
        return i != len(a) and a[i] == x

    def add(self, x: T) -> bool:
        "Add an element and return True if added. / O(√N)"
        if self.size == 0:
            self.a = [[x]]
            self.size = 1
            return True
        a, i = self._position(x)
        if i != len(a) and a[i] == x: return False
        a.insert(i, x)
        self.size += 1
        if len(a) > len(self.a) * self.REBUILD_RATIO:
            self._build()
        return True
    
    def _pop(self, a: List[T], i: int) -> T:
        ans = a.pop(i)
        self.size -= 1
        if not a: self._build()
        return ans

    def discard(self, x: T) -> bool:
        "Remove an element and return True if removed. / O(√N)"
        if self.size == 0: return False
        a, i = self._position(x)
        if i == len(a) or a[i] != x: return False
        self._pop(a, i)
        return True
    
    def lt(self, x: T) -> Optional[T]:
        "Find the largest element < x, or None if it doesn't exist."
        for a in reversed(self.a):
            if a[0] < x:
                return a[bisect_left(a, x) - 1]

    def le(self, x: T) -> Optional[T]:
        "Find the largest element <= x, or None if it doesn't exist."
        for a in reversed(self.a):
            if a[0] <= x:
                return a[bisect_right(a, x) - 1]

    def gt(self, x: T) -> Optional[T]:
        "Find the smallest element > x, or None if it doesn't exist."
        for a in self.a:
            if a[-1] > x:
                return a[bisect_right(a, x)]

    def ge(self, x: T) -> Optional[T]:
        "Find the smallest element >= x, or None if it doesn't exist."
        for a in self.a:
            if a[-1] >= x:
                return a[bisect_left(a, x)]
    
    def __getitem__(self, i: int) -> T:
        "Return the i-th element."
        if i < 0:
            for a in reversed(self.a):
                i += len(a)
                if i >= 0: return a[i]
        else:
            for a in self.a:
                if i < len(a): return a[i]
                i -= len(a)
        raise IndexError
    
    def pop(self, i: int = -1) -> T:
        "Pop and return the i-th element."
        if i < 0:
            for a in reversed(self.a):
                i += len(a)
                if i >= 0: return self._pop(a, i)
        else:
            for a in self.a:
                if i < len(a): return self._pop(a, i)
                i -= len(a)
        raise IndexError
    
    def index(self, x: T) -> int:
        "Count the number of elements < x."
        ans = 0
        for a in self.a:
            if a[-1] >= x:
                return ans + bisect_left(a, x)
            ans += len(a)
        return ans

    def index_right(self, x: T) -> int:
        "Count the number of elements <= x."
        ans = 0
        for a in self.a:
            if a[-1] > x:
                return ans + bisect_right(a, x)
            ans += len(a)
        return ans




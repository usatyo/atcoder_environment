from heapq import heappop, heappush


class Dijkstra:
    def __init__(self, n) -> None:
        """初期化

        Args:
            n (int): 頂点数
        """
        self.e = [[] for _ in range(n)]
        self.n = n

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
        d = [float("inf")] * self.n
        d[s] = 0
        q = []
        heappush(q, (0, s))
        v = [False] * self.n
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

    def diameter(self):
        """木の直径

        Returns:
            float: 木の直径
        """
        d = self.search(0)
        u = 0
        max_cost = -1
        for key in d.keys():
            if max_cost < d[key]:
                u = key
                max_cost = d[key]

        du = self.search(u)
        v = 0
        max_cost = 0
        for key in du.keys():
            if max_cost < du[key]:
                v = key
                max_cost = du[key]

        return max_cost


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

    def search(self):
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
from typing import Generic, Iterable, Iterator, List, Optional, Tuple, TypeVar

T = TypeVar("T")


class SortedSet(Generic[T]):
    BUCKET_RATIO = 50
    REBUILD_RATIO = 170

    def _build(self, a: Optional[List[T]] = None) -> None:
        "Evenly divide `a` into buckets."
        if a is None:
            a = list(self)
        size = len(a)
        bucket_size = int(math.ceil(math.sqrt(size / self.BUCKET_RATIO)))
        self.a = [
            a[size * i // bucket_size : size * (i + 1) // bucket_size]
            for i in range(bucket_size)
        ]

    def __init__(self, a: Iterable[T] = []) -> None:
        "Make a new SortedSet from iterable. / O(N) if sorted and unique / O(N log N)"
        a = list(a)
        self.size = len(a)
        if not all(a[i] < a[i + 1] for i in range(len(a) - 1)):
            a = sorted(set(a))
        self._build(a)

    def __iter__(self) -> Iterator[T]:
        for i in self.a:
            for j in i:
                yield j

    def __reversed__(self) -> Iterator[T]:
        for i in reversed(self.a):
            for j in reversed(i):
                yield j

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
            if x <= a[-1]:
                break
        return (a, bisect_left(a, x))

    def __contains__(self, x: T) -> bool:
        if self.size == 0:
            return False
        a, i = self._position(x)
        return i != len(a) and a[i] == x

    def add(self, x: T) -> bool:
        "Add an element and return True if added. / O(√N)"
        if self.size == 0:
            self.a = [[x]]
            self.size = 1
            return True
        a, i = self._position(x)
        if i != len(a) and a[i] == x:
            return False
        a.insert(i, x)
        self.size += 1
        if len(a) > len(self.a) * self.REBUILD_RATIO:
            self._build()
        return True

    def _pop(self, a: List[T], i: int) -> T:
        ans = a.pop(i)
        self.size -= 1
        if not a:
            self._build()
        return ans

    def discard(self, x: T) -> bool:
        "Remove an element and return True if removed. / O(√N)"
        if self.size == 0:
            return False
        a, i = self._position(x)
        if i == len(a) or a[i] != x:
            return False
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
                if i >= 0:
                    return a[i]
        else:
            for a in self.a:
                if i < len(a):
                    return a[i]
                i -= len(a)
        raise IndexError

    def pop(self, i: int = -1) -> T:
        "Pop and return the i-th element."
        if i < 0:
            for a in reversed(self.a):
                i += len(a)
                if i >= 0:
                    return self._pop(a, i)
        else:
            for a in self.a:
                if i < len(a):
                    return self._pop(a, i)
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


from collections import defaultdict, deque


class TopologicalSort:
    def __init__(self, n):
        """初期化

        Args:
            n (int): 頂点数
        """
        self.size = n
        self.edges = [[] for _ in range(self.size)]
        self.depend = [0] * self.size

    def add(self, u, v):
        """辺を追加

        Args:
            u (int): 始点
            v (int): 終点
        """
        self.edges[u].append(v)
        self.depend[v] += 1

    def search(self):
        """トポロジカルソート

        Returns:
            int[]: 結果の配列、閉路が存在すれば長さが n 未満
        """
        q = deque()
        for i in range(self.size):
            if self.depend[i] == 0:
                q.append(i)

        ans = []

        while q:
            v = q.popleft()
            ans.append(v)
            for node in self.edges[v]:
                self.depend[node] -= 1
                if self.depend[node] == 0:
                    q.append(node)

        return ans


def rotate_direction(p1, p2, p3):
    """3点の回転方向を求める

    Args:
        p1 (float[]): 点1
        p2 (float[]): 点2
        p3 (float[]): 点3

    Returns:
        int: 時計回りなら -1, 反時計回りなら1, 一直線上なら 0
    """
    val = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
    if val == 0:
        return 0
    elif val > 0:
        return 1
    else:
        return -1


def cross(p1, p2, q1, q2):
    """交差判定

    Args:
        p1 (float[]): 線分1の点1
        p2 (float[]): 線分1の点2
        q1 (float[]): 線分2の点1
        q2 (float[]): 線分2の点2

    Returns:
        bool: 交差しているかどうか
    """
    return (
        rotate_direction(p1, q1, p2) * rotate_direction(p1, p2, q2) >= 0
        and rotate_direction(q1, p1, q2) * rotate_direction(q1, q2, p2) >= 0
    )


from collections import deque


def convex_hull(points: list):
    """凸包

    Args:
        points (float[][]): 点のリスト、[x, y, ...]

    Returns:
        float[][]: 凸包を構成する頂点を反時計回りに返す
    """
    n = len(points)
    if n <= 2:
        return points
    elif n == 3:
        if rotate_direction(points[0], points[1], points[2]) < 0:
            return points
        else:
            return points[::-1]

    points.sort()
    top = deque([points[0][:], points[1][:]])

    for i in range(2, n):
        next = points[i][:]
        curr = top.pop()
        prev = top.pop()
        while True:
            if rotate_direction(prev, curr, next) <= 0:
                top.append(prev)
                top.append(curr)
                break
            if len(top) == 0:
                top.append(prev)
                break
            curr = prev[:]
            prev = top.pop()

        top.append(next)

    bottom = deque([points[n - 1], points[n - 2]])
    for i in reversed(range(n - 2)):
        next = points[i][:]
        curr = bottom.pop()
        prev = bottom.pop()
        while True:
            if rotate_direction(prev, curr, next) <= 0:
                bottom.append(prev)
                bottom.append(curr)
                break
            if len(bottom) == 0:
                bottom.append(prev)
                break
            curr = prev[:]
            prev = bottom.pop()

        bottom.append(next)

    top.pop()
    bottom.pop()

    ans = []
    while top:
        ans.append(top.pop())
    while bottom:
        ans.append(bottom.pop())

    return ans


# fps: https://github.com/shakayami/ACL-for-python/blob/master/fps.py
# wiki: https://github.com/shakayami/ACL-for-python/wiki/fps


# 関節点(Articulation Points)
def get_articulation_points(G, N, start=0):
    order = [None] * N
    result = []
    count = 0

    def dfs(v, prev):
        nonlocal count
        r_min = order[v] = count  # 到達時にラベル
        fcnt = 0
        p_art = 0
        count += 1
        for w in G[v]:
            if w == prev:
                continue
            if order[w] is None:
                ret = dfs(w, v)
                # 子の頂点が到達できたのが、自身のラベル以上の頂点のみ
                # => 頂点vは関節点
                p_art |= order[v] <= ret
                r_min = min(r_min, ret)
                fcnt += 1
            else:
                r_min = min(r_min, order[w])
        p_art |= r_min == order[v] and len(G[v]) > 1
        if (prev == -1 and fcnt > 1) or (prev != -1 and p_art):
            # 頂点startの場合は、二箇所以上の子頂点を調べたら自身は関節点
            result.append(v)
        return r_min

    dfs(start, -1)
    return result


# 全方位木DP
class Rerooting:
    """全方位木DP

    Arge:
        n: 頂点数
        merge ( (any left, any right) => any ): 部分木のマージ時の操作
        addNode ( (any value, int nodeId) => any ): 頂点での追加操作
        e (any): 単位元
    """

    def __init__(self, n, merge, addNode, e) -> None:
        self._n = n
        self._merge = merge
        self._addNode = addNode
        self._e = e
        self._adj = [[] for _ in range(n)]

        self._childVal = defaultdict(lambda: e)

    def add_edge(self, u, v):
        self._adj[u].append(v)
        self._adj[v].append(u)

    def reroot(self):
        parents = [-1] * self._n
        order = []
        stack = deque([0])
        ret = [self._e] * self._n

        # 行きがけ順
        while stack:
            target = stack.pop()
            order.append(target)
            for node in self._adj[target]:
                if parents[target] == node:
                    continue
                stack.append(node)
                parents[node] = target

        # 帰りがけ順
        for target in order[::-1]:
            result = self._e
            for node in self._adj[target]:
                if parents[target] == node:
                    continue
                result = self._merge(result, self._childVal[(node, target)])
            self._childVal[(target, parents[target])] = self._addNode(result, target)

        for target in order:
            accTail = [self._e]
            for node in self._adj[target][::-1]:
                accTail.append(self._merge(accTail[-1], self._childVal[(node, target)]))
            accTail = accTail[::-1]
            accHead = self._e
            for i, node in enumerate(self._adj[target]):
                result = self._addNode(self._merge(accHead, accTail[i + 1]), target)
                self._childVal[(target, node)] = result
                accHead = self._merge(accHead, self._childVal[(node, target)])

            ret[target] = self._addNode(accHead, target)

        return ret

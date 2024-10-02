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
            int: 木の直径
        """
        d = self.search(0)
        u = 0
        max_cost = -1
        for i in range(self.n):
            if max_cost < d[i]:
                u = i
                max_cost = d[i]

        du = self.search(u)
        v = 0
        max_cost = 0
        for i in range(self.n):
            if max_cost < du[i]:
                v = i
                max_cost = du[i]

        return max_cost

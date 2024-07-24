from collections import defaultdict


class LowLink:
    def __init__(self, adj):
        """初期化

        Args:
            adj (list<list<int>>): 隣接リスト
        """
        self.n = len(adj)
        self.d = [defaultdict(bool) for _ in range(self.n)]
        for u in range(self.n):
            for v in adj[u]:
                self.d[u][v] = True
        self.ord = [None] * self.n
        self.low = [None] * self.n
        self.son = [[] for _ in range(self.n)]
        self.back_edge = [[] for _ in range(self.n)]
        self.tps = []

        t = 0
        stack = [(None, 0)]
        while stack:
            pre, now = stack.pop()
            if self.ord[now] is not None:
                if self.ord[pre] < self.ord[now]:
                    continue
                self.low[pre] = min(self.low[pre], self.ord[now])
                self.back_edge[pre].append(now)
                continue
            if pre is not None:
                self.son[pre].append(now)
            self.tps.append(now)
            self.ord[now] = t
            self.low[now] = self.ord[now]
            t += 1
            for next in adj[now]:
                if next == pre:
                    continue
                stack.append((now, next))

        for u in reversed(self.tps[1:]):
            for v in self.son[u]:
                self.low[u] = min(self.low[u], self.low[v])

    def is_bridge(self, u, v):
        """橋判定

        Args:
            u (int): 頂点1
            v (int): 頂点2

        Returns:
            bool: 橋なら True, そうでない or 辺が存在しないなら False
        """
        if not self.d[u][v]:
            return False
        if self.ord[u] > self.ord[v]:
            u, v = v, u
        return self.ord[u] < self.low[v]

    def is_articulation(self, u):
        """関節点判定

        Args:
            u (int): 判定する頂点

        Returns:
            bool: 関節点なら True, そうでなければ False
        """
        if u == 0:
            return len(self.son[u]) > 1
        for v in self.son[u]:
            if self.ord[u] <= self.low[v]:
                return True
        return False

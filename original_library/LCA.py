from collections import deque


class LCA:
    def __init__(self, n, adj, root=0, max_bit=60) -> None:
        """初期化

        Args:
            n (int): 頂点数
            adj (list<int>): 隣接リスト
            root (int, optional): 木の root. Defaults to 0.
            max_bit (int, optional): 頂点数の最大ビット. Defaults to 60.
        """
        self.n = n
        self.max_bit = max_bit
        self.adj = adj
        self.parent = [[-1] * self.n for _ in range(self.max_bit)]
        self.dist = [-1] * self.n
        self.__dfs(root)
        for i in range(1, self.max_bit):
            for v in range(n):
                if self.parent[i - 1][v] < 0:
                    self.parent[i][v] = -1
                else:
                    self.parent[i][v] = self.parent[i - 1][self.parent[i - 1][v]]

    def __dfs(self, root):
        self.parent[0][root] = root
        self.dist[root] = 0
        q = deque([root])
        while q:
            v = q.pop()
            for u in self.adj[v]:
                if self.parent[0][u] != -1:
                    continue
                self.parent[0][u] = v
                self.dist[u] = self.dist[v] + 1
                q.append(u)

    def query(self, u, v):
        """u, v の最小共通祖先(LCA)を返す. O(log N)

        Args:
            u (int): 頂点1
            v (int): 頂点2

        Returns:
            int: 最小共通祖先(LCA)
        """
        if self.dist[u] < self.dist[v]:
            u, v = v, u
        for i in range(self.max_bit):
            if (self.dist[u] - self.dist[v]) >> i & 1:
                u = self.parent[i][u]
        if u == v:
            return u
        for i in range(self.max_bit)[::-1]:
            if self.parent[i][u] != self.parent[i][v]:
                u = self.parent[i][u]
                v = self.parent[i][v]
        return self.parent[0][u]

    def depth(self, v):
        """頂点 v の root からの距離. O(1)

        Args:
            v (int): 頂点

        Returns:
            int: root からの距離
        """
        return self.dist[v]

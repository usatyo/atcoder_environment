from collections import deque


# !!! 未テスト !!!


class TreeDP:
    def __init__(self, n, merge, addNode, l) -> None:
        """初期化

        Args:
            n (int): 頂点数
            merge ((parent, child) -> any): 部分木のひとつを親に反映する処理
            addNode ((value, nodeId) -> any): すべてのマージが終了した後の処理
            l (list<any>): 初期値のリスト
        """
        self.n = n
        self.adj = [[] for _ in range(n)]
        self.merge = merge
        self.addNode = addNode
        self.dp = l

    def add_edge(self, u, v):
        """辺を追加する

        Args:
            u (int): 頂点1
            v (int): 頂点2
        """
        self.adj[u].append(v)
        self.adj[v].append(u)

    def calc(self, root):
        """木DPを実行する

        Args:
            root (int): 根とする頂点

        Returns:
            any: 木DPの結果
        """
        stack = deque([root])
        visited = [False] * self.n
        order = []
        parent = [-1] * self.n
        while stack:
            u = stack.pop()
            visited[u] = True
            order.append(u)
            for v in self.adj[u]:
                if not visited[v]:
                    parent[v] = u
                    stack.append(v)

        for u in order[::-1]:
            for v in self.adj[u]:
                if v == parent[u]:
                    continue
                self.dp[u] = self.merge(self.dp[u], self.addNode(self.dp[v], v))

        return self.addNode(self.dp[root], root)

from collections import deque


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

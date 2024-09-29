class WarshallFloyd:
    """ワーシャルフロイド法. O(V^3)"""

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

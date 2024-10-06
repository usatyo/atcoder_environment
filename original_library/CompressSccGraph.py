from atcoder.scc import SCCGraph


class CompressSccGraph:
    def __init__(self, n) -> None:
        """初期化

        Args:
            n (int): 頂点数
        """
        self.n_before = n
        self.graph = SCCGraph(n)
        self.adj_before = [[] for _ in range(n)]
        self.label = [-1] * n
        self.compressed = False

    def add_edge(self, from_vertex, to_vertex):
        """辺を追加

        Args:
            from_vertex (int): 始点
            to_vertex (int): 終点
        """
        assert 0 <= from_vertex < self.n_before
        assert 0 <= to_vertex < self.n_before
        assert self.compressed == False, "use add_edge() before compress()"
        self.graph.add_edge(from_vertex, to_vertex)
        self.adj_before[from_vertex].append(to_vertex)

    def compress(self):
        """強連結成分ごとに圧縮
        """
        self.compressed = True
        self.res = self.graph.scc()
        self.n_after = len(self.res)
        for i in range(self.n_after):
            for x in self.res[i]:
                self.label[x] = i

        self.adj_after = [set() for _ in range(self.n_after)]
        for v in range(self.n_before):
            for u in self.adj_before[v]:
                if self.label[v] != self.label[u]:
                    self.adj_after[self.label[v]].add(self.label[u])

    def size(self):
        """圧縮後の頂点数を返す

        Returns:
            int: 圧縮後の頂点数
        """
        assert self.compressed == True, "use size() after compress()"
        return self.n_after

    def adj(self, v):
        """圧縮後の頂点vの隣接リストを返す

        Args:
            v (int): 圧縮後の頂点

        Returns:
            list: vの隣接リスト
        """
        assert self.compressed == True, "use adj() after compress()"
        assert 0 <= v < self.n_after
        return list(self.adj_after[v])

    def forward(self, v):
        """圧縮前の頂点を圧縮後の頂点に変換

        Args:
            v (int): 圧縮前の頂点

        Returns:
            int: 圧縮後の頂点
        """
        assert self.compressed == True, "use forward() after compress()"
        assert 0 <= v < self.n_before
        return self.label[v]

    def backward(self, v):
        """圧縮後の頂点を圧縮前の頂点のリストに変換

        Args:
            v (int): 圧縮後の頂点

        Returns:
            list: 圧縮前の頂点のリスト
        """
        assert self.compressed == True, "use backward() after compress()"
        assert 0 <= v < self.n_after
        return self.res[v]

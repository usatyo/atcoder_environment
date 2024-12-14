from sys import setrecursionlimit

setrecursionlimit(10**7)


class WeightedUnionFind:
    def __init__(self, N):
        """初期化

        Args:
            N (int): 頂点数
        """
        self.N = N
        self.parents = [-1] * N
        self.rank = [0] * N
        self.weight = [0] * N

    def root(self, x):
        if self.parents[x] == -1:
            return x
        rx = self.root(self.parents[x])
        self.weight[x] += self.weight[self.parents[x]]
        self.parents[x] = rx
        return self.parents[x]

    def get_weight(self, x):
        """頂点 x における重みを取得

        Args:
            x (int): 対象の頂点

        Returns:
            int: 頂点 x の重み
        """
        self.root(x)
        return self.weight[x]

    def merge(self, x, y, d):
        """2つの連結成分を結合

        Args:
            x (int): 頂点1
            y (int): 頂点2
            d (int): 重み

        Returns:
            bool: 矛盾発生時に False. それ以外で True
        """
        # A[x] - A[y] = d

        w = d + self.get_weight(x) - self.get_weight(y)
        rx = self.root(x)
        ry = self.root(y)
        if rx == ry:
            _, d_xy = self.diff(x, y)
            if d_xy == d:
                return True
            else:
                return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
            w = -w
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

        self.parents[ry] = rx
        self.weight[ry] = w
        return True

    def same(self, x, y):
        """

        Args:
            x (int): 頂点1
            y (int): 頂点2

        Returns:
            bool: x と y が同一の連結成分に属していれば True, そうでなければ False
        """
        return self.root(x) == self.root(y)

    def diff(self, x, y):
        """x, y 間の差分を取得

        Args:
            x (int): 頂点1
            y (int): 頂点2

        Returns:
            bool, int: 同一の連結成分に属しているか, 差分の値
        """
        if self.same(x, y):
            return True, self.get_weight(y) - self.get_weight(x)
        else:
            return False, 0

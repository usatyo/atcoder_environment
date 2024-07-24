class MultipleLinearEquation:
    """整数係数の連立方程式(mod p), O(N**3 * log p)"""

    def __init__(self, col, mod=998244353):
        """初期化

        Args:
            col (int): 変数の個数
            mod (int, optional): 法となる素数. Defaults to 998244353.
        """
        self.mod = mod
        self.a = []
        self.b = []
        self.row = 0
        self.col = col

    def __hakidashi(self):
        rank = 0
        mat = [self.a[i] + [self.b[i]] for i in range(self.row)]
        for i in range(self.col):
            pivot = -1
            for j in range(rank, self.row):
                if mat[j][i] != 0:
                    pivot = j
                    break
            else:
                continue

            mat[rank], mat[pivot] = mat[pivot][::], mat[rank][::]
            inv = pow(mat[rank][i], -1, self.mod)
            mat[rank] = [x * inv % self.mod for x in mat[rank]]

            for j in range(self.row):
                if j == rank or mat[j][i] == 0:
                    continue
                mat[j] = [
                    (x - y * mat[j][i]) % self.mod for x, y in zip(mat[j], mat[rank])
                ]
            rank += 1
        return rank, mat

    def add(self, a, b):
        """式を追加

        Args:
            a (list<int>): 係数のリスト
            b (int): 右辺の値
        """
        assert type(a) == list, "invalid type of a"
        assert type(b) == int, "invalid type of b"
        assert len(a) == self.col, "invalid length of a"
        self.a.append(a)
        self.b.append(b)
        self.row += 1

    def solve(self):
        """方程式を解く

        Returns:
            list: 解のひとつを返す. 解が存在しない場合は空のリストを返す
        """
        rank, mat = self.__hakidashi()
        for i in range(rank, self.row):
            if mat[i][-1] != 0:
                return []

        ret = []
        col = 0
        for i in range(rank):
            while col < self.col and mat[i][col] == 0:
                col += 1
                ret.append(0)
            ret.append(mat[i][-1])
            col += 1

        while len(ret) < self.col:
            ret.append(0)

        return ret

    def rank(self):
        """係数行列の rank

        Returns:
            int: 係数行列の rank を返す
        """
        rank, _ = self.__hakidashi()
        return rank

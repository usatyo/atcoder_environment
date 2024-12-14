from atcoder.dsu import DSU


class MonoidUnionFind(DSU):
    def __init__(self, n, init_arr, op):
        """初期化

        Args:
            n (int): サイズ
            init_arr (list): 初期配列
            op (function): 二項演算
        """
        super().__init__(n)
        self._arr = init_arr[::]
        self._op = op

    def merge(self, a, b):
        val = self._op(self._arr[self.leader(a)], self._arr[self.leader(b)])
        super().merge(a, b)
        self._arr[self.leader(a)] = val

    def update(self, a, val):
        self._arr[self.leader(a)] = val

    def get(self, a):
        return self._arr[self.leader(a)]

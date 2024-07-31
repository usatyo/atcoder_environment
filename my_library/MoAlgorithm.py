from operator import itemgetter


class Mo:
    def __init__(self, n, q, ls):
        self.ls = ls.copy()
        self.n = n
        self.block_range = int(n / q**0.5) + 1

    def _init_states(self):
        # TODO: 自分で定義
        self.states = [0] * self.n
        self.ans = 0

        # [l, r) の半開区間で考える
        self.l = 0
        self.r = 0

        # queryを格納する用
        self.bucket = [[] for _ in range((self.n // self.block_range + 1))]

    def _add(self, i):
        # TODO: i番目を追加する処理, self.status に応じて self.ansを更新する
        pass

    def _delete(self, i):
        # TODO: i番目を削除する処理, self.status に応じて self.ansを更新する
        pass

    def _one_process(self, l, r):
        for i in range(self.r, r):
            self._add(i)
        for i in range(self.r - 1, r - 1, -1):
            self._delete(i)
        for i in range(self.l, l):
            self._delete(i)
        for i in range(self.l - 1, l - 1, -1):
            self._add(i)

        self.l = l
        self.r = r

    def solve(self, queries):
        self._init_states()

        for i, (l, r) in enumerate(queries):
            self.bucket[l // self.block_range].append((r, l, i))

        ret = [-1] * len(queries)
        for idx, b in enumerate(self.bucket):
            for r, l, i in sorted(b, reverse=idx % 2, key=itemgetter(0)):
                self._one_process(l, r)
                ret[i] = self.ans

        return ret

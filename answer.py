from collections import defaultdict
from operator import itemgetter
import sys

input = sys.stdin.readline


class Mo:
    def __init__(self, n, q, ls):
        self.ls = ls.copy()
        self.n = n
        self.block_range = int(n / q**0.5) + 1
        # self.b = ceil(sqrt(self.n))  # bukectのサイズ及び個数

    def _init_states(self):
        # TODO: 自分で定義
        self.states = [0] * n
        self.ans = 0

        # [l,r)の半開区間で考える
        self.l = 0
        self.r = 0

        # queryを格納する用
        self.bucket = [[] for _ in range((self.n // self.block_range + 1))]

    def _add(self, i):
        # TODO: i番目を追加する処理
        self.states[self.ls[i]] += 1
        if not self.states[self.ls[i]] & 1:
            self.ans += 1

    def _delete(self, i):
        # TODO: i番目を削除する処理
        self.states[self.ls[i]] -= 1
        if self.states[self.ls[i]] & 1:
            self.ans -= 1

    def _one_process(self, l, r):
        # クエリ[l,r)に対してstatesを更新する
        for i in range(self.r, r):  # rまで伸長
            self._add(i)
        for i in range(self.r - 1, r - 1, -1):  # rまで短縮
            self._delete(i)
        for i in range(self.l, l):  # lまで短縮
            self._delete(i)
        for i in range(self.l - 1, l - 1, -1):  # lまで伸長
            self._add(i)

        self.l = l
        self.r = r

    def solve(self, queries):
        self._init_states()

        for i, (l, r) in enumerate(queries):  # queryをbucketに格納
            self.bucket[l // self.block_range].append((r, l, i))

        ret = [-1] * len(queries)
        for idx, b in enumerate(self.bucket):
            for r, l, i in sorted(
                b, reverse=idx % 2, key=itemgetter(0)
            ):  # クエリに答えていく
                self._one_process(l, r)

                # TODO: クエリに答える処理
                ret[i] = self.ans

        return ret


n = int(input())
a = list(map(lambda x: int(x) - 1, input().split()))
q = int(input())
queries = []

for i in range(q):
    l, r = map(int, input().split())
    l -= 1
    queries.append([l, r])

mo = Mo(n, q, a)
ans = mo.solve(queries)

print(*ans, sep="\n")

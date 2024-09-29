"""Heavy-Light-Decomposition

重み付きの無向辺から構成される木を、HL分解します。

中のセグメントツリーは、shakayamiさん作のものを使っています。
参考URL【https://github.com/shakayami/ACL-for-python/blob/master/segtree.py】
つかいかたURL【https://github.com/shakayami/ACL-for-python/wiki/segtree】

・つかいかた
(以下のノード番号、辺番号はすべて0-indexedです。)
(node1, node2, weight) が格納されている edge_list を用意します。
op, e: セグメントツリーにのせる関数と、その単位元です。詳しいことは上のつかいかたURLにあります。
hld = HLD(edge_list, op, e) : インスタンス生成
hld.weight_set(edge_number, new_weight): 辺 edge_number の重みを new_weight に変更します。
hld.solve(node1, node2): 2node間の計算をします(関数opを使います)。

・計算量
インスタンス作成: O(N)
辺の重み変更: O(1)
2ノード間の計算: O((logN)^2)
⇒ クエリ個数がQのとき、トータルでO(N+Q(logN)^2)
"""

from collections import deque


class HLD:
    def __init__(self, edge_list, op, e):
        self.edge_list = edge_list
        self.op = op
        self.e = e
        self.__build()

    def __build(self):
        self.N = len(self.edge_list) + 1  # 全ノードの数
        self.Graph = [[] for _ in range(self.N)]  # 無向グラフを構築
        for n1, n2, weight in self.edge_list:
            self.Graph[n1].append([n2, weight])
            self.Graph[n2].append([n1, weight])

        self.siz = [1] * self.N  # 部分木のサイズ
        self.parent = [-1] * self.N  # 親ノード番号
        self.to_parent_weight = [-1] * self.N  # 親ノードを結ぶ辺の重み
        dfs_stack = [(-1, 0)]
        back_stack = []
        while dfs_stack:
            par, pos = dfs_stack.pop()
            self.parent[pos] = par
            for npos, weight in self.Graph[pos]:
                if npos == par:
                    continue
                dfs_stack.append((pos, npos))
                back_stack.append((pos, npos))
                self.to_parent_weight[npos] = weight

        heaviest_child = [(-1, 0)] * self.N  # (ノード番号, 部分木サイズ)を格納
        while back_stack:
            par, pos = back_stack.pop()
            self.siz[par] += self.siz[pos]
            if heaviest_child[par][1] < self.siz[pos]:
                heaviest_child[par] = (pos, self.siz[pos])

        # self.top_dist[v]: ノードvの属するheavy-pathのtopと、そこまでの最短パス(通る辺の数)
        self.top_dist = [(-1, -1)] * self.N
        self.top_dist[0] = (0, 0)
        que = deque()
        que.append((0, -1, 0, 0))  # (pos, par, top, dist)を格納
        self.heavy_depth = [0] * self.N  # light-edge を通った回数
        weight_list_dict = dict()
        weight_list_dict[0] = []
        while que:
            pos, par, top, dist = que.popleft()
            heaviest_node = heaviest_child[pos][0]
            for npos, weight in self.Graph[pos]:
                if npos == par:
                    continue
                # おなじheavy-path
                if npos == heaviest_node:
                    que.append((npos, pos, top, dist + 1))
                    self.heavy_depth[npos] = self.heavy_depth[pos]
                    weight_list_dict[top].append(weight)
                    self.top_dist[npos] = (top, dist + 1)
                # light-edgeを通り、新しいheavy-pathを生成
                else:
                    que.append((npos, pos, npos, 0))
                    self.heavy_depth[npos] = self.heavy_depth[pos] + 1
                    weight_list_dict[npos] = []
                    self.top_dist[npos] = (npos, 0)
        self.weight_st_dict = dict()
        for top, weight_list in weight_list_dict.items():
            self.weight_st_dict[top] = segtree(weight_list_dict[top], self.op, self.e)

    def weight_set(self, edge_number, new_weight):
        a, b, old_weight = self.edge_list[edge_number]
        if self.parent[a] == b:
            a, b = b, a
        self.to_parent_weight[b] = new_weight
        b_top, b_dist = self.top_dist[b]
        if b_dist > 0:
            self.weight_st_dict[b_top].set(b_dist - 1, new_weight)

    def solve(self, n1, n2):
        hd1 = self.heavy_depth[n1]
        top1, dist1 = self.top_dist[n1]
        hd2 = self.heavy_depth[n2]
        top2, dist2 = self.top_dist[n2]
        ans = self.e
        while True:
            if top1 == top2:
                if dist1 < dist2:
                    ans = self.op(ans, self.weight_st_dict[top1].prod(dist1, dist2))
                elif dist2 < dist1:
                    ans = self.op(ans, self.weight_st_dict[top1].prod(dist2, dist1))
                break
            if hd1 < hd2:
                ans = self.op(ans, self.weight_st_dict[top2].prod(0, dist2))
                ans = self.op(ans, self.to_parent_weight[top2])
                n2 = self.parent[top2]
                top2, dist2 = self.top_dist[n2]
                hd2 -= 1
            else:
                ans = self.op(ans, self.weight_st_dict[top1].prod(0, dist1))
                ans = self.op(ans, self.to_parent_weight[top1])
                n1 = self.parent[top1]
                top1, dist1 = self.top_dist[n1]
                hd1 -= 1
        return ans


# shakayamiさん作のセグメントツリー
class segtree:
    n = 1
    size = 1
    log = 2
    d = [0]
    op = None
    e = 10**15

    def __init__(self, V, OP, E):
        self.n = len(V)
        self.op = OP
        self.e = E
        self.log = (self.n - 1).bit_length()
        self.size = 1 << self.log
        self.d = [E for i in range(2 * self.size)]
        for i in range(self.n):
            self.d[self.size + i] = V[i]
        for i in range(self.size - 1, 0, -1):
            self.update(i)

    def set(self, p, x):
        assert 0 <= p and p < self.n
        p += self.size
        self.d[p] = x
        for i in range(1, self.log + 1):
            self.update(p >> i)

    def get(self, p):
        assert 0 <= p and p < self.n
        return self.d[p + self.size]

    def prod(self, l, r):
        assert 0 <= l and l <= r and r <= self.n
        sml = self.e
        smr = self.e
        l += self.size
        r += self.size
        while l < r:
            if l & 1:
                sml = self.op(sml, self.d[l])
                l += 1
            if r & 1:
                smr = self.op(self.d[r - 1], smr)
                r -= 1
            l >>= 1
            r >>= 1
        return self.op(sml, smr)

    def all_prod(self):
        return self.d[1]

    def max_right(self, l, f):
        assert 0 <= l and l <= self.n
        assert f(self.e)
        if l == self.n:
            return self.n
        l += self.size
        sm = self.e
        while 1:
            while l % 2 == 0:
                l >>= 1
            if not (f(self.op(sm, self.d[l]))):
                while l < self.size:
                    l = 2 * l
                    if f(self.op(sm, self.d[l])):
                        sm = self.op(sm, self.d[l])
                        l += 1
                return l - self.size
            sm = self.op(sm, self.d[l])
            l += 1
            if (l & -l) == l:
                break
        return self.n

    def min_left(self, r, f):
        assert 0 <= r and r < self.n
        assert f(self.e)
        if r == 0:
            return 0
        r += self.size
        sm = self.e
        while 1:
            r -= 1
            while r > 1 & (r % 2):
                r >>= 1
            if not (f(self.op(self.d[r], sm))):
                while r < self.size:
                    r = 2 * r + 1
                    if f(self.op(self.d[r], sm)):
                        sm = self.op(self.d[r], sm)
                        r -= 1
                return r + 1 - self.size
            sm = self.op(self.d[r], sm)
            if (r & -r) == r:
                break
        return 0

    def update(self, k):
        self.d[k] = self.op(self.d[2 * k], self.d[2 * k + 1])

    def __str__(self):
        return str([self.get(i) for i in range(self.n)])

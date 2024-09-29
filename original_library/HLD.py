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
from atcoder.segtree import SegTree


class HLD:
    def __init__(self, edge_list, op, e):
        """初期化

        Args:
            edge_list (list<tuple>): [(node1, node2, weight), ...]
            op (func): セグ木にのせる操作
            e (any): セグ木にのせる単位元
        """
        self.edge_list = edge_list
        self.op = op
        self.e = e
        self.__build()

    def __build(self):
        self.N = len(self.edge_list) + 1  # 全ノードの数
        self.Graph = [[] for _ in range(self.N)]  # 無向グラフを構築
        for u, v, weight in self.edge_list:
            self.Graph[u].append([v, weight])
            self.Graph[v].append([u, weight])

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
            self.weight_st_dict[top] = SegTree(self.op, self.e, weight_list_dict[top])

    def weight_set(self, edge_number, new_weight):
        """重みを更新

        Args:
            edge_number (int): 初期化時の辺番号
            new_weight (int): 更新後の重み
        """
        a, b, old_weight = self.edge_list[edge_number]
        if self.parent[a] == b:
            a, b = b, a
        self.to_parent_weight[b] = new_weight
        b_top, b_dist = self.top_dist[b]
        if b_dist > 0:
            self.weight_st_dict[b_top].set(b_dist - 1, new_weight)

    def solve(self, u, v):
        """u, v 間のパス上の演算結果

        Args:
            u (int): 頂点1
            v (int): 頂点2

        Returns:
            any: パス上の演算結果
        """
        hd1 = self.heavy_depth[u]
        top1, dist1 = self.top_dist[u]
        hd2 = self.heavy_depth[v]
        top2, dist2 = self.top_dist[v]
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
                v = self.parent[top2]
                top2, dist2 = self.top_dist[v]
                hd2 -= 1
            else:
                ans = self.op(ans, self.weight_st_dict[top1].prod(0, dist1))
                ans = self.op(ans, self.to_parent_weight[top1])
                u = self.parent[top1]
                top1, dist1 = self.top_dist[u]
                hd1 -= 1
        return ans

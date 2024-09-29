from collections import defaultdict, deque


class Rerooting:
    """全方位木DP

    Arge:
        n: 頂点数
        merge ( (any left, any right) => any ): 部分木のマージ時の操作
        addNode ( (any value, int nodeId) => any ): 頂点での追加操作
        e (any): 単位元
    """

    def __init__(self, n, merge, addNode, e) -> None:
        self._n = n
        self._merge = merge
        self._addNode = addNode
        self._e = e
        self._adj = [[] for _ in range(n)]

        self._childVal = defaultdict(lambda: e)

    def add_edge(self, u, v):
        """u, v 間に無向辺を追加

        Args:
            u (int): 頂点1
            v (int): 頂点2
        """
        self._adj[u].append(v)
        self._adj[v].append(u)

    def reroot(self):
        """木上での演算を実行

        Returns:
            list<any>: 各頂点を根とした木DPの結果
        """
        parents = [-1] * self._n
        order = []
        stack = deque([0])
        ret = [self._e] * self._n

        # 行きがけ順
        while stack:
            target = stack.pop()
            order.append(target)
            for node in self._adj[target]:
                if parents[target] == node:
                    continue
                stack.append(node)
                parents[node] = target

        # 帰りがけ順
        for target in order[::-1]:
            result = self._e
            for node in self._adj[target]:
                if parents[target] == node:
                    continue
                result = self._merge(result, self._childVal[(node, target)])
            self._childVal[(target, parents[target])] = self._addNode(result, target)

        for target in order:
            accTail = [self._e]
            for node in self._adj[target][::-1]:
                accTail.append(self._merge(accTail[-1], self._childVal[(node, target)]))
            accTail = accTail[::-1]
            accHead = self._e
            for i, node in enumerate(self._adj[target]):
                result = self._addNode(self._merge(accHead, accTail[i + 1]), target)
                self._childVal[(target, node)] = result
                accHead = self._merge(accHead, self._childVal[(node, target)])

            ret[target] = self._addNode(accHead, target)

        return ret

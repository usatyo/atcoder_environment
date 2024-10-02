class Doubling:
    def __init__(self, n, next, max_bit=60) -> None:
        """初期化

        Args:
            n (int): 状態数
            next (list<int>): 遷移関数
            max_bit (int, optional): 遷移回数の最大ビット. Defaults to 60.
        """
        self.n = n
        self.next = [next[::]]
        self.max_bit = max_bit
        self.__calc_next()

    def __calc_next(self):
        for _ in range(self.max_bit):
            next = []
            for i in range(self.n):
                next.append(self.next[-1][self.next[-1][i]])

            self.next.append(next[::])

    def pow(self, start, k):
        """start の状態から k 回遷移後の状態を返す. O(log k)

        Args:
            start (int): 初期状態
            k (int): 遷移回数

        Returns:
            int: k 回遷移後の状態
        """
        ans = start
        for i in range(self.max_bit):
            if k >> i & 1:
                ans = self.next[i][ans]

        return ans

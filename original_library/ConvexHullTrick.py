from collections import deque


class CHT:
    def __init__(self) -> None:
        self.deq = deque()

    def __check(self, f1, f2, f3):
        return (f2[0] - f1[0]) * (f3[1] - f2[1]) >= (f2[1] - f1[1]) * (f3[0] - f2[0])

    def __f(self, f1, x):
        return f1[0] * x + f1[1]

    def add_line(self, a, b):
        """直線 f_i(x) = ax + bを追加（傾きの大きい方から追加）

        Args:
            a (int): 傾き
            b (int): y切片
        """
        f1 = (a, b)
        while len(self.deq) >= 2 and self.__check(self.deq[-2], self.deq[-1], f1):
            self.deq.pop()
        self.deq.append(f1)

    def query_min(self, x):
        """追加された直線のxにおける最小値（xの大きい方から入れる）

        Args:
            x (int): x座標

        Returns:
            int: min f_i(x)
        """
        while len(self.deq) >= 2 and self.__f(self.deq[0], x) >= self.__f(
            self.deq[1], x
        ):
            self.deq.popleft()
        return self.__f(self.deq[0], x)

    def query_max(self, x):
        """追加された直線のxにおける最大値（xの大きい方から入れる）

        Args:
            x (int): x座標

        Returns:
            int: min f_i(x)
        """
        while len(self.deq) >= 2 and self.__f(self.deq[0], x) <= self.__f(
            self.deq[1], x
        ):
            self.deq.popleft()
        return self.__f(self.deq[0], x)

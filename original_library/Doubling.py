class Doubling:
    def __init__(self, n, next, max_bit=60) -> None:
        self.n = n
        self.next = [next[::]]
        self.max_bit = max_bit
        self.calc_next()

    def calc_next(self):
        for _ in range(self.max_bit):
            next = []
            for i in range(self.n):
                next.append(self.next[-1][self.next[-1][i]])

            self.next.append(next[::])

    def solve(self, start, k):
        ans = start
        for i in range(self.max_bit):
            if k >> i & 1:
                ans = self.next[i][ans]

        return ans

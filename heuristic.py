from collections import defaultdict
from math import exp
from random import randint, random
from sys import stdin
import os


class Solver:
    DX = [0, 1, 0, -1]
    DY = [1, 0, -1, 0]
    DIR = ["R", "D", "L", "U"]
    ROT = ["L", "R", "."]
    DIST = [16, 8, 4, 2, 1]

    def __init__(self, input) -> None:
        self._recieve_input(input)
        self._init_variable()

    def _recieve_input(self, input):
        self.N, self.M, self.V = map(int, input().split())
        self.S = [[int(x) for x in input()] for _ in range(self.N)]
        self.T = [[int(x) for x in input()] for _ in range(self.N)]

    def _generate_input(self):
        self.N = randint(15, 30)
        self.M = randint(self.N**2 // 10 + 1, self.N**2 // 2)
        self.V = randint(5, 15)

        while True:
            self.S = self._generate_random_board()
            self.T = self._generate_random_board()
            diff = 0
            for i in range(self.N):
                for j in range(self.N):
                    if self.S[i][j] != self.T[i][j]:
                        diff += 1
            if diff >= self.M:
                break

    def _generate_random_board(self):
        w = [[0] * self.N for _ in range(self.N)]
        sm, ct = 0, 0
        ret = [[0] * self.N for _ in range(self.N)]
        for _ in range(randint(1, 5)):
            cx, cy = random() * (self.N + 1) - 1, random() * (self.N + 1) - 1
            alpha = random()
            sigma = random() * 3 + 2
            for i in range(self.N):
                for j in range(self.N):
                    dif = alpha * exp(-((i - cx) ** 2 + (j - cy) ** 2) / (2 * sigma**2))
                    w[i][j] += dif
                    sm += dif

        v = random() * sm
        for _ in range(self.M):
            for i in range(self.N):
                for j in range(self.N):
                    v -= w[i][j]
                    if v < 0:
                        ret[i][j] = 1
                        sm -= w[i][j]
                        w[i][j] = 0
                        ct += 1
                        v = random() * sm
            if ct == self.M:
                return ret

    def _init_variable(self):
        self.start = [0, 0]
        self.machine: list[tuple] = []
        self.hand = [[-1, -1] for _ in range(self.V)]
        self.operation: list[list[int]] = []
        self.have = [False] * self.V
        self.match = 0
        for i in range(self.N):
            for j in range(self.N):
                if self.S[i][j] == self.T[i][j] == 1:
                    self.match += 1
                    self.S[i][j] = 0
                    self.T[i][j] = 0

        if self.N <= 16:
            self.DIST = [8, 4, 2, 1]

        self.pos_to_op = defaultdict(list)
        for bit in range(4**5):
            op = []
            x, y = 0, 0
            rot = 0
            for i in range(len(self.DIST)):
                r = bit % 4 - 1
                op.append(r)
                rot = (rot - r) % 4
                dist = self.DIST[i]
                x += dist * self.DX[rot]
                y += dist * self.DY[rot]

                bit //= 4

            self.pos_to_op[(x, y)] = op
        self.small_board = False
        if self.N <= 16:
            self.small_board = True

    def solve(self):
        self.construct_machine()
        self.operate_machine()
        self.output()

    def construct_machine(self):
        if self.V == 5:
            self._v_five_construct()
        else:
            self._log_tree_construct()

    def operate_machine(self):
        if self.V == 5:
            self._v_five_operate()
        else:
            self._log_tree_operate()

    def output(self):
        assert 1 <= len(self.machine) <= self.V
        print(len(self.machine))

        for i, (p, l) in enumerate(self.machine[1:]):
            assert 0 <= p <= i
            assert 1 <= l < self.N
            print(p, l)

        assert 0 <= self.start[0] < self.N, 0 <= self.start[1] < self.N
        print(*self.start)

        for op in self.operation:
            assert len(op) == len(self.machine) * 2
            print("".join(op))

    def move(self, idx, dif):
        return (
            self.hand[idx][0] + dif[0],
            self.hand[idx][1] + dif[1],
        )

    def rotate(self, idx, par, rot):
        x, y = self.hand[idx]
        px, py = self.hand[par]
        if rot == 0:
            return [px - (y - py), py + (x - px)]
        elif rot == 1:
            return [px + (y - py), py - (x - px)]
        else:
            return [x, y]

    def _simple_construct(self):
        self.start = [self.N // 2, self.N // 2]
        self.hand[0] = self.start[::]
        self.machine.append((0, 0))
        for i in range(self.V - 1):
            self.machine.append((i, 1))
            self.hand[i + 1] = [self.hand[i][0], self.hand[i][1] + 1]

    def _simple_operate(self):
        for _ in range(10**5):
            op = []
            while True:
                dir = randint(0, 3)
                x, y = self.move(0, [self.DX[dir], self.DY[dir]])
                if 0 <= x < self.N and 0 <= y < self.N:
                    self.hand[0] = [x, y]
                    break
            op.append(self.DIR[dir])
            dif = [self.DX[dir], self.DY[dir]]
            for i in range(1, self.V):
                self.hand[i] = self.move(i, dif)
            for i in range(1, self.V):
                x, y = self.hand[i]
                rot = randint(0, 2)
                op.append(self.ROT[rot])
                for j in range(i, self.V):
                    self.hand[j] = list(self.rotate(j, i - 1, rot))

            op = op + ["."] * self.V
            lx, ly = self.hand[self.V - 1]
            self._check_leaf(lx, ly, op, self.V * 2 - 1)

            self.operation.append(op[::])
            if self.match == self.M:
                return

    def _check_leaf(self, x, y, op, i):
        if not 0 <= x < self.N or not 0 <= y < self.N:
            return
        elif self.have[self.V - 1] and self.S[x][y] == 1:
            return
        elif not self.have[self.V - 1] and self.S[x][y] == 0:
            return
        elif self.S[x][y] == self.T[x][y]:
            return
        else:
            op[i] = "P"
            self.S[x][y] = 1 - self.S[x][y]
            if self.S[x][y] == 1:
                self.match += 1
            self.have[self.V - 1] = not self.have[self.V - 1]

    def _log_tree_construct(self):
        assert self.V >= 6

        self.start = [self.N // 2, self.N // 2]
        self.hand[0] = self.start[::]
        self.machine.append((0, 0))
        if not self.small_board:
            self.machine.append((0, 16))
            self.machine.append((1, 8))
            self.machine.append((2, 4))
            self.machine.append((3, 2))
            self.machine.append((4, 1))
        else:
            self.machine.append((0, 8))
            self.machine.append((1, 4))
            self.machine.append((2, 2))
            self.machine.append((3, 1))

    def _log_tree_operate(self):
        assert self.V >= 6

        machine = len(self.machine)
        s = []
        t = []
        cur = [0] * (machine - 1)

        for i in range(self.N):
            for j in range(self.N):
                if self.S[i][j] == 1:
                    s.append((i, j))
                if self.T[i][j] == 1:
                    t.append((i, j))

        path = []
        while s and t:
            path.append(s.pop())
            path.append(t.pop())

        for i in range(len(path)):
            x, y = path[i]
            odd = False
            if not (x + y + self.hand[0][0] + self.hand[0][1]) & 1:
                odd = True
                self.hand[0][1] += 1

            op = self.pos_to_op[(x - self.hand[0][0], y - self.hand[0][1])][::]
            # print(x, y, self.hand[0], op, cur)
            for _ in range(2):
                self.operation.append(["."])
                for j in range(machine - 1):
                    dif = (op[j] - cur[j]) % 4
                    # print(f"{j=} {dif=}")
                    if dif == 1:
                        self.operation[-1].append("L")
                        cur[j] += 1
                    elif dif == 0:
                        self.operation[-1].append(".")
                    else:
                        self.operation[-1].append("R")
                        cur[j] -= 1
                for _ in range(machine):
                    self.operation[-1].append(".")

            self.operation[-1][-1] = "P"
            if odd:
                self.operation[-2][0] = "R"
                self.operation.append(["L"] + ["."] * (len(self.machine) * 2 - 1))
                self.hand[0][1] -= 1

        self.match = self.M

    def _v_five_construct(self):
        assert self.V == 5

        self._simple_construct()

    def _v_five_operate(self):
        assert self.V == 5

        self._simple_operate()

    def calc_score(self):
        self.construct_machine()
        self.operate_machine()
        if self.match == self.M:
            return len(self.operation)
        else:
            return 10**5 + 1000 * (self.M - self.match)


def main(input):
    solver = Solver(input)
    solver.solve()


def test(input):
    solver = Solver(input)
    print(solver.calc_score())


if __name__ == "__main__":
    if "ATCODER" in os.environ:
        main(lambda: stdin.readline().rstrip())
    else:
        # main(lambda: stdin.readline().rstrip())
        test(lambda: stdin.readline().rstrip())

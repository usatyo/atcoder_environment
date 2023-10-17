from collections import defaultdict, deque
from random import expovariate, randint, random, shuffle
from statistics import mean, pvariance


class Solver:
    def __init__(self, simulation, n, d, q, split, border):
        self.DEBUG_MODE = True
        self.SIMU = simulation

        if self.SIMU:
            self.n = randint(30, 100) if n == -1 else n
            self.d = randint(2, self.n // 4) if d == -1 else d
            self.q = (
                round(self.n * 2 ** (random() * 4 + 1))
                if q == -1
                else round(self.n * q)
            )
            self.w = [expovariate(lambd=10 ** (-5)) for _ in range(self.n)]
            if self.q / self.n >= 10:
                self.split = self.n
            else:
                self.split = split
            self.border = border
        else:
            self.n, self.d, self.q = map(int, input().split())
            if self.DEBUG_MODE:
                self.w = list(map(int, input().split()))
            self.split = max(
                2, min(self.n, round(2 ** ((self.q - 6 * self.d) / self.n)))
            )
            self.print("# split: ", self.split)
            self.border = 6

        self.rest = self.q
        self.ans = [i % self.d for i in range(self.n)]
        self.dic = defaultdict(set)
        self.label = defaultdict(int)

        self.print("#c", *self.ans)

        for i in range(self.d):
            for j in range(self.n):
                if j % self.d == i:
                    self.dic[i].add(j)

    def print(self, *args):
        if not self.SIMU:
            print(*args)

    def debug_balance(self, l, r):
        if self.rest == 0:
            self.output()
            raise Exception("true end")
        self.rest -= 1

        self.print(len(l), len(r), *l, *r)
        self.print("#c", *self.ans)
        sum_l = sum([self.w[idx] for idx in l])
        sum_r = sum([self.w[idx] for idx in r])
        if sum_l < sum_r:
            return 1
        elif sum_l > sum_r:
            return -1
        else:
            return 0

    def balance(self, l, r):
        if self.rest == 0:
            self.output()
            raise Exception("true end")
        self.rest -= 1

        self.print(len(l), len(r), *l, *r)
        res = input()
        if res == ">":
            return -1
        elif res == "<":
            return 1
        else:
            return 0

    def quick_sort(self, l, individual: bool, need, start, end, length):
        if len(l) <= 1:
            return l
        if need < start and end < length - need:
            return l
        pivot = l[0]
        left, right = [], []
        for val in l[1:]:
            if individual:
                pivot_list = [pivot]
                val_list = [val]
            else:
                pivot_list = list(self.dic[pivot])
                val_list = list(self.dic[val])
            if self.DEBUG_MODE:
                if self.debug_balance(pivot_list, val_list) >= 0:
                    right.append(val)
                else:
                    left.append(val)
            else:
                if self.balance(pivot_list, val_list) >= 0:
                    right.append(val)
                else:
                    left.append(val)

        left = self.quick_sort(left, individual, need, start, start + len(left), length)
        right = self.quick_sort(right, individual, need, end - len(right), end, length)
        return left + [pivot] + right

    def merge_sort(self, l, individual: bool, need, start, end, length):
        if len(l) == 1:
            return l
        left = self.merge_sort(
            l[: len(l) // 2], individual, need, start, start + len(l) // 2, length
        )
        right = self.merge_sort(
            l[len(l) // 2 :], individual, need, start + len(l) // 2, end, length
        )
        ret = []

        while len(left) > 0 and len(right) > 0:
            l_val = left.pop()
            r_val = right.pop()
            if individual:
                l_list = [l_val]
                r_list = [r_val]
            else:
                l_list = list(self.dic[l_val])
                r_list = list(self.dic[r_val])
            if self.DEBUG_MODE:
                if self.debug_balance(l_list, r_list) >= 0:
                    ret.append(r_val)
                    left.append(l_val)
                else:
                    ret.append(l_val)
                    right.append(r_val)
            else:
                if self.balance(l_list, r_list) >= 0:
                    ret.append(r_val)
                    left.append(l_val)
                else:
                    ret.append(l_val)
                    right.append(r_val)
        return left + right + ret[::-1]

    def pickup_max_min(self):
        ret_min = 0
        ret_max = 0
        for a in list(self.active - {0}):
            if self.DEBUG_MODE:
                ret_min = (
                    ret_min
                    if self.debug_balance(list(self.dic[ret_min]), list(self.dic[a]))
                    > 0
                    else a
                )
                ret_max = (
                    ret_max
                    if self.debug_balance(list(self.dic[ret_max]), list(self.dic[a]))
                    < 0
                    else a
                )
            else:
                ret_min = (
                    ret_min
                    if self.balance(list(self.dic[ret_min]), list(self.dic[a])) > 0
                    else a
                )
                ret_max = (
                    ret_max
                    if self.balance(list(self.dic[ret_max]), list(self.dic[a])) < 0
                    else a
                )
        return ret_min, ret_max

    def pickup(self, group):
        l = list(self.dic[group])
        l.sort(key=lambda x: self.label[x] + random())
        self.print("# pickup: ", *[str(val) + "," + str(self.label[val]) for val in l])
        idx = int((len(l) - 1) * self.rate)
        idx = max(0, idx)
        self.print("# idx: ", idx)
        return l[idx]

    def evaluate(self):
        original = list(range(self.n))
        shuffle(original)
        for i in range(self.n // self.split):
            l = original[i * self.split: (i + 1) * self.split]
            res = self.quick_sort(l, True, self.n, 0, self.split, self.split)
            for j in range(len(res)):
                self.label[res[j]] = j
            self.print("# res: ", res)
            self.print(
                "# sorted: ", *[str(self.w[r]) + ", " + str(self.label[r]) for r in res]
            )

        l = original[self.n // self.split * self.split:]
        res = self.quick_sort(l, True, self.n, 0, len(l), len(l))
        for i in range(len(l)):
            self.label[res[i]] = (self.split - len(l)) // 2 + i

        self.checkpoint = self.rest
        for i in range(self.n):
            print("# ", i, self.label[i])

    def swap(self):
        self.active = set(range(self.d))
        while True:
            a, b = self.pickup_max_min()
            self.rate = self.rest / self.checkpoint

            self.print("# active: ", *list(self.active))

            if len(self.dic[b]) <= 1:
                if b in self.active:
                    self.active.remove(b)
                continue
            target = self.pickup(b)

            self.print("# label: ", self.label[target])
            self.dic[b].remove(target)
            self.dic[a].add(target)
            self.ans[target] = a

    def pickup_sort(self, group):
        idx = round(len(self.dic[group]) * self.rate)
        idx = max(0, min(len(self.dic[group]) - 2, idx))
        l = list(self.dic[group])
        shuffle(l)
        l = self.quick_sort(
            l,
            True,
            idx,
            0,
            len(self.dic[group]),
            len(self.dic[group]),
        )
        return l[idx]

    def multiswap(self):
        self.print("# multi")
        self.active = set(range(self.d))
        while True:
            self.rate = self.rest / self.q
            prev_rest = self.rest
            loop = max(1, int(self.d * self.rate / 2))
            l = list(self.active)
            shuffle(l)
            deq = deque(
                self.quick_sort(
                    l,
                    False,
                    loop,
                    0,
                    len(self.active),
                    len(self.active),
                )
            )
            self.print("# loop", loop, "rest diff:", prev_rest - self.rest)

            while loop > 0 and len(deq) >= 2:
                a = deq.popleft()
                b = deq.pop()
                if len(self.dic[b]) <= 1:
                    deq.appendleft(a)
                    continue
                target = self.pickup_sort(b)

                self.dic[b].remove(target)
                self.dic[a].add(target)
                self.ans[target] = a
                loop -= 1

    def solve(self):
        try:
            if self.d <= self.border:
                self.evaluate()
                self.swap()
            else:
                self.multiswap()
        except Exception as e:
            if e.args[0] == "true end":
                if self.DEBUG_MODE:
                    return self.calc_score()
                else:
                    exit()
            else:
                print(e)
                exit()

    def output(self):
        self.print(*self.ans)

    def calc_score(self):
        weight = [0] * self.d
        for i in range(self.n):
            weight[self.ans[i]] += self.w[i]
        v = pvariance(weight)
        return 1 + round(100 * v**0.5)


solver = Solver(False, -1, -1, -1, -1, -1)
solver.solve()

# n_list = [30, 65, 100]
# d_list = [2, 4, 8, 16, 25]
# q_list = [2, 4, 8, 16, 32]
# split_list = [2, 4, 6]
# loop = 50

# for q in q_list:
#     for n in n_list:
#         for d in d_list:
#             best_split = -1
#             best_score = float("inf")
#             for split in split_list:
#                 score = 0
#                 for _ in range(loop):
#                     solver = Solver(True, n, d, q, split, border)
#                     score += solver.solve()
#                 mean = score // loop
#                 if best_score > mean:
#                     best_split = split
#                     best_score = mean

#             print(q, n, d, best_split, best_score)
#     print()

# border = range(1, 10)
# loop = 3000

# for b in border:
#     for split in split_list:
#         score = []
#         for _ in range(loop):
#             solver = Solver(True, 65, -1,-1, split, b)
#             score.append(solver.solve())
#         m = mean(score)

#         print(b, split, m)

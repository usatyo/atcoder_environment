from random import randint, random, shuffle


class generator:
    def generate(self):
        self.file = open("input.txt", "w")
        info = self._make_sample()
        self.file.close()
        return info

    def _make_sample(self):
        n = randint(1, 5)
        m = randint(0, n * (n - 1) // 2)
        edge = []
        for _ in range(m):
            while True:
                a = randint(1, n)
                b = randint(1, n)
                if a < b and (a, b) not in edge:
                    edge.append((a, b))
                    break

        self.file.write(" ".join([str(n), str(m), "\n"]))
        for i in range(m):
            self.file.write(" ".join(map(str, edge[i])) + "\n")

        return [n, m, edge]

    def _random_tree(self):
        pass

    def _gen_random_prime(self, min=2, max=10**8):
        pass

    def _push_random_array(self, n, min=1, max=10**8):
        l = [str(randint(min, max)) for _ in range(n)]
        self.file.write(" ".join(map(str, l)))

    def _push_random_permutation(self, n):
        ret = list(range(1, n + 1))
        shuffle(ret)
        self.file.write(" ".join(map(str, ret)))

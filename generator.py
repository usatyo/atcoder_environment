from random import randint, random, shuffle


class generator:
    def generate(self):
        self.file = open("input.txt", "w")
        self._make_sample()
        self.file.close()

    def _make_sample(self):
        n = randint(1, 7)
        m = randint(0, n * (n - 1) // 2)
        edges = []
        for _ in range(m):
            while True:
                e = (randint(1, n), randint(1, n))
                if e[0] < e[1] and not e in edges:
                    break
            edges.append(e)

        self.file.write(" ".join([str(n), str(m), "\n"]))
        for i in range(m):
            self.file.write(" ".join([str(edges[i][0]), str(edges[i][1]), "\n"]))

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

from random import randint, random, shuffle


class generator:
    def generate(self):
        self.file = open("input.txt", "w")
        info = self._make_sample()
        self.file.close()
        return info

    def _make_sample(self):
        n = randint(2, 100)
        edge = self._random_tree(n)
        self.file.write(f"{n}\n")
        for u, v in edge:
            self.file.write(f"{u} {v}\n")

    def _random_tree(self, n):
        assert n >= 2
        edge = []
        prufer = [randint(0, n - 1) for _ in range(n - 2)]
        d = [prufer.count(i) + 1 for i in range(n)]
        for i in range(n - 2):
            for j in range(n):
                if d[j] == 1:
                    edge.append((j + 1, prufer[i] + 1))
                    d[j] -= 1
                    d[prufer[i]] -= 1
                    break

        rest = []
        for i in range(n):
            if d[i] == 1:
                rest.append(i)
        edge.append((rest[0] + 1, rest[1] + 1))
        return edge

    def _gen_random_prime(self, min=2, max=10**3):
        while True:
            num = randint(min, max)
            for i in range(2, int(num**0.5) + 1):
                if num % i == 0:
                    break
            else:
                return num

    def _push_random_array(self, n, min=1, max=10**8):
        l = [str(randint(min, max)) for _ in range(n)]
        self.file.write(" ".join(map(str, l)))

    def _push_random_permutation(self, n):
        ret = list(range(1, n + 1))
        shuffle(ret)
        self.file.write(" ".join(map(str, ret)))

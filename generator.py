from random import randint, random, shuffle


class generator:
    def generate(self):
        self.file = open("input.txt", "w")
        n = randint(1, 10)
        k = randint(1, 10)
        self.file.write(f"1\n")
        self.file.write(f"{n} {k}\n")
        self.file.close()

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

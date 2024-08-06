from random import randint, shuffle


class generator:
    def generate(self):
        file = open("input.txt", "w")
        n = randint(1, 100)
        a = [randint(1, 100) for _ in range(n)]
        file.write(f"{n}\n")
        file.write(" ".join(map(str, a)) + "\n")
        file.close()

    def _random_tree(self):
        pass

    def _random_array(self, n, min=1, max=10**8):
        return [randint(min, max) for _ in range(n)]

    def _random_permutation(self, n):
        ret = list(range(1, n + 1))
        shuffle(ret)
        return ret

    def _random_prime(self, min=2, max=10**8):
        pass

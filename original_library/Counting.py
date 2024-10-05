class Counting:
    def __init__(self, mod, maxsize=10**6) -> None:
        self.mod = mod
        self.fact = [1]
        self.inv = [1]
        for i in range(maxsize):
            self.fact.append(self.fact[-1] * (i + 1) % mod)
            self.inv.append(pow(self.fact[-1], -1, mod))

    def c(self, n, k):
        return self.fact[n] * self.inv[k] * self.inv[n - k] % self.mod

    def p(self, n, k):
        return self.fact[n] * self.inv[n - k] % self.mod

    def h(self, n, k):
        return self.c(n + k - 1, k)

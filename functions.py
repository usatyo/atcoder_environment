# DANGER OF TLE
def p(n, r, q):
    ret = 1
    for i in range(r):
        ret = (ret * (n - i)) % q
    return ret


# DANGER OF TLE
def c(n, r, q):
    ret = 1
    for i in range(r):
        ret = (ret * (n - i)) % q
        ret = (ret * pow(i + 1, q - 2, q)) % q
    return ret


def gcd(a, b):
    x, y = min(a, b), max(a, b)
    while x != 0:
        x, y = y % x, x
    return y


from collections import defaultdict
from math import sqrt


def primes(n):
    ret = []
    dic = defaultdict(bool)
    root = int(sqrt(n)) + 1
    for i in range(2, root):
        if dic[i]:
            continue
        ret.append(i)
        for j in range(2, int(n / i) + 1):
            dic[j * i] = True
    for i in range(root, n):
        if dic[i]:
            continue
        ret.append(i)

    return ret


def run_length(s):
    ret = []
    ret.append([s[0], 1])
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            ret[-1][1] += 1
        else:
            ret.append([s[i], 1])
    return ret

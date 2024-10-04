from sys import stdin
from numpy import convolve

mod = 998244353


def main(input):
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))

    c = convolve(a, b)
    print(*[x % mod for x in c])


if __name__ == "__main__":
    main(lambda: stdin.readline().rstrip())

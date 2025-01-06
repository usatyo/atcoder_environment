from functools import reduce
from itertools import combinations
from sys import stdin


def main(input):
    k = int(input())
    d = int(input())

    ans = 0
    for i in range(1, k + 1):
        if sum(map(int, str(i))) % d == 0:
            ans += 1

    return ans


def honesty():
    file = open("input.txt", "r")
    ans = main(lambda: file.readline().rstrip())
    file.close()
    return ans


if __name__ == "__main__":
    main(lambda: stdin.readline().rstrip())

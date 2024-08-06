from itertools import product
from sys import stdin


def main(input):
    n = int(input())
    s = input().rstrip()
    hand = ["RPS".index(x) for x in s]
    ans = 0

    for p in product((0, 1, 2), repeat=n):
        for i in range(n - 1):
            if p[i] == p[i + 1]:
                break
        else:
            val = 0
            for i in range(n):
                if (p[i] - hand[i]) % 3 == 1:
                    val += 1
                elif (p[i] - hand[i]) % 3 == 2:
                    break
            else:
                ans = max(ans, val)

    return ans


def honesty():
    file = open("input.txt", "r")
    return main(file.readline)

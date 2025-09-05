from sys import stdin
from typing import Callable


def main(input: Callable[[], str]):
    n = int(input())
    ans = n + 1
    return ans


if __name__ == "__main__":
    main(lambda: stdin.readline().rstrip())

from sys import stdin


def main(input):
    n = int(input())

    ans = 0
    ans = n + 1

    return ans


def honesty():
    file = open("./texts/input.txt", "r")
    ans = main(lambda: file.readline().rstrip())
    file.close()
    return ans


if __name__ == "__main__":
    main(lambda: stdin.readline().rstrip())

from sys import stdin


def main(input):
    n = int(input())
    a = list(map(int, input().split()))
    ans = []
    for i in range(n):
        val = a[i]
        left = a[:i]
        right = a[i + 1 :][::-1]
        while True:
            if left and left[-1] < val:
                val += left.pop()
                continue
            if right and right[-1] < val:
                val += right.pop()
                continue
            break
        ans.append(val)

    return ans


def honesty():
    file = open("input.txt", "r")
    ans = main(lambda: file.readline().rstrip())
    file.close()
    return ans


if __name__ == "__main__":
    main(lambda: stdin.readline().rstrip())

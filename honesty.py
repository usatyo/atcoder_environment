def main(input):
    n = int(input())
    a = list(map(int, input().split()))
    ans = 0

    for i in range(n):
        for j in range(i + 1, n):
            val = 0
            for k in range(i, j + 1):
                val ^= a[k]
            ans += val

    return ans


def honesty():
    file = open("input.txt", "r")
    return main(lambda: file.readline().rstrip())

def main(input):
    n = int(input())
    a = list(map(int, input().split()))
    a.sort()
    return a


def honesty():
    file = open("input.txt", "r")
    ans = main(lambda: file.readline().rstrip())
    file.close()
    return ans

from sys import stdin

"""n を素因数分解. O(sqrt n)

Args:
    n (int): 素因数分解の対象

Returns:
    int[][]: [[素因数, 指数], ...]の2次元リスト.
"""


def factorization(n):

    if n == 1:
        return []

    arr = []
    temp = n
    for i in range(2, int(n**0.5 + 10)):
        if temp % i == 0:
            cnt = 0
            while temp % i == 0:
                cnt += 1
                temp //= i
            arr.append([i, cnt])

    if temp != 1:
        arr.append([temp, 1])

    if arr == []:
        arr.append([n, 1])

    return arr


def main(input):
    #!/usr/bin/python3

    small_n = 1
    large_n = 1

    n = int(input())
    fact = factorization(n)

    for k, v in fact:
        if k == 2:
            small_n *= v * 2 - 1
        else:
            small_n *= v * 2 + 1

        if k % 4 == 1:
            large_n *= v * 2 + 1

    # 重複と二等辺三角形を除去するために2で割る
    ans = small_n // 2 + large_n // 2

    # print(ans)
    return ans


def honesty():
    file = open("input.txt", "r")
    ans = main(lambda: file.readline().rstrip())
    file.close()
    return ans


if __name__ == "__main__":
    main(lambda: stdin.readline().rstrip())

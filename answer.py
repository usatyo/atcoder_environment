from sys import stdin


def main(input):
    hand = ["R", "P", "S"]

    n = int(input())
    s = [hand.index(x) for x in input().rstrip()]

    dp = [[0, 0, 0] for _ in range(n + 1)]

    for i in range(n):
        for j in range(3):
            if (s[i] - j) % 3 == 1:
                dp[i + 1][j] = -(10**6)
            elif (s[i] - j) % 3 == 2:
                dp[i + 1][j] = max(dp[i][(j + 1) % 3], dp[i][(j + 2) % 3]) + 1
            else:
                dp[i + 1][j] = max(dp[i][(j + 1) % 3], dp[i][(j + 2) % 3])

    return max(dp[n])


def _format(ans):
    if type(ans) == str or type(ans) == int or type(ans) == float:
        return str(ans)
    elif type(ans) == list or type(ans) == tuple:
        if type(ans[0]) == list or type(ans[0]) == tuple:
            return "\n".join([(" ".join([str(x) for x in a])) for a in ans])
        else:
            return "\n".join(ans)
    else:
        raise TypeError("Return type is not supported.")


if __name__ == "__main__":
    ans = main(stdin.readline)
    ans = _format(ans)
    print(ans)

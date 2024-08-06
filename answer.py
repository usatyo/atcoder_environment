from sys import stdin


def main(input):
    n, m = map(int, input().split())
    return


if __name__ == "__main__":
    ans = main(stdin.readline)
    if type(ans) == str or type(ans) == int:
        print(ans)
    elif type(ans) == list or type(ans) == tuple:
        print(*ans, sep="\n")
    else:
        raise TypeError("Return type is not supported.")

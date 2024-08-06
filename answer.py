from sys import stdin


def main(input):
    ans = 0

    return ans


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
    ans = main(lambda: stdin.readline().rstrip())
    ans = _format(ans)
    print(ans)

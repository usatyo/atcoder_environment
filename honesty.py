def main(input):
    pass


def honesty():
    file = open("input.txt", "r")
    ans = main(lambda: file.readline().rstrip())
    file.close()
    return ans

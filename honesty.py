def main(input):
    ans = input()
    return ans


def honesty():
    file = open("input.txt", "r")
    return main(lambda: file.readline().rstrip())

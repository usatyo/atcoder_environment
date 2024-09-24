from sys import stdin


class Solver:
    def __init__(self, input) -> None:
        self.N, self.M, self.T, self.la, self.lb = map(int, input().split())
        for _ in range(self.M):
            u, v = map(int, input().split())
        self.t = list(map(int, input().split()))
        for _ in range(self.N):
            x,y = map(int, input().split())
            

def main(input):
    pass


if __name__ == "__main__":
    main(lambda: stdin.readline().rstrip())

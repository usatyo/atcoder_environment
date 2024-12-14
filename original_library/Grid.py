from os import environ


class Grid:
    BLACK = 0
    RED = 1
    GREEN = 2
    YELLOW = 3
    BLUE = 4
    MAGENTA = 5
    CYAN = 6
    WHITE = 7

    def __init__(self, h, w) -> None:
        self.h = h
        self.w = w
        self._grid = [[self.WHITE] * self.w for _ in range(self.h)]

    def draw(self, x, y, color=RED):
        self._grid[x][y] = color

    def cell(self, color):
        return f"\033[3{color}m██\033[0m"

    def output(self):
        if "ATCODER" in environ:
            return
        for row in self._grid:
            print(*[self.cell(x) for x in row], sep="")

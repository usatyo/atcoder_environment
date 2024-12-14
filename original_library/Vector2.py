from math import cos, sin


class Vector2:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other: "Vector2"):
        return Vector2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vector2"):
        return Vector2(self.x - other.x, self.y - other.y)

    def __mul__(self, other: int):
        return Vector2(self.x * other, self.y * other)

    def __truediv__(self, other):
        assert other != 0, "division by zero"
        return Vector2(self.x / other, self.y / other)

    def __abs__(self):
        return (self.x**2 + self.y**2) ** 0.5

    def __eq__(self, other: "Vector2"):
        return self.x == other.x and self.y == other.y

    def __ne__(self, other: "Vector2"):
        return self.x != other.x or self.y != other.y

    def __str__(self):
        return f"({self.x:.3f}, {self.y:.3f})"

    def dot(self, other: "Vector2"):
        return self.x * other.x + self.y * other.y

    def cross(self, other: "Vector2"):
        return self.x * other.y - self.y * other.x

    def rotate(self, theta):
        return Vector2(
            self.x * cos(theta) - self.y * sin(theta),
            self.x * sin(theta) + self.y * cos(theta),
        )

    def norm(self):
        return abs(self)

    def square_norm(self):
        return self.x**2 + self.y**2

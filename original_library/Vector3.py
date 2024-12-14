from math import cos, sin


class Vector3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other: "Vector3"):
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vector3"):
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other: int):
        return Vector3(self.x * other, self.y * other, self.z * other)

    def __truediv__(self, other):
        assert other != 0, "division by zero"
        return Vector3(self.x / other, self.y / other, self.z / other)

    def __abs__(self):
        return (self.x**2 + self.y**2 + self.z**2) ** 0.5

    def __eq__(self, other: "Vector3"):
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __ne__(self, other: "Vector3"):
        return self.x != other.x or self.y != other.y and self.z != other.z

    def __str__(self):
        return f"({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"

    def dot(self, other: "Vector3"):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vector3"):
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def norm(self):
        return abs(self)

    def square_norm(self):
        return self.x**2 + self.y**2 + self.z**2

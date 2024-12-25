from collections import deque
from math import atan2, cos, pi, sin

EPS = 1e-7
SCALE = 10**5
DIGITS = 10


def equal(a, b):
    return abs(a - b) < EPS


class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other: "Vector"):
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vector"):
        return Vector(self.x - other.x, self.y - other.y)

    def __mul__(self, other):
        return Vector(self.x * other, self.y * other)

    def __truediv__(self, other):
        assert other != 0, "division by zero"
        return Vector(self.x / other, self.y / other)

    def __abs__(self):
        return (self.x**2 + self.y**2) ** 0.5

    def __eq__(self, other: "Vector"):
        return equal(self.x, other.x) and equal(self.y, other.y)

    def __ne__(self, other: "Vector"):
        return not (self == other)

    def __str__(self):
        return f"{self.x:.{DIGITS}f} {self.y:.{DIGITS}f}"

    def format(self):
        return f"({self.x:.{DIGITS}f}, {self.y:.{DIGITS}f})"

    def dot(self, other: "Vector"):
        return self.x * other.x + self.y * other.y

    def cross(self, other: "Vector"):
        return self.x * other.y - self.y * other.x

    def move(self, dx, dy):
        return Vector(self.x + dx, self.y + dy)

    def rotate(self, theta, origin: "Vector" = None):
        if origin is None:
            origin = Vector(0, 0)
        self = self - origin
        return origin + Vector(
            self.x * cos(theta) - self.y * sin(theta),
            self.x * sin(theta) + self.y * cos(theta),
        )

    def norm(self):
        return abs(self)

    def square_norm(self):
        return self.x**2 + self.y**2

    def copy(self):
        return Vector(self.x, self.y)

    def ccw(self, other: "Vector"):
        if equal(self.cross(other), 0):
            return 0
        elif self.cross(other) < 0:
            return -1
        else:
            return 1

    def unit_vector(self):
        if equal(abs(self), 0):
            return Vector(0, 0)
        return self / abs(self)


class Segment:
    def __init__(self, p1: Vector, p2: Vector, extend1=False, extend2=False):
        self.p1 = p1.copy()
        self.p2 = p2.copy()
        if extend1:
            self.extend(reverse=True)
        if extend2:
            self.extend()

    def __abs__(self):
        return abs(self.p2 - self.p1)

    def __str__(self):
        return f"{self.p1} {self.p2}"

    def fromat(self):
        return f"{self.p1.format()} -- {self.p2.format()}"

    def to_vector(self):
        return self.p2 - self.p1

    def extend(self, dist=SCALE, reverse=False):
        if self.p1 == self.p2:
            return False
        if reverse:
            self.p1 = self.p2 + self.to_vector().unit_vector() * (abs(self) + dist)
        else:
            self.p2 = self.p1 + self.to_vector().unit_vector() * (abs(self) + dist)
        return True

    def projection(self, p: Vector):
        base = self.to_vector()
        return self.p1 + base * (p - self.p1).dot(base) / base.square_norm()

    def reflection(self, p: Vector):
        return p + (self.projection(p) - p) * 2

    def is_parallel(self, other: "Segment"):
        return equal(self.to_vector().cross(other.to_vector()), 0)

    def is_orthogonal(self, other: "Segment"):
        return equal(self.to_vector().dot(other.to_vector()), 0)

    def is_contain_point(self, p: Vector):
        if self.p1 == p or self.p2 == p:
            return True
        if self.p1 == self.p2:
            return False
        onLine = self.to_vector().ccw(p - self.p1) == 0
        between = (
            -EPS
            < self.to_vector().dot(p - self.p1)
            < self.to_vector().square_norm() + EPS
        )
        return onLine and between

    def is_crossing(self, other: "Segment"):
        if (
            self.is_contain_point(other.p1)
            or self.is_contain_point(other.p2)
            or other.is_contain_point(self.p1)
            or other.is_contain_point(self.p2)
        ):
            return True

        return self.to_vector().ccw(other.p1 - self.p1) != self.to_vector().ccw(
            other.p2 - self.p1
        ) and other.to_vector().ccw(self.p1 - other.p1) != other.to_vector().ccw(
            self.p2 - other.p1
        )

    def crossing_point(self, other: "Segment"):
        if self.is_parallel(other) or not self.is_crossing(other):
            return None
        d1 = self.to_vector().cross(other.to_vector())
        d2 = self.to_vector().cross(self.p2 - other.p1)
        if equal(d1, 0) and equal(d2, 0):
            return other.p1
        return other.p1 + other.to_vector() * (d2 / d1)

    def distance_to_point(self, p: Vector, line=False):
        projection = self.projection(p)
        if line or self.is_contain_point(projection):
            return abs(p - projection)
        return min(abs(p - self.p1), abs(p - self.p2))

    def distance_to_segment(self, other: "Segment"):
        if self.is_crossing(other):
            return 0
        return min(
            self.distance_to_point(other.p1),
            self.distance_to_point(other.p2),
            other.distance_to_point(self.p1),
            other.distance_to_point(self.p2),
        )


class Polygon:
    def __init__(self, points: list[Vector]):
        """初期化

        Args:
            points (list[Vector]): 頂点を反時計回りに追加したリスト
        """
        self.points = [p.copy() for p in points]
        self.n = len(self.points)

    def __str__(self):
        return "\n".join([str(p) for p in self.points])

    def format(self):
        return " -> ".join([p.format() for p in self.points])

    def area(self):
        area = 0
        for i in range(self.n):
            p1 = self.points[i]
            p2 = self.points[(i + 1) % self.n]
            area += p1.cross(p2) / 2
        return abs(area)

    def is_convex(self):
        top = 0
        bottom = 0
        for i in range(self.n):
            a = self.points[i]
            b = self.points[(i + 1) % self.n]
            c = self.points[(i + 2) % self.n]
            top = max(top, (b - a).ccw(c - b))
            bottom = min(bottom, (b - a).ccw(c - b))
        return not (top == 1 and bottom == -1)

    def is_inside(self, p: Vector):
        """多角形と点の位置関係を判定

        Args:
            p (Vector): 判定対象の点

        Returns:
            int: 1: 内部, 0: 線上, -1: 外部
        """

        theta = 0  # p の周りを何度周回するか
        for i in range(self.n):
            a = self.points[i]
            b = self.points[(i + 1) % self.n]
            if Segment(a, b).is_contain_point(p):
                return 0
            theta += atan2((a - p).cross(b - p), (a - p).dot(b - p))
        return -1 if equal(theta, 0) else 1

    def construct_convex_hull(self):
        points = self.points
        points.sort(key=lambda p: (p.y, p.x))

        if self.n <= 2:
            return 2
        elif self.n == 3:
            if (points[1] - points[0]).ccw(points[2] - points[0]) < 0:
                self.points = [points[0], points[2], points[1]]
            return 3

        right = deque([points[0].copy(), points[1].copy()])
        for i in range(2, self.n):
            next = points[i].copy()
            curr = right.pop()
            prev = right.pop()
            while True:
                if (curr - prev).ccw(next - curr) >= 0:
                    right.append(prev)
                    right.append(curr)
                    break
                if len(right) == 0:
                    right.append(prev)
                    break
                curr = prev.copy()
                prev = right.pop()

            right.append(next)

        left = deque([points[self.n - 1].copy(), points[self.n - 2].copy()])
        for i in range(self.n - 2)[::-1]:
            next = points[i].copy()
            curr = left.pop()
            prev = left.pop()
            while True:
                if (curr - prev).ccw(next - curr) >= 0:
                    left.append(prev)
                    left.append(curr)
                    break
                if len(left) == 0:
                    left.append(prev)
                    break
                curr = prev.copy()
                prev = left.pop()

            left.append(next)

        right.pop()
        left.pop()
        self.points = list(right) + list(left)
        self.n = len(self.points)
        return self.n

    def diameter(self):
        self.construct_convex_hull()
        if self.n == 2:
            return abs(self.points[0] - self.points[1])
        i = j = 0
        for k in range(self.n):
            if self.points[k].x < self.points[i].x:
                i = k
            if self.points[k].x > self.points[j].x:
                j = k
        res = 0
        si, sj = i, j
        while i != sj or j != si:
            res = max(res, abs(self.points[i] - self.points[j]))
            vi = self.points[(i + 1) % self.n] - self.points[i]
            vj = self.points[(j + 1) % self.n] - self.points[j]
            if vi.cross(vj) < 0:
                i = (i + 1) % self.n
            else:
                j = (j + 1) % self.n

        return res

    def common_polygon(self, other: "Polygon"):
        """多角形同士の共通部分

        Args:
            other (Polygon): もう片方の凸多角形
        """
        self.construct_convex_hull()
        other.construct_convex_hull()

        points = []

        for p in self.points:
            if other.is_inside(p) == 1:
                points.append(p)

        for p in other.points:
            if self.is_inside(p) == 1:
                points.append(p)

        for i in range(self.n):
            seg1 = Segment(self.points[i], self.points[(i + 1) % self.n])
            for j in range(other.n):
                seg2 = Segment(other.points[j], other.points[(j + 1) % other.n])
                if seg1.is_crossing(seg2):
                    points.append(seg1.crossing_point(seg2))

        polygon = Polygon(points)
        polygon.construct_convex_hull()
        return polygon


class Circle:
    def __init__(self, center: Vector, radius: float):
        assert radius > 0, "radius must be positive"
        self.center = center.copy()
        self.radius = radius

    def __str__(self):
        return f"{self.center} {self.radius}"

    def format(self):
        return f"o: {self.center.format()}, r: {self.radius:.{DIGITS}f}"

    def is_touching_circle(self, other: "Circle"):
        """円が接しているかどうかを判定

        Args:
            other (Circle): もう片方の円

        Returns:
            int: 1: 内接, 0: 接しない, -1: 外接
        """
        if self.center == other.center:
            return 0
        elif equal(abs(self.center - other.center), abs(self.radius - other.radius)):
            return 1
        elif equal(abs(self.center - other.center), self.radius + other.radius):
            return -1
        else:
            return 0

    def is_crossing_circle(self, other: "Circle"):
        """円が交差しているかどうかを判定

        Args:
            other (Circle): もう片方の円

        Returns:
            bool: True: 交差, False: 交差しない
        """
        if self.is_touching_circle(other):
            return False
        elif abs(self.center - other.center) < abs(self.radius - other.radius):
            return False
        elif self.radius + other.radius < abs(self.center - other.center):
            return False
        else:
            return True

    def crossing_points_circle(self, other: "Circle"):
        if self.is_touching_circle(other) == 1:
            unit = (other.center - self.center).unit_vector()
            if self.radius > other.radius:
                return [self.center + unit * self.radius]
            else:
                return [other.center - unit * other.radius]
        elif self.is_touching_circle(other) == -1:
            unit = (other.center - self.center).unit_vector()
            return [self.center + unit * self.radius]
        elif self.is_crossing_circle(other):
            dist = abs(self.center - other.center)
            cosine = (self.radius**2 - other.radius**2 + dist**2) / (2 * dist)
            h = (self.radius**2 - cosine**2) ** 0.5
            unit = (other.center - self.center).unit_vector()
            p = self.center + unit * cosine
            return [p + unit.rotate(pi / 2) * h, p - unit.rotate(pi / 2) * h]
        else:
            return []

    def is_touching_segment(self, other: Segment):
        return equal(other.distance_to_point(self.center, line=True), self.radius)

    def is_crossing_segment(self, other: Segment):
        if self.is_touching_segment(other):
            return False
        return other.distance_to_point(self.center, line=True) < self.radius

    def crossing_points_segment(self, other: Segment):
        if not self.is_touching_segment(other) and not self.is_crossing_segment(other):
            return []
        projection = other.projection(self.center)
        if self.is_touching_segment(other):
            return [projection]
        dist = abs(projection - self.center)
        unit = other.to_vector().unit_vector()
        d = (self.radius**2 - dist**2) ** 0.5
        return [projection + unit * d, projection - unit * d]

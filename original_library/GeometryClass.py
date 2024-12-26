from collections import deque
from math import atan2, cos, pi, sin

EPS = 1e-8
SCALE = 10**5
DIGITS = 10


def equal(a, b):
    return abs(a - b) < EPS


class Vector:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y

    def __add__(self, other: "Vector") -> "Vector":
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vector") -> "Vector":
        return Vector(self.x - other.x, self.y - other.y)

    def __mul__(self, other) -> "Vector":
        return Vector(self.x * other, self.y * other)

    def __truediv__(self, other) -> "Vector":
        assert other != 0, "division by zero"
        return Vector(self.x / other, self.y / other)

    def __abs__(self) -> float:
        return (self.x**2 + self.y**2) ** 0.5

    def __eq__(self, other: "Vector") -> bool:
        return equal(self.x, other.x) and equal(self.y, other.y)

    def __ne__(self, other: "Vector") -> bool:
        return not (self == other)

    def __str__(self) -> str:
        return f"{self.x:.{DIGITS}f} {self.y:.{DIGITS}f}"

    def format(self) -> str:
        return f"({self.x:.{DIGITS}f}, {self.y:.{DIGITS}f})"

    def dot(self, other: "Vector") -> float:
        """内積

        Args:
            other (Vector): 演算対象

        Returns:
            float: 計算結果
        """
        return self.x * other.x + self.y * other.y

    def cross(self, other: "Vector") -> float:
        """外積

        Args:
            other (Vector): 演算対象のベクトル

        Returns:
            float: 計算結果
        """
        return self.x * other.y - self.y * other.x

    def move(self, dx, dy) -> "Vector":
        """平行移動

        Args:
            dx (float): x軸方向の移動量
            dy (float): y軸方向の移動量

        Returns:
            Vector: 移動後の座標
        """
        return Vector(self.x + dx, self.y + dy)

    def rotate(self, theta, origin: "Vector" = None) -> "Vector":
        """回転移動

        Args:
            theta (float): 回転する角度（ラジアン）
            origin (Vector, optional): 原点. Defaults to (0, 0).

        Returns:
            Vector: 移動後の座標
        """
        if origin is None:
            origin = Vector(0, 0)
        self = self - origin
        return origin + Vector(
            self.x * cos(theta) - self.y * sin(theta),
            self.x * sin(theta) + self.y * cos(theta),
        )

    def square_norm(self) -> float:
        return self.x**2 + self.y**2

    def copy(self) -> "Vector":
        return Vector(self.x, self.y)

    def ccw(self, other: "Vector") -> int:
        """回転方向を判定

        Args:
            other (Vector): もう片方のベクトル

        Returns:
            int: (self から見て other が) 1: 反時計回り, -1: 時計回り, 0: 直線上
        """
        if equal(self.cross(other), 0):
            return 0
        elif self.cross(other) < 0:
            return -1
        else:
            return 1

    def unit_vector(self) -> "Vector":
        """単位ベクトルを取得

        Returns:
            Vector: 同じ方向の単位ベクトル
        """
        if equal(abs(self), 0):
            return Vector(0, 0)
        return self / abs(self)


class Segment:
    def __init__(self, p1: Vector, p2: Vector) -> None:
        self.p1 = p1.copy()
        self.p2 = p2.copy()

    def __abs__(self) -> float:
        return abs(self.p2 - self.p1)

    def __str__(self) -> str:
        return f"{self.p1} {self.p2}"

    def format(self) -> str:
        return f"{self.p1.format()} -- {self.p2.format()}"

    def to_vector(self) -> Vector:
        return self.p2 - self.p1

    def coef(self) -> float:
        """傾き

        Returns:
            float: 直線の傾き, y軸に平行な場合は inf
        """
        if equal(self.p1.x, self.p2.x):
            return float("inf")
        else:
            return (self.p2.y - self.p1.y) / (self.p2.x - self.p1.x)

    def projection(self, p: Vector) -> Vector:
        """射影

        Args:
            p (Vector): もとの座標

        Returns:
            Vector: 移動後の座標
        """
        base = self.to_vector()
        return self.p1 + base * (p - self.p1).dot(base) / base.square_norm()

    def reflection(self, p: Vector) -> Vector:
        """反射

        Args:
            p (Vector): もとの座標

        Returns:
            Vector: 反射後の座標
        """
        return p + (self.projection(p) - p) * 2

    def bisecter(self) -> "Segment":
        """垂直二等分線

        Returns:
            Segment: 計算結果の線分
        """
        center = (self.p1 + self.p2) / 2
        p1 = self.p1.rotate(pi / 2, center)
        p2 = self.p2.rotate(pi / 2, center)
        return Segment(p1, p2)

    def is_parallel(self, other: "Segment") -> bool:
        """平行かどうか判定

        Args:
            other (Segment): 比較対象の線分

        Returns:
            bool: True: 平行, False: 平行でない
        """
        return equal(self.to_vector().cross(other.to_vector()), 0)

    def is_orthogonal(self, other: "Segment") -> bool:
        """ "垂直かどうか判定

        Args:
            other (Segment): 比較対象の線分

        Returns:
            bool: True: 垂直, False: 垂直でない
        """
        return equal(self.to_vector().dot(other.to_vector()), 0)

    def is_contain_point(self, p: Vector) -> bool:
        """線分上に点 p が存在するかどうか

        Args:
            p (Vector): 判定対象の点

        Returns:
            bool: True: 線分上に存在, False: 線分上に存在しない
        """
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

    def is_crossing(self, other: "Segment") -> bool:
        """線分の交差判定

        Args:
            other (Segment): 判定対象の線分

        Returns:
            bool: True: 交差, False: 交差しない
        """
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

    def crossing_point(self, other: "Segment") -> Vector:
        """線分同士の交点

        Args:
            other (Segment): 対象の線分

        Returns:
            Vector: 交点の座標. 交差しない場合は None
        """
        if self.is_parallel(other) or not self.is_crossing(other):
            return None
        d1 = self.to_vector().cross(other.to_vector())
        d2 = self.to_vector().cross(self.p2 - other.p1)
        if equal(d1, 0) and equal(d2, 0):
            return other.p1
        return other.p1 + other.to_vector() * (d2 / d1)

    def distance_to_point(self, p: Vector, line=False) -> float:
        """線分と点の距離

        Args:
            p (Vector): 対象の点
            line (bool, optional): 直線に変更する場合 True. Defaults to False.

        Returns:
            float: 距離
        """
        projection = self.projection(p)
        if line or self.is_contain_point(projection):
            return abs(p - projection)
        return min(abs(p - self.p1), abs(p - self.p2))

    def distance_to_segment(self, other: "Segment") -> float:
        """線分と線分の距離

        Args:
            other (Segment): 対象の線分

        Returns:
            float: 最も近い2点の距離
        """
        if self.is_crossing(other):
            return 0
        return min(
            self.distance_to_point(other.p1),
            self.distance_to_point(other.p2),
            other.distance_to_point(self.p1),
            other.distance_to_point(self.p2),
        )


class Polygon:
    def __init__(self, points: list[Vector]) -> None:
        """初期化

        Args:
            points (list[Vector]): 頂点を反時計回りに追加したリスト
        """
        self.points = [p.copy() for p in points]
        self.n = len(self.points)

    def __str__(self) -> str:
        return "\n".join([str(p) for p in self.points])

    def format(self) -> str:
        return " -> ".join([p.format() for p in self.points])

    def area(self) -> float:
        """多角形内部の面積

        Returns:
            float: 面積
        """
        area = 0
        for i in range(self.n):
            p1 = self.points[i]
            p2 = self.points[(i + 1) % self.n]
            area += p1.cross(p2) / 2
        return abs(area)

    def is_convex(self) -> bool:
        """凸多角形かどうか判定

        Returns:
            bool: True: 凸多角形, False: 凹多角形. 3点が一直線上にある場合も True
        """
        top = 0
        bottom = 0
        for i in range(self.n):
            a = self.points[i]
            b = self.points[(i + 1) % self.n]
            c = self.points[(i + 2) % self.n]
            top = max(top, (b - a).ccw(c - b))
            bottom = min(bottom, (b - a).ccw(c - b))
        return not (top == 1 and bottom == -1)

    def is_inside(self, p: Vector) -> int:
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

    def construct_convex_hull(self) -> int:
        """現在 self に含まれている点から凸包を構成し、自身を置き換える

        Returns:
            int: 凸包の頂点数
        """
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

    def diameter(self) -> float:
        """多角形の直径（最遠点対）

        Returns:
            float: 直径
        """
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

    def common_polygon(self, other: "Polygon") -> "Polygon":
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
    def __init__(self, center: Vector, radius: float) -> None:
        assert radius > 0, "radius must be positive"
        self.center = center.copy()
        self.radius = radius

    def __str__(self) -> str:
        return f"{self.center} {self.radius}"

    def format(self) -> str:
        return f"o: {self.center.format()}, r: {self.radius:.{DIGITS}f}"

    def is_touching_circle(self, other: "Circle") -> int:
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

    def is_crossing_circle(self, other: "Circle") -> bool:
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

    def crossing_points_circle(self, other: "Circle") -> list[Vector]:
        """円と円の交点

        Args:
            other (Circle): 対象の円

        Returns:
            list[Vector]: 0~2個の交点を含むリスト
        """
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

    def is_touching_line(self, other: Segment) -> bool:
        """直線と円が接しているかどうかを判定

        Args:
            other (Segment): 対象の直線

        Returns:
            bool: True: 接している, False: 接していない
        """
        return equal(other.distance_to_point(self.center, line=True), self.radius)

    def is_crossing_line(self, other: Segment) -> bool:
        """直線と円が2点以上で交わるかどうかを判定

        Args:
            other (Segment): 対象の直線

        Returns:
            bool: 2点以上で交わるかどうか
        """
        if self.is_touching_line(other):
            return False
        return other.distance_to_point(self.center, line=True) < self.radius

    def crossing_points_segment(self, other: Segment) -> list[Vector]:
        """直線と円の交点

        Args:
            other (Segment): 対象の直線

        Returns:
            list[Vector]: 0~2個の交点を含むリスト
        """
        if not self.is_touching_line(other) and not self.is_crossing_line(other):
            return []
        projection = other.projection(self.center)
        if self.is_touching_line(other):
            return [projection]
        dist = abs(projection - self.center)
        unit = other.to_vector().unit_vector()
        d = (self.radius**2 - dist**2) ** 0.5
        return [projection + unit * d, projection - unit * d]


class PillowManager:
    SIZE = 1000
    OFFSET = 0

    def __init__(self, bottom=0, top=500, axis=True, grid: int = None) -> None:
        from PIL import Image, ImageDraw

        self.im = Image.new(
            "RGB",
            (self.SIZE + self.OFFSET * 2, self.SIZE + self.OFFSET * 2),
            (255, 255, 255),
        )
        self.draw = ImageDraw.Draw(self.im)
        self.bottom = bottom
        self.top = top
        axis and self._add_axis()
        (grid is not None) and self._add_grid(grid)

    def _check_point(self, p: Vector) -> None:
        assert self.bottom <= p.x <= self.top, f"{p.x=} is out of range"
        assert self.bottom <= p.y <= self.top, f"{p.y=} is out of range"

    def _convert(self, p: Vector) -> Vector:
        magn = self.SIZE / (self.top - self.bottom)
        x = (p.x - self.bottom) * magn + self.OFFSET
        y = (p.y - self.bottom) * magn + self.OFFSET
        return Vector(x, y)

    def _add_axis(self) -> None:
        axis_x = Segment(Vector(self.bottom, 0), Vector(self.top, 0))
        axis_y = Segment(Vector(0, self.bottom), Vector(0, self.top))
        self.draw_segment(axis_x, width=3, color=(200, 200, 200))
        self.draw_segment(axis_y, width=3, color=(200, 200, 200))

    def _add_grid(self, grid: int) -> None:
        for i in range(0, self.top + 1, grid):
            parallel_x = Segment(Vector(i, self.bottom), Vector(i, self.top))
            parallel_y = Segment(Vector(self.bottom, i), Vector(self.top, i))
            self.draw_segment(parallel_x, color=(200, 200, 200))
            self.draw_segment(parallel_y, color=(200, 200, 200))
        for i in range(0, self.bottom - 1, -grid):
            parallel_x = Segment(Vector(i, self.bottom), Vector(i, self.top))
            parallel_y = Segment(Vector(self.bottom, i), Vector(self.top, i))
            self.draw_segment(parallel_x, color=(200, 200, 200))
            self.draw_segment(parallel_y, color=(200, 200, 200))

    def draw_point(self, p: Vector, size=None, color=(0, 0, 0)) -> None:
        self._check_point(p)
        if size is None:
            size = self.SIZE / 150
        center = self._convert(p)
        self.draw.ellipse(
            (center.x - size, center.y - size, center.x + size, center.y + size),
            fill=color,
        )

    def draw_segment(self, segment: Segment, width=1, color=(0, 0, 0)) -> None:
        p1 = self._convert(segment.p1)
        p2 = self._convert(segment.p2)
        self.draw.line((p1.x, p1.y, p2.x, p2.y), fill=color, width=width)

    def draw_circle(self, circle: Circle, color=(0, 0, 0)) -> None:
        lb = self._convert(circle.center - Vector(circle.radius, circle.radius))
        rt = self._convert(circle.center + Vector(circle.radius, circle.radius))
        self.draw.ellipse(
            (lb.x, lb.y, rt.x, rt.y),
            outline=color,
        )

    def save(self, path: str = "./pillow_image.jpg") -> None:
        from PIL import ImageOps

        ImageOps.flip(self.im).save(path, quality=95)

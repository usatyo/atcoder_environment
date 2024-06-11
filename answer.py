# 出力が指数形式にならないよう注意
# print(f"{num:.10f}") などとする
# 適当な大きさの素数 99961, 99971, 99989, 99991


# 内積
from collections import deque
from math import atan2


def dot(v, w):
    return v[0] * w[0] + v[1] * w[1]


# 外積の係数(= vを90度回転させた内積)
def cross(v, w, o=[0, 0]):
    v = [v[0] - o[0], v[1] - o[1]]
    w = [w[0] - o[0], w[1] - o[1]]
    return v[0] * w[1] - v[1] * w[0]


# v に対する w の射影
def project(v, w, o=[0, 0]):
    v = [v[0] - o[0], v[1] - o[1]]
    w = [w[0] - o[0], w[1] - o[1]]
    return [o[0] + v[0] * dot(v, w) / dot(v, v), o[1] + v[1] * dot(v, w) / dot(v, v)]


# v に対する w の反射(対称な位置にある点)
def reflect(v, w):
    p = project(v, w)
    return [2 * p[0] - w[0], 2 * p[1] - w[1]]


# v を基準とした w の回転方向判定
# 反時計回り:1, 時計回り:-1, 直線上:0
# 直線上の判定は整数の場合のみ可能
def counter_clockwise(v, w, o=[0, 0]):
    v = [v[0] - o[0], v[1] - o[1]]
    w = [w[0] - o[0], w[1] - o[1]]
    if cross(v, w) > 0:
        return 1
    elif cross(v, w) < 0:
        return -1
    else:
        return 0


# w が v の線分上にあるか判定
# 整数のみ使用可能
def on_segment(v, w, o=[0, 0]):
    v = [v[0] - o[0], v[1] - o[1]]
    w = [w[0] - o[0], w[1] - o[1]]
    if counter_clockwise(v, w) != 0:
        return False
    return 0 <= dot(v, w) <= dot(v, v)


# 線分 p0-p1 と q0-q1 の交差判定
def intersect_segment(p0, p1, q0, q1):
    return (
        counter_clockwise(p0, q1, q0) * counter_clockwise(q1, p1, q0) > 0
        and counter_clockwise(q0, p1, p0) * counter_clockwise(p1, q1, p0) > 0
        or on_segment(p1, q0, p0)
        or on_segment(p1, q1, p0)
        or on_segment(q1, p0, q0)
        or on_segment(q1, p1, q0)
    )


# 線分 p0-p1 と q0-q1 の交点の座標
def intersection_point(p0, p1, q0, q1):
    t = cross(q0, p1, p0) / cross(
        [p1[0] - p0[0], p1[1] - p0[1]], [q1[0] - q0[0], q1[1] - q0[1]]
    )
    return [q0[0] + t * (q1[0] - q0[0]), q0[1] + t * (q1[1] - q0[1])]


# 点 p と点 q の距離
def distance(p, q):
    return ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5


# 点 p と線分 q0-q1 の距離
def distance_point_segment(p, q0, q1):
    pj = project(q1, p, q0)
    if distance(pj, q0) < distance(q0, q1) and distance(pj, q1) < distance(q0, q1):
        return abs(cross(q1, p, q0) / distance(q0, q1))
    else:
        return min(
            distance(p, q0),
            distance(p, q1),
        )


# 線分 p0-p1 と q0-q1 の距離
def distance_segment(p0, p1, q0, q1):
    if intersect_segment(p0, p1, q0, q1):
        return 0
    return min(
        distance_point_segment(p0, q0, q1),
        distance_point_segment(p1, q0, q1),
        distance_point_segment(q0, p0, p1),
        distance_point_segment(q1, p0, p1),
    )


# 多角形の面積
def area_polygon(ps):
    n = len(ps)
    s = 0
    for i in range(n):
        s += cross(ps[i], ps[(i + 1) % n])
    return abs(s) / 2


# 凸多角形判定
def is_convex_polygon(ps):
    n = len(ps)
    l = []
    for i in range(n):
        l.append(counter_clockwise(ps[i], ps[(i + 1) % n], ps[(i + 2) % n]))
    return min(l) >= 0 or max(l) <= 0


# 多角形の点の内外判定
# 内部:1, 外部:-1, 辺上:0
def is_inside_polygon(ps, q):
    n = len(ps)
    q0 = [q[0] + 99989, q[1] + 99991]
    count = 0
    for i in range(n):
        if on_segment(ps[i], q, ps[(i + 1) % n]):
            return 0
        if intersect_segment(q, q0, ps[i], ps[(i + 1) % n]):
            count += 1

    if count & 1:
        return 1
    else:
        return -1


# 凸包
def convex_hull(ps):
    """凸包

    Args:
        points (float[][]): 点のリスト、[x, y, ...]

    Returns:
        float[][]: 凸包を構成する頂点を下側から順に反時計回りに返す
    """
    n = len(ps)
    ps.sort(key=lambda x: (x[1], x[0]))

    if n <= 2:
        return ps
    elif n == 3:
        if counter_clockwise(ps[0], ps[1], ps[2]) == 1:
            return ps
        else:
            return [ps[0], ps[2], ps[1]]

    right = deque([ps[0][:], ps[1][:]])
    for i in range(2, n):
        next = ps[i][:]
        curr = right.pop()
        prev = right.pop()
        while True:
            if counter_clockwise(prev, curr, next) >= 0:
                right.append(prev)
                right.append(curr)
                break
            if len(right) == 0:
                right.append(prev)
                break
            curr = prev[:]
            prev = right.pop()

        right.append(next)

    left = deque([ps[n - 1][:], ps[n - 2][:]])
    for i in range(n - 2)[::-1]:
        next = ps[i][:]
        curr = left.pop()
        prev = left.pop()
        while True:
            if counter_clockwise(prev, curr, next) >= 0:
                left.append(prev)
                left.append(curr)
                break
            if len(left) == 0:
                left.append(prev)
                break
            curr = prev[:]
            prev = left.pop()

        left.append(next)

    right.pop()
    left.pop()
    return list(right) + list(left)


# 凸多角形の直径
def diameter_convex_polygon(ps):
    ch = convex_hull(ps)
    n = len(ch)
    if n == 2:
        return distance(ch[0], ch[1])
    i = j = 0
    for k in range(n):
        if ch[k][0] < ch[i][0]:
            i = k
        if ch[k][0] > ch[j][0]:
            j = k
    res = 0
    si, sj = i, j
    while i != sj or j != si:
        res = max(res, distance(ch[i], ch[j]))
        if (
            cross(
                [ch[(i + 1) % n][0] - ch[i][0], ch[(i + 1) % n][1] - ch[i][1]],
                [ch[(j + 1) % n][0] - ch[j][0], ch[(j + 1) % n][1] - ch[j][1]],
            )
            < 0
        ):
            i = (i + 1) % n
        else:
            j = (j + 1) % n

    return res


# 三角形の内心
def incenter(p0, p1, p2):
    a = distance(p1, p2)
    b = distance(p2, p0)
    c = distance(p0, p1)
    x = (a * p0[0] + b * p1[0] + c * p2[0]) / (a + b + c)
    y = (a * p0[1] + b * p1[1] + c * p2[1]) / (a + b + c)
    r = area_polygon([p0, p1, p2]) * 2 / (a + b + c)
    return x, y, r


# 三角形の外心
def curcumcenter(p1, p2, p3):
    a = 2 * (p1[0] - p2[0])
    b = 2 * (p1[1] - p2[1])
    p = p1[0] ** 2 - p2[0] ** 2 + p1[1] ** 2 - p2[1] ** 2
    c = 2 * (p1[0] - p3[0])
    d = 2 * (p1[1] - p3[1])
    q = p1[0] ** 2 - p3[0] ** 2 + p1[1] ** 2 - p3[1] ** 2
    det = a * d - b * c
    x = d * p - b * q
    y = a * q - c * p
    if det < 0:
        x = -x
        y = -y
        det = -det
    x /= det
    y /= det
    r = ((x - p1[0]) ** 2 + (y - p1[1]) ** 2) ** 0.5
    return x, y, r


while True:
    n, m = map(int, input().split())
    if n == 0 and m == 0:
        break
    circles = []
    for _ in range(n):
        x, y, r = map(int, input().split())
        circles.append([x, y, r])

    neighbors = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if (
                abs(circles[i][2] - circles[j][2])
                < distance(circles[i][:2], circles[j][:2])
                < circles[i][2] + circles[j][2]
            ):
                neighbors[i].append(j)

    for i in range(n):
        neighbors[i].sort(
            key=lambda x: atan2(
                circles[x][1] - circles[i][1], circles[x][0] - circles[i][0]
            )
        )

    for _ in range(n):
        for i in range(n):
            if len(neighbors[i]) == 1:
                neighbors[neighbors[i][0]].remove(i)
                neighbors[i] = []

    polygons = []
    for i in range(n):
        for j in range(n):
            if neighbors[i].count(j) == 0:
                continue
            l = [i, j]
            while True:
                curr = l.pop()
                prev = l.pop()
                next = neighbors[curr][
                    (neighbors[curr].index(prev) + 1) % len(neighbors[curr])
                ]
                l.append(prev)
                l.append(curr)
                if next == prev:
                    break
                if next == i:
                    polygons.append([circles[p] for p in l])
                    break
                l.append(next)

    answers = []
    for _ in range(m):
        px, py, qx, qy = map(int, input().split())
        # draw.ellipse((px + 900, py + 900, px + 1100, py + 1100), fill=(255, 0, 0))
        # draw.ellipse((qx + 900, qy + 900, qx + 1100, qy + 1100), fill=(255, 0, 0))
        ans = "YES"
        flag = []
        for x, y, r in circles:
            if (distance([px, py], [x, y]) < r) ^ (distance([qx, qy], [x, y]) < r):
                ans = "NO"
            flag.append(distance([px, py], [x, y]) < r)

        if ans == "NO":
            answers.append(ans)
            continue
        if any(flag):
            answers.append(ans)
            continue
        for ps in polygons:
            if is_inside_polygon(ps, [px, py]) * is_inside_polygon(ps, [qx, qy]) == -1:
                ans = "NO"
                break

        answers.append(ans)

    print(*answers)

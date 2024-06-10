# 出力が指数形式にならないよう注意
# print(f"{num:.10f}") などとする
# 適当な大きさの素数 99961, 99971, 99989, 99991


# 内積
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

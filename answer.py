# 出力が指数形式にならないよう注意
# print(f"{num:.10f}") などとする
# 適当な大きさの素数 99961, 99971, 99989, 99991
# matplotlib, pillow などの図形描画ライブラリがあるといいかも
# from PIL import Image, ImageDraw
# im = Image.new('RGB', (500, 300), (255, 255, 255))
# draw = ImageDraw.Draw(im)
# draw.ellipse((100, 100, 200, 200), fill=None, outline=(0, 0, 0))
# draw.rectangle((200, 100, 300, 200), fill=(0, 192, 192), outline=(255, 255, 255))
# draw.line((350, 200, 450, 100), fill=(255, 255, 0), width=10)
# im.save('./pillow_image.jpg', quality=95)


from collections import deque
from math import atan2


def dot(v, w):
    """内積

    Args:
        v (list): ベクトル1
        w (list): ベクトル2

    Returns:
        number: v と w の内積
    """
    return v[0] * w[0] + v[1] * w[1]


def cross(v, w, o=[0, 0]):
    """外積の係数(= vを90度回転させた内積)

    Args:
        v (list): ベクトル1
        w (list): ベクトル2
        o (list, optional): 原点. Defaults to [0, 0].

    Returns:
        number: o を原点とした v と w の外積の係数
    """
    v = [v[0] - o[0], v[1] - o[1]]
    w = [w[0] - o[0], w[1] - o[1]]
    return v[0] * w[1] - v[1] * w[0]


def project(v, w, o=[0, 0]):
    """射影

    Args:
        v (list): 射影先のベクトル
        w (list): 射影元のベクトル
        o (list, optional): 原点. Defaults to [0, 0].

    Returns:
        list: v に対する w の射影
    """
    v = [v[0] - o[0], v[1] - o[1]]
    w = [w[0] - o[0], w[1] - o[1]]
    return [o[0] + v[0] * dot(v, w) / dot(v, v), o[1] + v[1] * dot(v, w) / dot(v, v)]


def reflect(v, w):
    """反射

    Args:
        v (list): 軸となるベクトル
        w (list): 移動する点

    Returns:
        list: v に対する w の反射
    """
    p = project(v, w)
    return [2 * p[0] - w[0], 2 * p[1] - w[1]]


def counter_clockwise(p1, p2, p3):
    """3点 p1, p2, p3 の回転方向判定

    Args:
        p1 (list): 点1
        p2 (list): 点2
        p3 (list): 点3

    Returns:
        number: 反時計回り:1, 時計回り:-1, 直線上:0
    """
    p2 = [p2[0] - p1[0], p2[1] - p1[1]]
    p3 = [p3[0] - p1[0], p3[1] - p1[1]]
    if cross(p2, p3) > 0:
        return 1
    elif cross(p2, p3) < 0:
        return -1
    else:
        return 0


def on_segment(p, q1, q2):
    """線分上の判定

    Args:
        p (list): 点の座標
        q1 (list): 線分の端点1
        q2 (list): 線分の端点2

    Returns:
        bool: p が q1-q2 の線分上にある場合 True
    """
    if counter_clockwise(q1, q2, p) != 0:
        return False
    v = [q2[0] - q1[0], q2[1] - q1[1]]
    w = [p[0] - q1[0], p[1] - q1[1]]
    return 0 <= dot(v, w) <= dot(v, v)


def intersect_segment(p1, p2, q1, q2):
    """線分の交差判定

    Args:
        p1 (list): 線分1の始点
        p2 (list): 線分1の終点
        q1 (list): 線分2の始点
        q2 (list): 線分2の終点

    Returns:
        bool: 線分1と線分2が交差している場合 True
    """
    return (
        counter_clockwise(q1, q2, p1) * counter_clockwise(q1, q2, p2) < 0
        and counter_clockwise(p1, p2, q1) * counter_clockwise(p1, p2, q2) < 0
        or on_segment(q1, p1, p2)
        or on_segment(q2, p1, p2)
        or on_segment(p1, q1, q2)
        or on_segment(p2, q1, q2)
    )


def intersection_point(p1, p2, q1, q2):
    """線分の交点

    Args:
        p1 (list): 線分1の始点
        p2 (list): 線分1の終点
        q1 (list): 線分2の始点
        q2 (list): 線分2の終点

    Returns:
        list: 交点の座標
    """
    t = cross(q1, p2, p1) / cross(
        [p2[0] - p1[0], p2[1] - p1[1]], [q2[0] - q1[0], q2[1] - q1[1]]
    )
    return [q1[0] + t * (q2[0] - q1[0]), q1[1] + t * (q2[1] - q1[1])]


def distance(p, q):
    """２点間の距離

    Args:
        p (list): 点1
        q (list): 点2

    Returns:
        number: 点1 と 点2 の距離
    """
    return ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5


def distance_point_segment(p, q1, q2):
    """点と線分の距離

    Args:
        p (list): 点の座標
        q1 (list): 線分の端点1
        q2 (list): 線分の端点2

    Returns:
        number: 点と線分の距離
    """
    pj = project(q1, p, q2)
    if distance(pj, q1) < distance(q1, q2) and distance(pj, q2) < distance(q1, q2):
        return abs(cross(q2, p, q1) / distance(q1, q2))
    else:
        return min(
            distance(p, q1),
            distance(p, q2),
        )


def distance_point_line(p, q1, q2):
    """点と直線の距離

    Args:
        p (list): 点の座標
        q1 (list): 直線の端点1
        q2 (list): 直線の端点2

    Returns:
        number: 点と直線の距離
    """
    return abs(cross(q2, p, q1) / distance(q1, q2))


def distance_segment(p1, p2, q1, q2):
    """線分と線分の距離

    Args:
        p1 (list): 線分1の始点
        p2 (list): 線分1の終点
        q1 (list): 線分2の始点
        q2 (list): 線分2の終点

    Returns:
        number: 線分1と線分2の距離
    """
    if intersect_segment(p1, p2, q1, q2):
        return 0
    return min(
        distance_point_segment(p1, q1, q2),
        distance_point_segment(p2, q1, q2),
        distance_point_segment(q1, p1, p2),
        distance_point_segment(q2, p1, p2),
    )


def area_polygon(ps):
    """多角形の面積(凹凸、回転方向問わず)

    Args:
        ps (list<list>): 多角形の頂点のリスト

    Returns:
        number: 面積
    """
    n = len(ps)
    s = 0
    for i in range(n):
        s += cross(ps[i], ps[(i + 1) % n])
    return abs(s) / 2


def is_convex_polygon(ps):
    """凸多角形判定

    Args:
        ps (list<list>): 多角形の頂点のリスト

    Returns:
        bool: 凸多角形であれば True
    """
    n = len(ps)
    l = []
    for i in range(n):
        l.append(counter_clockwise(ps[i], ps[(i + 1) % n], ps[(i + 2) % n]))
    return min(l) >= 0 or max(l) <= 0


def is_inside_polygon(ps, q):
    """多角形と点の関係

    Args:
        ps (list<list>): 多角形の頂点のリスト
        q (list): 点の座標

    Returns:
        number: 内部:1, 外部:-1, 辺上:0
    """
    n = len(ps)
    q0 = [q[0] + 99989, q[1] + 99991]
    count = 0
    for i in range(n):
        if on_segment(q, ps[i], ps[(i + 1) % n]):
            return 0
        if intersect_segment(q, q0, ps[i], ps[(i + 1) % n]):
            count += 1

    if count & 1:
        return 1
    else:
        return -1


def convex_hull(ps):
    """凸包

    Args:
        points (list<list>): 多角形の頂点のリスト

    Returns:
        list<list>: 凸包を構成する頂点(下側から順に反時計回り)
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


def diameter_convex_polygon(ps):
    """凸多角形の直径(凸包を求めてから計算)

    Args:
        ps (list<list>): 凸多角形の頂点のリスト

    Returns:
        number: 最も遠い2点間の距離
    """
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


def incenter(p1, p2, p3):
    """三角形の内心

    Args:
        p1 (list): 頂点1
        p2 (list): 頂点2
        p3 (list): 頂点3

    Returns:
        list: 内心のx座標、内心のy座標、内接円の半径を順に返す
    """
    a = distance(p2, p3)
    b = distance(p3, p1)
    c = distance(p1, p2)
    x = (a * p1[0] + b * p2[0] + c * p3[0]) / (a + b + c)
    y = (a * p1[1] + b * p2[1] + c * p3[1]) / (a + b + c)
    r = area_polygon([p1, p2, p3]) * 2 / (a + b + c)
    return x, y, r


def curcumcenter(p1, p2, p3):
    """三角形の外心

    Args:
        p1 (list): 頂点1
        p2 (list): 頂点2
        p3 (list): 頂点3

    Returns:
        list: 外心のx座標、外心のy座標、外接円の半径を順に返す
    """
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


def intersection_circle_line(cx, cy, r, p1, p2):
    """円と直線の交点

    Args:
        cx (number): 円の中心のx座標
        cy (number): 円の中心のy座標
        r (number): 円の半径
        p1 (list): 直線の端点1
        p2 (list): 直線の端点2

    Returns:
        list<list>: 交点の座標のリスト
    """
    d = distance_point_line([cx, cy], p1, p2)
    h = (r**2 - d**2) ** 0.5
    if d > r:
        return []
    n = [0, 0]
    if counter_clockwise(p1, p2, [cx, cy]) >= 0:
        n = [p2[1] - p1[1], -1 * (p2[0] - p1[0])]
    else:
        n = [-1 * (p2[1] - p1[1]), p2[0] - p1[0]]
    n = [n[0] * d / distance(p1, p2), n[1] * d / distance(p1, p2)]
    e = [(p2[0] - p1[0]) * h / distance(p1, p2), (p2[1] - p1[1]) * h / distance(p1, p2)]
    q1 = [cx + n[0] + e[0], cy + n[1] + e[1]]
    q2 = [cx + n[0] - e[0], cy + n[1] - e[1]]
    return q1, q2

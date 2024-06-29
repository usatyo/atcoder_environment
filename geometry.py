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
from math import sqrt
from os import close
from sys import setrecursionlimit

INF = 10**9
setrecursionlimit(10**7)


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


def common_polygon(ps1, ps2):
    """多角形の共通部分(凹凸問わず)

    Args:
        ps1 (list<list>): 多角形1の頂点のリスト
        ps2 (list<list>): 多角形2の頂点のリスト

    Returns:
        list<list<list>>: 共通部分の多角形のリスト
    """
    pass


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
    if n == 1:
        if q == ps[0]:
            return 0
        else:
            return -1
    if n == 2:
        if on_segment(q, ps[0], ps[1]):
            return 0
        else:
            return -1
    q0 = [q[0] + 4735706939, q[1] + 1929495389]
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


def intersection_circle(c1x, c1y, c1r, c2x, c2y, c2r):
    """2つの円の交点

    Args:
        c1x (number): 円1の中心のx座標
        c1y (number): 円1の中心のy座標
        c1r (number): 円1の半径
        c2x (number): 円2の中心のx座標
        c2y (number): 円2の中心のy座標
        c2r (number): 円2の半径

    Returns:
        list<list>: 交点の座標のリスト
    """
    d = distance([c1x, c1y], [c2x, c2y])
    if d > c1r + c2r or d < abs(c1r - c2r):
        return []
    a = (c1r**2 - c2r**2 + d**2) / (2 * d)
    h = (c1r**2 - a**2) ** 0.5
    x0 = c1x + a * (c2x - c1x) / d
    y0 = c1y + a * (c2y - c1y) / d
    p1 = [x0 + h * (c2y - c1y) / d, y0 - h * (c2x - c1x) / d]
    p2 = [x0 - h * (c2y - c1y) / d, y0 + h * (c2x - c1x) / d]
    return p1, p2


def tangent_circle(cx, cy, r, p):
    """円と点の接線

    Args:
        cx (number): 円の中心のx座標
        cy (number): 円の中心のy座標
        r (number): 円の半径
        p (list): 点の座標

    Returns:
        list<list>: 接点のリスト
    """
    d = distance([cx, cy], p)
    if d < r:
        return []
    h = (d**2 - r**2) ** 0.5
    x0 = cx + (p[0] - cx) * r / d
    y0 = cy + (p[1] - cy) * r / d
    n = [p[1] - cy, cx - p[0]]
    n = [n[0] * h / d, n[1] * h / d]
    e = [(p[0] - cx) * h / d, (p[1] - cy) * h / d]
    q1 = [x0 + n[0] + e[0], y0 + n[1] + e[1]]
    q2 = [x0 + n[0] - e[0], y0 + n[1] - e[1]]
    return q1, q2


def tangent_circle(cx, cy, r, p):
    """ある点を通る接線

    Args:
        cx (number): 円の中心のx座標
        cy (number): 円の中心のy座標
        r (number): 円の半径
        p (list): 点の座標

    Returns:
        list<list>: 接点のリスト
    """
    d = distance([cx, cy], p)
    if d < r:
        return []
    h = (d**2 - r**2) ** 0.5
    e = [(p[0] - cx) / d, (p[1] - cy) / d]
    n = [-1 * (p[1] - cy) / d, (p[0] - cx) / d]
    q1 = [
        cx + e[0] * r * r / d + n[0] * r * h / d,
        cy + e[1] * r * r / d + n[1] * r * h / d,
    ]
    q2 = [
        cx + e[0] * r * r / d - n[0] * r * h / d,
        cy + e[1] * r * r / d - n[1] * r * h / d,
    ]
    return q1, q2


# cp_rec - 再帰用関数
# 入力: 配列と区間
# 出力: 距離と区間内の要素をY座標でソートした配列
def cp_rec(ps, i, n):
    if n <= 1:
        return INF, [ps[i]]
    m = n // 2
    x = ps[i + m][0]  # 半分に分割した境界のX座標
    # 配列を半分に分割して計算
    d1, qs1 = cp_rec(ps, i, m)
    d2, qs2 = cp_rec(ps, i + m, n - m)
    d = min(d1, d2)
    # Y座標が小さい順にmergeする
    qs = [None] * n
    s = t = idx = 0
    while s < m and t < n - m:
        if qs1[s][1] < qs2[t][1]:
            qs[idx] = qs1[s]
            s += 1
        else:
            qs[idx] = qs2[t]
            t += 1
        idx += 1
    while s < m:
        qs[idx] = qs1[s]
        s += 1
        idx += 1
    while t < n - m:
        qs[idx] = qs2[t]
        t += 1
        idx += 1
    # 境界のX座標x(=ps[i+m][0])から距離がd以下のものについて距離を計算していく
    # bは境界のX座標から距離d以下のものを集めたもの
    b = []
    for i in range(n):
        ax, ay = q = qs[i]
        if abs(ax - x) >= d:
            continue
        # Y座標について、qs[i]から距離がd以下のj(<i)について計算していく
        for j in range(len(b) - 1, -1, -1):
            dx = ax - b[j][0]
            dy = ay - b[j][1]
            if dy >= d:
                break
            d = min(d, sqrt(dx**2 + dy**2))
        b.append(q)
    return d, qs


def closest_pair(ps):
    """最近点対

    Args:
        ps (list): 点の座標のリスト

    Returns:
        number: 最近点対の距離
    """
    ps.sort()
    n = len(ps)
    return cp_rec(ps, 0, n)[0]

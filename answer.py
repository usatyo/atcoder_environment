from collections import defaultdict, deque

n, m = map(int, input().split())
a = list(map(int, input().split()))

# m <= 2 はほぼ自明に Yes
if m <= 2:
    print("Yes")
    exit()

# 境界線の左に来る数字の個数
left = []
# 隣の境界線としてありえるもののリスト
next_border = [[] for _ in range(1 << n)]

# 境界線関係の処理を前計算
for i in range(1 << n):
    left.append(sum([(i >> j) % 2 * (j + 1) for j in range(n)]))
    for j in range(i + 1, 1 << n):
        if i.bit_count() + 1 != j.bit_count():
            continue

        split = False
        for k in reversed(range(n)):
            if (i >> k).bit_count() == (j >> k).bit_count():
                if split:
                    break
            elif (i >> k).bit_count() > (j >> k).bit_count():
                break
            else:
                split = True
        else:
            next_border[i].append(j)

# Aから可能な分割を前計算
q = deque([a])

for i in range(n - m):
    length = m + i
    while True:
        target = q.popleft()
        if len(target) != length:
            break
        for cut in range(n * (n + 1) // 2):
            for j in range(length):
                if cut == 0:
                    break
                if cut < target[j]:
                    q.append(target[:j] + target[j + 1 :] + [cut, target[j] - cut])
                    break
                else:
                    cut -= target[j]

dic = defaultdict(bool)
while q:
    dic[tuple(sorted(q.pop()))] = True


def check(l):
    return dic[tuple(sorted(l))]


def dfs(prev, curr, l):
    new_l = l + [left[curr] - left[prev]]
    if curr == (1 << n) - 1:
        return check(new_l)
    ret = False
    for next in next_border[curr]:
        ret = ret or dfs(curr, next, new_l)
    return ret


ans = False
for next in next_border[0]:
    ans = ans or dfs(0, next, [])

if ans:
    print("Yes")
else:
    print("No")

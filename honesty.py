from itertools import combinations


n = int(input())
s = input()
t = input()

if s.count("1") < t.count("1"):
    print(-1)
    exit()

if (s.count("1") - t.count("1")) & 1:
    print(-1)
    exit()

si = set()
ti = []

for i in range(n):
    if s[i] == "1":
        si.add(i)
    if t[i] == "1":
        ti.append(i)

ans = float("inf")
for p in combinations(list(si), len(ti)):
    flag = True
    val = 0
    p = list(p)
    p.sort()
    for i in range(len(ti)):
        if ti[i] > p[i]:
            flag = False
        else:
            val += p[i] - ti[i]
    if not flag:
        continue

    rest = si - set(p)
    rest = list(rest)
    rest.sort()
    for i in range(len(rest) // 2):
        val += rest[i * 2 + 1] - rest[i * 2]

    ans = min(ans, val)


if ans == float("inf"):
    print(-1)
else:
    print(ans)

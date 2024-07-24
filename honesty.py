t = int(input())

for _ in range(t):
    n = int(input())
    p = list(map(int, input().split()))
    ans = []

    for i in range(n - 1):
        if p[i] == i + 1:
            continue
        idx = p.index(i + 1)
        if (idx + i) & 1:
            for x in range(idx - 1, i - 1, -1):
                ans.append(x + 1)
        else:
            ans.append(i + 1)
            p[i], p[i + 1] = p[i + 1], p[i]
            for x in range(idx - 1, i - 1, -1):
                ans.append(x + 1)
        p = p[:i] + [i + 1] + p[i:idx] + p[idx + 1 :]

    print(len(ans))
    print(*ans)

    if len(ans) > n**2:
        print("over length")
    for i in range(len(ans)):
        if (ans[i] + i) & 1:
            print("digit rule broken")

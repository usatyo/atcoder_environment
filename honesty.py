t = int(input())
for _ in range(t):
    n = int(input())
    edges = [[] for _ in range(n)]
    for _ in range(n - 1):
        a, b = map(int, input().split())
        a -= 1
        b -= 1
        edges[a].append(b)
        edges[b].append(a)
    s = input()

    ans = [-1] * n
    flag = True
    bridge = []

    for i in range(n):
        if len(edges[i]) >= 2:
            bridge.append(i)
            continue
        node = edges[i][0]
        if s[i] == "B" and ans[node] == 1:
            flag = False
        if s[i] == "W" and ans[node] == 0:
            flag = False
        if s[i] == "B":
            ans[node] = 0
        else:
            ans[node] = 1

    for i in bridge:
        count = 0
        for node in edges[i]:
            if s[i] == "B" and ans[node] == 1:
                count += 1
                continue
            if s[i] == "W" and ans[node] == 0:
                count += 1
                continue
            if s[i] == "B":
                ans[node] = 0
            else:
                ans[node] = 1
        if count > len(edges[i]) // 2:
            flag = False

    if not flag:
        print(-1)
    else:
        print(*["B" if x == 0 else "W" for x in ans], sep="")

n, s = map(int, input().split())
a = list(map(int, input().split()))

dp = [[False] * (s + 1) for _ in range(n + 1)]
dp[0][0] = True

for i in range(n):
    for j in range(s+1):
        if dp[i][j]:
            dp[i + 1][j] = True
            if j + a[i] < s + 1:
                dp[i + 1][j + a[i]] = True

if dp[n][s]:
    print("Yes")
else:
    print("No")

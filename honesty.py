h, w, k = map(int, input().split())
a = list(map(int, input().split()))
b = list(map(int, input().split()))

if (sum(a) - sum(b)) % k != 0:
    print(-1)
    exit()

ans = 0

for bit in range(k ** (h * w)):
    row = [0] * h
    col = [0] * w
    val = 0
    for i in range(h):
        for j in range(w):
            x = bit // (k ** (i * w + j)) % k
            row[i] += x
            col[j] += x
            row[i] %= k
            col[j] %= k
            val += x
    if row == a and col == b:
        ans = max(ans, val)

print(ans)

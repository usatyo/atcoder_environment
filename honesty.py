n, s = map(int, input().split())
a = list(map(int, input().split()))

for i in range(2**s):
    val = 0
    for j in range(s):
        if (i >> j) % 2 == 1:
            val += a[j]
    if val == s:
        print("Yes")
        exit()

print("No")

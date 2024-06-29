from random import randint

f = open("input.txt", "w")
h = 2
w = 2
k = 10
a = [randint(0, k - 1) for _ in range(h)]
b = [randint(0, k - 1) for _ in range(w)]

f.write(f"{h} {w} {k}\n")
f.write(" ".join([str(x) for x in a]) + "\n")
f.write(" ".join([str(x) for x in b]) + "\n")
f.close()

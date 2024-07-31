from random import randint


f = open("input.txt", "w")
n = randint(1, 10)
s = "".join([str(randint(0, 1)) for _ in range(n)])
t = "".join([str(randint(0, 1)) for _ in range(n)])
f.write(f"{n}\n")
f.write(f"{s}\n")
f.write(f"{t}\n")
f.close()

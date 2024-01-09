from random import randint

f = open("input.txt", "w")
f.write("10 10\n")
a = [str(randint(1, 15)) for _ in range(10)]
f.write(" ".join(a) + "\n")
f.close()

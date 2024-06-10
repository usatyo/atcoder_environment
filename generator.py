from random import randint

f = open("input.txt", "w")
s = ["W" if randint(0, 1) == 0 else "B" for _ in range(8)]
f.write(str("1\n"))
f.write(str("8\n"))
f.write(str("7 5\n"))
f.write(str("7 2\n"))
f.write(str("7 4\n"))
f.write(str("7 3\n"))
f.write(str("1 7\n"))
f.write(str("6 7\n"))
f.write(str("8 7\n"))
f.write("".join(s))
f.close()

from random import randint, random, shuffle


def prime(x):
    if x == 1:
        return False
    for i in range(2, int(x**0.5) + 1):
        if x % i == 0:
            return False
    return True


f = open("input.txt", "w")
p = 1
while not prime(p):
    p = randint(2, 10**5)
a = randint(1, p - 1)
b = randint(1, p - 1)
f.write(f"{p} {a} {b}\n")
# f.write(" ".join(str(x) for x in p) + "\n")
f.close()

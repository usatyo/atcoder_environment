from bisect import bisect_left


def LIS(a):
    n = len(a)
    lis = [a[0]]
    for i in range(n):
        if a[i] > lis[-1]:
            lis.append(a[i])
        else:
            lis[bisect_left(lis, a[i])] = a[i]
    return lis

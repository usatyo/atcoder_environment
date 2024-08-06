def factorization(n):
    """n を素因数分解

    Args:
        n (int): 素因数分解の対象

    Returns:
        int[][]: [[素因数, 指数], ...]の2次元リスト
    """
    arr = []
    temp = n
    for i in range(2, int(-(-(n**0.5) // 1)) + 1):
        if temp % i == 0:
            cnt = 0
            while temp % i == 0:
                cnt += 1
                temp //= i
            arr.append([i, cnt])

    if temp != 1:
        arr.append([temp, 1])

    if arr == []:
        arr.append([n, 1])

    return arr

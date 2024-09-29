class RollingHash:
    def __init__(self, s, base=998244353, mod=10**9 + 9):
        """初期化

        Args:
            s (str): 対象の文字列
            base (int, optional): 素数1. Defaults to 998244353.
            mod (int, optional): 素数2. Defaults to 10**9+9.
        """
        self.mod = mod
        self.pw = pw = [1] * (len(s) + 1)

        l = len(s)
        self.h = h = [0] * (l + 1)

        v = 0
        for i in range(l):
            h[i + 1] = v = (v * base + ord(s[i])) % mod
        v = 1
        for i in range(l):
            pw[i + 1] = v = v * base % mod

    def get(self, l, r):
        """s[l:r]のハッシュ値を生成

        Args:
            l (int): 開始位置
            r (int): 終了位置

        Returns:
            _type_: _description_
        """
        return (self.h[r] - self.h[l] * self.pw[r - l]) % self.mod

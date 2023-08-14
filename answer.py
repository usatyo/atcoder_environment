class Solver():
    def __init__(self):
        self.l, self.n, self.s = map(int, input().split())
        self.x, self.y = [],[]
        for _ in range(self.n):
            x,y=map(int, input().split())
            self.x.append(x)
            self.y.append(y)
    
    def place(self):
        


solver = Solver()


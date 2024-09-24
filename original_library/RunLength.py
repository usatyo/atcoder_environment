def runLength(s):
    ret = []
    for i in range(len(s)):
        if ret and ret[-1][0] == s[i]:
            ret[-1][1] += 1
        else:
            ret.append([s[i], 1])

    return ret

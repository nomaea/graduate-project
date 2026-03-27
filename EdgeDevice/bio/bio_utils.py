import time


def clamp01(x: float) -> float:
    return 0.0 if x < 0 else (1.0 if x > 1 else x)


def norm_range(x, lo, hi) -> float:
    if x is None:
        return 0.0
    if hi <= lo:
        return 0.0
    return clamp01((x - lo) / (hi - lo))


def now_iso():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def stdev(xs):
    if xs is None or len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return var ** 0.5


def median(xs):
    if not xs:
        return None
    ys = sorted(xs)
    n = len(ys)
    mid = n // 2
    if n % 2 == 1:
        return ys[mid]
    return 0.5 * (ys[mid - 1] + ys[mid])


def mad(xs):
    if not xs:
        return None
    m = median(xs)
    dev = [abs(x - m) for x in xs]
    return median(dev)

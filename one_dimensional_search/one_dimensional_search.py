import math


def dichotomy(func, left, right, eps=1e-1):
    lefts = []
    rights = []
    delta = eps / 3
    iterations = 0
    while right - left > eps:
        lefts.append(left)
        rights.append(right)
        iterations += 1
        mid = (right + left) / 2
        x1 = mid - delta
        x2 = mid + delta

        f1 = func(x1)
        f2 = func(x2)

        if f1 < f2:
            right = x2
        elif f1 > f2:
            left = x1
        else:
            left = x1
            right = x2
    return (right + left) / 2, iterations, lefts, rights


def golden_ratio(func, left, right, eps=1e-1):
    lefts = [left]
    rights = [right]
    iterations = 1
    phi = (math.sqrt(5) + 1) / 2

    x1 = left + (2 - phi) * (right - left)
    x2 = right - (2 - phi) * (right - left)

    f1 = func(x1)
    f2 = func(x2)
    while right - left > eps:
        iterations += 1
        if f1 < f2:
            right = x2
            x2 = x1
            # не нужно заново считать f2
            f2 = f1
            x1 = left + (2 - phi) * (right - left)
            f1 = func(x1)
        elif f1 > f2:
            left = x1
            x1 = x2
            f1 = f2
            x2 = right - (2 - phi) * (right - left)
            f2 = func(x2)
        else:
            left = x1
            right = x2
        lefts.append(left)
        rights.append(right)
    return (left + right) / 2, iterations, lefts, rights


def fibonacci(func, left, right, eps=1e-1):
    lefts = [left]
    rights = [right]
    iterations = 1
    fib = [1, 1, 2]
    low = (right - left) / eps

    def add_fib():
        fib.append(fib[-1] + fib[-2])

    n = 0
    while low >= fib[n + 2]:
        n += 1
        add_fib()
    x1 = left + (fib[n - 2] / fib[n]) * (right - left)
    x2 = right - (fib[n - 1] / fib[n]) * (right - left)

    f1 = func(x1)
    f2 = func(x2)

    while n > 0:
        n -= 1
        iterations += 1
        if f1 < f2:
            right = x2
            x2 = x1
            f2 = f1
            x1 = left + (fib[n - 2] / fib[n]) * (right - left)
            f1 = func(x1)
        elif f1 > f2:
            left = x1
            x1 = x2
            f1 = f2
            x2 = right - (fib[n - 1] / fib[n]) * (right - left)
            f2 = func(x2)
        else:
            left = x1
            right = x2
        lefts.append(left)
        rights.append(right)
    return (left + right) / 2, iterations, lefts, rights


def line_search(func, left, eps=1e-3):
    delta0 = 0.01
    f0 = func(left)
    right = left + delta0
    delta = delta0
    while func(right) <= f0 + eps:
        delta *= 2
        right += delta
    return right




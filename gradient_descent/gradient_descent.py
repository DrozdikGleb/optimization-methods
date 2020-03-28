import sys
sys.path.insert(1, '../one_dimensional_search')
from one_dimensional_search import line_search


def gradient_descent(func, func_grad, x, step_search_method, const_step=False, step=1e-5, eps=1e-3):

    y = func(x)
    step_number = 0
    while True:
        step_number += 1
        grad = func_grad(x)
        if not const_step:
            step = get_step(func, x, grad, step_search_method)
        next_x = x - step * grad
        next_y = func(next_x)
        if abs(next_y - y) < eps:
            return step_number
        x = next_x
        y = next_y


def get_step(function, x, grad, search_method):
    def optimization_problem(alpha):
        return function(x - alpha * grad)
    right_border = line_search(optimization_problem, 0)
    res, _, _, _ = search_method(optimization_problem, 0, right_border)
    return res

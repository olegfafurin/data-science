import math

n, m = map(int, input().split())
X_train = [[0 for j in range(m)] for i in range(n)]
y_train = [0 for i in range(n)]

for i in range(n):
    line = list(map(int, input().split()))
    X_train[i] = line[:-1]
    y_train[i] = line[-1]
request = list(map(int, input().split()))

kernel_dict = {"uniform": lambda x: 1 / 2 if -1 < x < 1 else 0,
               "triangular": lambda x: max(1 - abs(x), 0),
               "epanechnikov": lambda x: (3 / 4) * (1 - x ** 2) if -1 < x < 1 else 0,
               "quartic": lambda x: (15 / 16) * (1 - x ** 2) ** 2 if -1 < x < 1 else 0,
               "triweight": lambda x: (35 / 32) * (1 - x ** 2) ** 3 if -1 < x < 1 else 0,
               "tricube": lambda x: (70 / 81) * (1 - abs(x) ** 3) ** 3 if -1 < x < 1 else 0,
               "gaussian": lambda x: math.exp(- x ** 2 / 2) / math.sqrt(2 * math.pi),
               "cosine": lambda x: (math.pi / 4) * math.cos(math.pi * x / 2) if -1 < x < 1 else 0,
               "logistic": lambda x: 1 / (2 + math.exp(x) + math.exp(-x)),
               "sigmoid": lambda x: 2 / (math.pi * (math.exp(x) + math.exp(-x)))}

dist_dict = {"euclidean": lambda u, v: math.sqrt(sum([(u[i] - v[i]) ** 2 for i in range(len(u))])),
             "chebyshev": lambda u, v: max([abs(u[i] - v[i]) for i in range(len(u))]),
             "manhattan": lambda u, v: sum([abs(u[i] - v[i]) for i in range(len(u))])}


distance = dist_dict[input()]
kernel = kernel_dict[input()]
fixed_k = False if input() == "variable" else True
param = int(input())

order = sorted(list(range(n)), key=lambda i: distance(request, X_train[i]))
cum_res = 0
cum_weight = 0
if not fixed_k:
    norm_dist = distance(request, X_train[order[param]])
    if norm_dist != 0:
        for i in range(n):
            cum_res += y_train[order[i]] * kernel(distance(request, X_train[order[i]]) / norm_dist)
            cum_weight += kernel(distance(request, X_train[order[i]]) / norm_dist)
    else:
        cum_weight = X_train.count(request)
        cum_res = sum([y_train[i] if X_train[i] == request else 0 for i in range(m)])
    if cum_weight == 0:
        cum_weight = sum([distance(request, X_train[i]) == norm_dist for i in range(n)])
        cum_res = sum([y_train[i] if distance(request, X_train[i]) == norm_dist else 0 for i in range(n)])
        result = cum_res / cum_weight
    else:
        result = cum_res / cum_weight
    print("%0.10f" % result)
    exit(0)
else:
    if param == 0:
        cum_weight = X_train.count(request)
        cum_res = sum(y_train[i] if X_train[i] == request else 0 for i in range(m))
    else:
        for i in range(n):
            d = distance(request, X_train[order[i]])
            cum_res += y_train[order[i]] * kernel(d / param)
            cum_weight += kernel(d / param)
    if cum_weight == 0:
        result = sum(y_train) / n
    else:
        result = cum_res / cum_weight
    print("%0.10f" % result)
    exit(0)

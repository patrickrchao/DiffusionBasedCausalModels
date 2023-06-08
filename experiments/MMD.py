from sklearn.metrics import euclidean_distances
from sklearn.metrics.pairwise import rbf_kernel
import numpy as np

def mmd(source, generated, prec = None):
    X = source
    Y = generated
    if prec is None:
        prec = _median_precision(X,Y)
    
    XX = rbf_kernel(X, X, prec).mean()
    YY = rbf_kernel(Y, Y, prec).mean()
    XY = rbf_kernel(X, Y, prec).mean()
    
    dist = XX + YY - 2 * XY
    return dist
    
# _median_bw code modified from
# https://github.com/py-why/dowhy/blob/ead8d47102f0ac6db51d84432874c331fb84f3cb/dowhy/gcm/independence_test/kernel_operation.py#L127
# MIT License

# Copyright (c) Microsoft Corporation. All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE

def _median_precision(X, Y) -> float:
    tmp = euclidean_distances(X, Y, squared=True)
    tmp = tmp - np.tril(tmp, -1)
    tmp = tmp.reshape(-1, 1)
    return 1 / np.median(tmp[tmp > 0])

# https://github.com/psanch21/VACA/blob/a14b9b93726647907c07c44860f7c6bc85b31f88/utils/metrics/mmd.py

# MIT License

# Copyright (c) 2021

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import torch
def mmd_vaca(source, generated, kernel_mul = 2.0, kernel_num = 5):
    batch_size = int(source.size()[0])
    kernels = gaussian_kernel(source, generated, kernel_mul, kernel_num)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss
    
def gaussian_kernel(source, generated, kernel_mul, kernel_num):
    n_samples = int(source.size()[0]) + int(generated.size()[0])

    total = torch.cat([source, generated], dim=0)

    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)
    
    bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)  #
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)
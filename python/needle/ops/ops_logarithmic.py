from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_Z = array_api.max(Z, axis=1, keepdims=True)
        exp_shift = array_api.exp(Z - max_Z)
        sum_exp = exp_shift.sum(axis=1, keepdims=True) 
        return Z - max_Z - array_api.log(sum_exp)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        shape = list(out_grad.shape)
        shape[1] = 1
        return out_grad - exp(node) * out_grad.sum(axes=(1,)).reshape(shape)
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_Z = array_api.max(Z, axis=self.axes, keepdims=True)
        exp_shift = array_api.exp(Z - max_Z)
        sum_exp = exp_shift.sum(axis=self.axes, keepdims=True)
        result = array_api.log(sum_exp) + max_Z
        return result.squeeze(axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        if self.axes:
            shape =list(Z.shape)
            for axis in self.axes:
                shape[axis] = 1
            node_new = node.reshape(shape)
            grad_new = out_grad.reshape(shape)
            return grad_new * exp(Z - node_new)
        return out_grad * exp(Z - node)
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)


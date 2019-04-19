import tensorflow as tf
import numpy as np
from .initializations import *
from .capsules import safe_norm


def h_lambda(c, x):
    return 2. / (1. - c * tf.square(safe_norm(x, keepdims = True)))


def mobius_add(c, x, y):
    # print(y.shape)
    # print(tf.reduce_sum(x * y, axis = -1,keepdims = True).shape

    # )
    # print(x.shape, y.shape)
    numerator = (
        (1. + 2. * c * tf.reduce_sum(x * y, axis = -1, keepdims = True) +
         c * tf.square(safe_norm(y, keepdims = True))) * x +
        (1. - c * tf.square(safe_norm(x, keepdims = True))) * y
    )
    # print(numerator.shape)
    denominator = (
        1. + 2. * c * tf.reduce_sum(x * y, axis = -1, keepdims = True) +
        c ** 2. * tf.square(safe_norm(x, keepdims = True)) * tf.square(safe_norm(y, keepdims = True))
    )

    output = numerator / denominator
    
    # print(output.shape)

    return output


def mobius_add_batch(c, x, y):
    xy = tf.einsum('ij,kj->ik', x, y)
    x2 = tf.reduce_sum(tf.square(x), axis = -1, keepdims = True)
    y2 = tf.reduce_sum(tf.square(y), axis = -1, keepdims = True)
    num = (1. + 2. * c * xy + c * tf.transpose(y2))
    num = tf.expand_dims(num, axis = 2) * tf.expand_dims(x, axis = 1)
    num = num + tf.expand_dims(1. - c * x2, axis = 2) * y
    denom_part1 = 1. + 2. * c * xy
    denom_part2 = tf.square(c) * x2 * tf.transpose(y2)
    denom = denom_part1 + denom_part2

    res = num / (tf.expand_dims(denom, axis = 2) + 1e-5)

    return res


def h_log(c, x, input):

    xpy = mobius_add(c, -x, input)
    # print('xpy')
    # print(xpy.shape)
    # output = tf.identity(xpy, name = 'output')
    # print(c)
    # output = tf.identity(xpy, name = 'output')

    xpy_norm = safe_norm(xpy, axis = -1, keepdims = True)
    # output = tf.identity(xpy_norm, name = 'output')
    
    # print('clip')
    # print(xpy_norm.shape)
    # xpy_norm = tf.clip_by_value(
    #     xpy_norm,
    #     clip_value_min = -1. / np.sqrt(c) * (1. - 1e-3),
    #     clip_value_max = 1. / np.sqrt(c) * (1. - 1e-3)
    # )
    # print(xpy_norm.shape)

    output = (
        2. / (np.sqrt(c) * h_lambda(c, x)) *
        tf.atanh(np.sqrt(c) * xpy_norm ) *
        xpy / xpy_norm
    )
    # output = tf.atanh(np.sqrt(c) * xpy_norm )
    # print('norm')
    # print(xpy.shape)
    # print(safe_norm(xpy).shape)

    return output


def h_exp(c, x, input):

    tanh = (
        tf.tanh(np.sqrt(c) * h_lambda(c, x) * safe_norm(input, keepdims = True) / 2.) *
        input / (np.sqrt(c) * safe_norm(input, keepdims = True))
    )
    
    output = mobius_add(c, x, tanh)


    return output


def mobius(f, c, input):
    return h_exp(
        c, 0.,
        f(
            h_log(c, 0., input)
        )
    )


def h_matmul(c, M, x):
    Mx = tf.matmul(M, x)

    output = (
        1. / np.sqrt(c) *
        tf.tanh(safe_norm(Mx) / safe_norm(x) * tf.atanh(np.sqrt(c) * safe_norm(x))) *
        Mx / safe_norm(Mx)
    )

    return output


def h_linear(input, **params):
    n_inputs = input.shape[-1].value
    # # n_outputs = params['units']
    n_outputs = 10
    c = params['c']

    kernel = tf.Variable(
        initial_value = default_initialization(shape = [n_inputs, n_outputs]),
        name = 'kernel',
        dtype = tf.float32
    )
    # f = lambda x: h_matmul(c, x, kernel)

    # output = mobius(f, params['c'], input)

    output = h_exp(c, 0., h_log(c, 0., input))


    # print(output.shape)

    return tf.matmul(input, kernel)

    # return output


def project(c, x):
    norm = safe_norm(x, keepdims = True)
    norm = tf.clip_by_value(
        norm,
        clip_value_min = 1e-5,
        clip_value_max = 1000
    )
    maxnorm = (1 - 1e-3) / np.sqrt(c)
    cond = tf.reshape(tf.greater(norm, maxnorm), shape = [-1])
    projected = x / norm * maxnorm

    output = tf.where(cond, projected, x)

    return output


def h_exp0(c, u):
    u_norm = safe_norm(u, axis = -1, keepdims = True)
    u_norm = tf.clip_by_value(
        u_norm,
        clip_value_min = 1e-5,
        clip_value_max = 1000
    )
    gamma_1 = tf.tanh(np.sqrt(c) * u_norm) * u / (np.sqrt(c) * u_norm)

    return gamma_1



def to_poincare(input, **params):
    c = params['c']

    output = project(c, h_exp0(c, input))

    return input


def arsinh(x):
    # x = tf.clip_by_value(
    #     x,
    #     clip_value_min = -1 + 1e-5,
    #     clip_value_max = 1 - 1e-5
    # )
    
    # output = .5 * (tf.log(1. + x) - tf.log(1. - x))

    output = x + tf.sqrt(1. + tf.square(x))
    output = tf.clip_by_value(
        output,
        clip_value_min = 1e-5,
        clip_value_max = 1000.
    )
    output = tf.log(output)

    return output


def h_softmax(input, a, p, c):

    lambda_pkc = 2. / (1. - c * tf.reduce_sum(tf.square(p), axis = 1))
    k = lambda_pkc * safe_norm(a, axis = 1) / np.sqrt(c)
    mob_add = mobius_add_batch(c, -p, input)
    num = (
        2. * np.sqrt(c) *
        tf.reduce_sum(mob_add * tf.expand_dims(a, axis = 1), axis = -1)
    )
    denom = (
        safe_norm(a, axis = 1, keepdims = True) *
        (1. - c * tf.reduce_sum(tf.square(mob_add), axis = 2))
    )
    logit = tf.expand_dims(k, axis = 1) * arsinh(num / denom)
    logit = tf.transpose(logit)

    return logit


def h_mlr(input, **params):
    c = params['c']
    n_classes = params['n_classes']
    ball_dim = params['ball_dim']

    a = tf.Variable(
        initial_value = default_initialization(shape = [n_classes, ball_dim]),
        dtype = tf.float32
    )
    p = tf.Variable(
        initial_value = default_initialization(shape = [n_classes, ball_dim]),
        dtype = tf.float32
    )

    p = h_exp0(c, p)
    conformal_factor = (
        1. - c *
        tf.reduce_sum(
            tf.square(p), axis = 1, keepdims = True
        )
    )
    a = a * conformal_factor
    logits = h_softmax(input, a, p, c)

    output = tf.identity(logits, name = 'output')

    return logits




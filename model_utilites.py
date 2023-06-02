from math import sqrt


def get_model_size(model):  # возвращает количество весов в модели
    return sum(i.size for i in model.get_weights())


def get_recommended_learning_rate(model, time_for_learning, time_for_step):
    # пусть в сети n весов. Предположим, что все они меняются от -1 до 1. Тогда максимальный путь,
    # который проделывают веса в пространстве весов - 2*2*n в корне, то есть 2 * корень из n.
    # Если на обучение дается T времени, а каждый шаг занимает t, то lr*2n^0.5 = T/t. lr = T/(2*t*n^0.5)
    model_size = get_model_size(model)
    return time_for_learning / 2 / time_for_step / sqrt(model_size)


def get_model_weights_module(weights):
    square_sum = 0.0
    for i in weights:
        square_sum += sum((i * i).flatten())
    return sqrt(square_sum)


def get_model_weights_sum(weights1, weights2):
    return [weights2[i] + weights1[i] for i in range(len(weights1))]


def get_model_weights_difference(weights1, weights2):
    return [weights2[i] - weights1[i] for i in range(len(weights1))]


def get_model_weights_scalar_mul(weights1, weights2):
    return sum(sum((weights2[i] * weights1[i]).flatten()) for i in range(len(weights1)))


def get_model_weights_cos_angle(weights1, weights2):
    return get_model_weights_scalar_mul(weights1, weights2) / get_model_weights_module(
        weights1) / get_model_weights_module(weights2)


def get_all_shapes(layer):
    return [i.shape for i in layer.get_weights()]

# from keras.activations import relu
# import tensorflow as tf
# закомментил, чтобы не импортировалось лишний раз

# @tf.function
# def relu_with_max(x):
#    return relu(x, max_value=1.0)
#    pass

from keras.constraints import MaxNorm,MinMaxNorm
from keras import backend
import tensorflow as tf


class MaxNormL1(MaxNorm):  # ограничение по норме 1 порядка, а не по 2
    def __call__(self, w):
        norms = tf.reduce_sum(tf.abs(w), axis=self.axis, keepdims=True)
        desired = backend.clip(norms, 0, self.max_value)
        return w * (desired / (backend.epsilon() + norms))


class MinMaxNormL1(MinMaxNorm):  # ограничение по норме 1 порядка, а не по 2
    def __call__(self, w):
        norms = tf.reduce_sum(tf.abs(w), axis=self.axis, keepdims=True)
        desired = (
                self.rate * backend.clip(norms, self.min_value, self.max_value)
                + (1 - self.rate) * norms
        )
        return w * (desired / (backend.epsilon() + norms))

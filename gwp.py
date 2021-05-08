import numpy as np
import tensorflow as tf
from gpflow import models, likelihoods, kernels

class InvWishartProcessLikelihood(likelihoods.Likelihood):
    def __init__(self, D, R=1):
        super().__init__()
        self.R, self.D = R, D
        self.A_diag = np.ones(D)

    def _variational_expectations(self, Fmu, Fvar, Y):
        N, D = tf.shape(Y)
        W = tf.random.normal([self.R, N, tf.shape(Fmu)[1]])
        F = W * (Fvar ** 0.5) + Fmu
        AF = self.A_diag[:, None] * tf.reshape(F, [self.R, N, D, -1])
        yffy = tf.reduce_sum(tf.einsum('jk,ijkl->ijl', Y, AF) ** 2.0, axis=-1)
        chols = tf.linalg.cholesky(tf.matmul(AF, AF, transpose_b=True))
        logp = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chols)), axis=2) - 0.5 * yffy
        return tf.reduce_mean(logp, axis=0)

if __name__ == '__main__':
    test = InvWishartProcessLikelihood(1)

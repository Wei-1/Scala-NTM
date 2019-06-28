package com.scalaml.ntm

import org.apache.mxnet._

object MxNetMath {
  val machineEpsilon: Double = 2.2e-16
  val machineEpsilonSqrt: Double = 1e-8 // Math.sqrt(machineEpsilon)

  // Sigmoid computes 1 / (1 + Math.exp(-x))
  def Sigmoid(x: NDArray): NDArray =
    NDArray.power(NDArray.exp(-x) + 1, -1)

  def delta(a: Int, b: Int): Int =
    if(a == b) 1 else 0

  def cosineSimilarity(u: NDArray, v: NDArray): NDArray = {
    // sum: NDArray, usum: NDArray, vsum: NDArray
    NDArray.sum(u * v) / NDArray.sqrt(NDArray.sum(u * u) * NDArray.sum(v * v))
  }

  def makeTensor2(n: Int , m: Int): NDArray = NDArray.zeros(n, m)

  // Sprint2 pretty prints a 2 dimensional tensor.
  def Sprint2(t: Array[Array[Float]]): String =
    t.map(_.mkString("[", " ", "]")).mkString("[", "", "]")

  // Gemv(t Trans, alpha f64, A General, x Vec, beta f64, y Vec)
  // y = alpha * A * x + beta * y; if t == blas.NoTrans
  def Gemv(trans: Boolean, alpha: Float, M2: NDArray,
    x: NDArray, beta: Float, y: NDArray): Unit = {
    // y = alpha * A * x + beta * y; if t == blas.NoTrans
    y *= beta
    y += NDArray.dot( { if(trans) NDArray.transpose(M2) else M2 } , x) * alpha
  }

  def Ger(alpha: Float, x: NDArray, y: NDArray, A: NDArray): Unit = {
    // A += alpha * x * y^T
    A += NDArray.dot(x, NDArray.transpose(y)) * alpha
  }
}

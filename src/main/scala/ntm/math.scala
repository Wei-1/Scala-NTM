package com.scalaml.ntm

import org.apache.mxnet._

object math {
  val machineEpsilon: Float = 2.2e-16
  val machineEpsilonSqrt: Float = 1e-8 // Math.sqrt(machineEpsilon)

  // Sigmoid computes 1 / (1 + Math.exp(-x))
  def Sigmoid(x: Float): Float =
    1.0 / (1.0 + Math.exp(-x))

  def delta(a: Int, b: Int): Float =
    if(a == b) 1.0 else 0.0

  def cosineSimilarity(u: NDArray, v: NDArray): Float = {
    var sum: Float = (u * v).sum
    var usum: Float = (u * u).sum
    var vsum: Float = (v * v).sum
    sum / Math.sqrt(usum * vsum)
  }

  def makeTensor2(n: Int , m: Int): NDArray = NDArray.zeros(n, m)

  // Sprint2 pretty prints a 2 dimensional tensor.
  def Sprint2(t: Array[Array[Float]]): String =
    t.map(_.mkString("[", " ", "]")).mkString("[", "", "]")

  // Gemv(t Trans, alpha f64, A General, x Vec, beta f64, y Vec)
  // y = alpha * A * x + beta * y; if t == blas.NoTrans
  def Gemv(trans: Boolean, alpha: Float, M2: Array[Array[Float]],
    x: Array[Float], beta: Float, y: Array[Float]): Unit = {
    // y = alpha * A * x + beta * y; if t == blas.NoTrans
    val n = x.size
    val m = y.size
    for(i <- 0 until n; j <- 0 until m)
      y(j) = alpha * (if(trans) M2(i)(j) else M2(j)(i)) * x(i) + beta * y(j)
  }

  def Ger(alpha: Double, x: Array[Double], y: Array[Double], A: Array[Array[Double]]): Unit = {
    // A += alpha * x * y^T
    val n = x.size
    val m = y.size
    for(i <- 0 until m; j <- 0 until n)
      A(j)(i) += alpha * x(j) * y(i)
  }
}

package com.scalaml.ntm

object math {
  val machineEpsilon = 2.2e-16
  val machineEpsilonSqrt = 1e-8 // Math.sqrt(machineEpsilon)

  // Sigmoid computes 1 / (1 + Math.exp(-x))
  def Sigmoid(x: Double): Double =
    1.0 / (1.0 + Math.exp(-x))

  def delta(a: Int, b: Int): Double =
    if(a == b) 1.0 else 0.0

  def cosineSimilarity(u: Array[Double], v: Array[Double]): Double = {
    var sum: Double = 0
    var usum: Double = 0
    var vsum: Double = 0
    for(i <- 0 until u.size) {
      sum += u(i) * v(i)
      usum += u(i) * u(i)
      vsum += v(i) * v(i)
    }
    sum / Math.sqrt(usum * vsum)
  }

  def makeTensor2(n: Int , m: Int): Array[Array[Double]] =
    Array.ofDim[Double](n, m)

  // Sprint2 pretty prints a 2 dimensional tensor.
  def Sprint2(t: Array[Array[Double]]): String =
    t.map(_.mkString("[", " ", "]")).mkString("[", "", "]")

  // Gemv(t Trans, alpha f64, A General, x Vec, beta f64, y Vec)
  // y = alpha * A * x + beta * y; if t == blas.NoTrans
  def Gemv(trans: Boolean, alpha: Double, M2: Array[Array[Double]],
    x: Array[Double], beta: Double, y: Array[Double]): Unit = {
    // y = alpha * A * x + beta * y; if t == blas.NoTrans
    val n = x.size
    val m = y.size
    val z = y.map(_ * beta)
    for(i <- 0 until n; j <- 0 until m)
      z(j) += alpha * (if(trans) M2(i)(j) else M2(j)(i)) * x(i)
    for(i <- 0 until m) y(i) = z(i)
  }

  def Ger(alpha: Double, x: Array[Double], y: Array[Double], A: Array[Array[Double]]): Unit = {
    // A += alpha * x * y^T
    val n = x.size
    val m = y.size
    for(i <- 0 until m; j <- 0 until n)
      A(j)(i) += alpha * x(j) * y(i)
  }
}

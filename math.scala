package ntm

object math {
  val machineEpsilon     = 2.2e-16
  val machineEpsilonSqrt = 1e-8 // math.Sqrt(machineEpsilon)

  // Sigmoid computes 1 / (1 + math.Exp(-x))
  def Sigmoid(x: Double): Double = {
    1.0 / (1 + Math.exp(-x))
  }

  def delta(a: Int, b: Int): Double = {
    if(a == b) 1.0
    else 0.0
  }

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

  def makeTensor2(n: Int , m: Int): Array[Array[Double]] = {
    Array.ofDim[Double](n, m)
  }

  // Sprint2 pretty prints a 2 dimensional tensor.
  def Sprint2(t: Array[Array[Double]]): String = {
    t.map(_.mkString("[", " ", "]")).mkString("[", "", "]")
  }
}

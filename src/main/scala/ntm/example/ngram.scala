package ntm.example

object ngram {
  // GenProb generates a probability lookup table for a n-gram model.
  def GenProb(): Array[Double] = {
    val n = 5
    val probs = new Array[Double](1 << n)
    for(i <- 0 until probs.size) {
      probs(i) = beta()
    }
    probs
  }

  def GenSeq(prob: Array[Double]):
    (Array[Array[Double]], Array[Array[Double]]) = {
    val n = (Math.log(prob.size) / Math.log(2)).toInt
    val seqLen = 200

    val input = new Array[Array[Double]](seqLen + 1)
    for(i <- 0 until n) {
      input(i) = Array(Math.random * 2)
    }
    for(i <- n until input.size) {
      val idx = Binarize(input.drop(i - n).take(n))
      if(Math.random < prob(idx)) input(i)(0) = 1.0
      else input(i)(0) = 0.0
    }

    val output = new Array[Array[Double]](seqLen)
    for(i <- 0 until n-1) {
      output(i) = Array(0.0)
    }
    // copy(output[n-1:], input[n:])
    for(i <- n-1 until output.size) {
      output(i) = input(i + 1)
    }
    (input.take(seqLen), output)
  }

  def Binarize(seq: Array[Array[Double]]): Int = {
    var idx = 0
    for(i <- 0 until seq.size) {
      val a = seq(i)
      idx += (a(0) * (1 << i)).toInt
    }
    idx
  }

  // beta generates a random number from the Beta(1/2, 1/2) distribution.
  def beta(): Double = {
    val x = gamma()
    val y = gamma()
    x / (x + y)
  }

  // gamma generates a random number from the Gamma(1/2, 1) distribution.
  def gamma(): Double = {
    val n = Math.pow(Math.random * 4 - 2, 3)
    0.5 * n * n
  }

}

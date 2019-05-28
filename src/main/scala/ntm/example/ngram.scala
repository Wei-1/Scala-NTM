package com.scalaml.ntm.example
import com.scalaml.ntm

object ngram {
  // GenProb generates a probability lookup table for a n-gram model.
  def GenProb(): Array[Double] = {
    val n = 3
    val probs = new Array[Double](1 << n)
    for(i <- 0 until probs.size) {
      probs(i) = beta()
    }
    probs
  }

  def GenSeq(prob: Array[Double]):
    (Array[Array[Double]], Array[Array[Double]]) = {
    val n = (Math.log(prob.size) / Math.log(2)).toInt
    val seqLen = 100

    val input = new Array[Array[Double]](seqLen + 1)
    for(i <- 0 until n) {
      input(i) = Array(Math.random * 2)
    }
    for(i <- n until input.size) {
      val idx = Binarize(input.drop(i - n).take(n))
      if(Math.random < prob(idx % prob.size)) input(i) = Array(1.0)
      else input(i) = Array(0.0)
    }

    val output = new Array[Array[Double]](seqLen)
    for(i <- 0 until n-1) {
      output(i) = Array(0.0)
    }
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

  def main(args: Array[String]) {

    val h1Size = 12
    val numHeads = 1
    val n = 20
    val m = 8
    val c = ntm.Controller.NewEmptyController(1, 1, h1Size, numHeads, n, m)
    val weights = c.WeightsValVec()
    for(i <- 0 until weights.size; j <- 0 until weights(i).size)
      weights(i)(j) = 1 * (Math.random - 0.5)

    var doPrint = false

    val rmsp = ntm.RMSProp.NewRMSProp(c)
    println("Training -")
    println(s"numweights: ${c.WeightsValVec().size}")
    var model: ntm.LogisticModel = null
    for(i <- 0 to 5000) {
      val (ox, oy) = GenSeq(GenProb())
      model = new ntm.LogisticModel(Y = oy)
      rmsp.Train(ox, model, 0.95, 0.5, 1e-3, 1e-3)
      if(i % 1000 == 0) {
        val prob = GenProb()
        var l = 0.0
        var samn = 100
        for(j <- 0 until samn) {
          val (x, y) = GenSeq(prob)
          model = new ntm.LogisticModel(Y = y)
          val machines = ntm.NTM.ForwardBackward(c, x, model)
          l += model.Loss(ntm.NTM.Predictions(machines))
        }
        l /= samn
        val today = java.util.Calendar.getInstance().getTime()
        println(s"$today | i:$i, loss:$l, seq_length:${oy.size}")
      }
    }

    println("Predicting -")
    val prob = GenProb()
    var samn = 100
    var l = 0.0
    for(j <- 0 until samn) {
      val (x, y) = GenSeq(prob)
      model = new ntm.LogisticModel(Y = y)
      val machines = ntm.NTM.ForwardBackward(c, x, model)
      l += model.Loss(ntm.NTM.Predictions(machines))
      if((j + 1) % 10 == 0) {
        val today = java.util.Calendar.getInstance().getTime()
        println(s"$today | i:$j, loss:$l, seq_length:${y.size}")
      }
    }

    println("Print Weights -")
    val finalWeights = c.WeightsValVec()
    finalWeights.foreach(arr => println(arr.mkString(",")))
  }

}

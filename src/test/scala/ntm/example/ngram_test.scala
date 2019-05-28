package com.scalaml.ntm.example
import com.scalaml.ntm

object ngram_test {
  def TestRun(t: ntm.T) {
    val h1Size = 12
    val numHeads = 1
    val n = 20
    val m = 8
    var runCount = 0
  	try {
      val prob = ngram.GenProb()
      val c = ntm.Controller.NewEmptyController(1, 1, h1Size, numHeads, n, m)
      val weights = c.WeightsValVec()
      for(i <- 0 until weights.size; j <- 0 until weights(i).size)
        weights(i)(j) = 1 * (Math.random - 0.5)
      var samn = 40
      var l = 0.0
      for(j <- 0 until samn) {
        val (x, y) = ngram.GenSeq(prob)
        val model = new ntm.LogisticModel(Y = y)
        val machines = ntm.NTM.ForwardBackward(c, x, model)
        l += model.Loss(ntm.NTM.Predictions(machines))
        if((j + 1) % 10 == 0) t.Logf(s"[EXAMPLE - NGRAM] j: $j, loss: $l")
        runCount
      }
    } catch { case e: Exception =>
      t.Fatalf(s"[EXAMPLE - NGRAM] fatal after run with exception message: $e")
    }
  }
}
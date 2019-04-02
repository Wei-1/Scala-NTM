package ntm.example

object copytask_test {
  def TestRun(t: ntm.T) {
    val vectorSize = 3 // down-scaling for faster predicting
    val h1Size = 12 // down-scaling for faster predicting
    val numHeads = 1
    val n = 20 // down-scaling for faster predicting
    val m = 8 // down-scaling for faster predicting
    val seqLens = Array(2, 4, 6, 10, 16)
    var runs = Array[Run]()
  	try {
      val c = ntm.Controller.NewEmptyController(vectorSize + 2, vectorSize, h1Size, numHeads, n, m)
      val weights = c.WeightsValVec()
      for(i <- 0 until weights.size; j <- 0 until weights(i).size)
        weights(i)(j) = 1 * (Math.random - 0.5)

      for(seql <- seqLens) {
        val (x, y) = copytask.GenSeq(seql, vectorSize)
        val model = new ntm.LogisticModel(Y = y)
        val machines = ntm.NTM.ForwardBackward(c, x, model)
        val predicts = ntm.NTM.Predictions(machines)
        val hWeights = ntm.NTM.HeadWeights(machines)
        val l = model.Loss(predicts)
        val bps = l / (y.size * y.head.size)
        t.Logf(s"[EXAMPLE - COPYTASK] sequence length: $seql, loss: $bps")

        val r = new Run(
          SeqLen = seql,
          BitsPerSeq = bps,
          X = x,
          Y = y,
          Predictions = predicts,
          HeadWeights = hWeights
        )
        runs :+= r
      }
    } catch { case e: Exception =>
      if(runs.size < seqLens.size)
        t.Fatalf(s"[EXAMPLE - COPYTASK] fatal error at run ${seqLens(runs.size)} with exception message: $e")
      else
        t.Fatalf(s"[EXAMPLE - COPYTASK] fatal after run with exception message: $e")
    }
  }
}
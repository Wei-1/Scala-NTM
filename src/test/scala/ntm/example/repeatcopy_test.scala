package ntm.example

object repeatcopy_test {
  def TestRun(t: ntm.T) {
    val h1Size = 12
    val numHeads = 2
    val n = 20
    val m = 8
    val confs = Array(1 -> 1, 2 -> 3, 3 -> 4, 7 -> 7)
    var runCount = 0
  	try {
      val (ox, oy) = repeatcopy.GenSeqBT(1, 1)
      val c = ntm.Controller.NewEmptyController(ox.head.size, oy.head.size, h1Size, numHeads, n, m)
      val weights = c.WeightsValVec()
      for(i <- 0 until weights.size; j <- 0 until weights(i).size)
        weights(i)(j) = 1 * (Math.random - 0.5)

      for((repeat, len) <- confs) {
        val (x, y) = repeatcopy.GenSeqBT(repeat, len)
        val model = new ntm.LogisticModel(Y = y)
        val machines = ntm.NTM.ForwardBackward(c, x, model)
        val predicts = ntm.NTM.Predictions(machines)
        val hWeights = ntm.NTM.HeadWeights(machines)
        val l = model.Loss(predicts)
        val bps = l / (y.size * y.head.size)
        t.Logf(s"[EXAMPLE - REPEATCOPY] repeat: $repeat, length: $len, loss: $bps")
        runCount += 1
      }
    } catch { case e: Exception =>
      if(runCount < confs.size)
        t.Fatalf(s"[EXAMPLE - REPEATCOPY] fatal error at run ${confs(runCount)} with exception message: $e")
      else
        t.Fatalf(s"[EXAMPLE - REPEATCOPY] fatal after run with exception message: $e")
    }
  }
}
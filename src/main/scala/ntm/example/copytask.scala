package ntm.example

object copytask {
  def GenSeq(size: Int, vectorSize: Int): (Array[Array[Double]], Array[Array[Double]]) = {
    val data = new Array[Array[Double]](size)
    for(i <- 0 until data.size) {
      data(i) = new Array[Double](vectorSize)
      for(j <- 0 until vectorSize)
        data(i)(j) = (Math.random * 2).toInt
    }

    val input = new Array[Array[Double]](size * 2 + 2)
    for(i <- 0 until input.size) {
      input(i) = new Array[Double](vectorSize + 2)
      if(i == 0) {
        input(i)(vectorSize) = 1
      } else if(i <= size) {
        for(j <- 0 until vectorSize)
          input(i)(j) = data(i - 1)(j)
      } else if(i == size + 1) {
        input(i)(vectorSize + 1) = 1
      }
    }

    val output = new Array[Array[Double]](size * 2 + 2)
    for(i <- 0 until output.size) {
      output(i) = new Array[Double](vectorSize)
      if(i >= size + 2) {
        for(j <- 0 until vectorSize)
          output(i)(j) = data(i - (size + 2))(j)
      }
    }

    (input, output)
  }


  def main(args: Array[String]) {
    val vectorSize = 3 // down-scaling for faster predicting
    val h1Size = 12 // down-scaling for faster predicting
    val numHeads = 1
    val n = 20 // down-scaling for faster predicting
    val m = 8 // down-scaling for faster predicting
    val c = ntm.Controller.NewEmptyController(vectorSize + 2, vectorSize, h1Size, numHeads, n, m)
    val weights = c.WeightsValVec()
    for(i <- 0 until weights.size; j <- 0 until weights(i).size)
      weights(i)(j) = 1 * (Math.random - 0.5)

    var losses = Array[Double]()

    val rmsp = ntm.RMSProp.NewRMSProp(c)
    println("Training -")
    println(s"numweights: ${c.WeightsValVec().size}")
    for(i <- 0 to 1000) {
      val randLength = if(i % 100 == 0) 6 else ((Math.random * 6).toInt + 1)
      val (x, y) = GenSeq(randLength, vectorSize)
      val model = new ntm.LogisticModel(Y = y)
      val machines = rmsp.Train(x, model, 0.95, 0.5, 1e-3, 1e-3)
      val l = model.Loss(ntm.NTM.Predictions(machines))
      if(i % 100 == 0) {
        val bpc = l / (y.size * y.head.size)
        losses :+= bpc
        println(s"$i, bpc: $bpc, seq length: ${y.size}")
      }
    }

    println("Predicting -")
    val seqLens = Array(2, 4, 6, 10, 16)
    for(seql <- seqLens) {
      val (x, y) = GenSeq(seql, vectorSize)
      val model = new ntm.LogisticModel(Y = y)
      val machines = ntm.NTM.ForwardBackward(c, x, model)
      val predicts = ntm.NTM.Predictions(machines)
      val hWeights = ntm.NTM.HeadWeights(machines)
      val l = model.Loss(predicts)
      val bps = l / (y.size * y.head.size)
      println(s"sequence length: $seql, loss: $bps")
      println(s"x: ${ntm.math.Sprint2(x)}, y: ${ntm.math.Sprint2(y)}")
      println(s"predictions: ${ntm.math.Sprint2(predicts)}")
    }

    println("Print Weights -")
    val finalWeights = c.WeightsValVec()
    finalWeights.foreach(arr => println(arr.mkString(",")))
  }
}

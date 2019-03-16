package ntm

class Run(
  SeqLen: Int,
  BitsPerSeq: Double,
  X: Array[Array[Double]],
  Y: Array[Array[Double]],
  Predictions: Array[Array[Double]],
  HeadWeights: Array[Array[Array[Double]]]
) {}

object copytask {
  def GenSeq(size: Int, vectorSize: Int): (Array[Array[Double]], Array[Array[Double]]) = {
    val data = new Array[Array[Double]](size)
    for(i <- 0 until data.size) {
      data(i) = new Array[Double](vectorSize)
      for(j <- 0 until vectorSize) {
        data(i)(j) = (Math.random * 2).toInt
      }
    }

    val input = new Array[Array[Double]](size * 2 + 2)
    for(i <- 0 until input.size) {
      input(i) = new Array[Double](vectorSize + 2)
      if(i == 0) {
        input(i)(vectorSize) = 1
      } else if(i <= size) {
        for(j <- 0 until vectorSize) {
          input(i)(j) = data(i - 1)(j)
        }
      } else if(i == size + 1) {
        input(i)(vectorSize + 1) = 1
      }
    }

    val output = new Array[Array[Double]](size * 2 + 2)
    for(i <- 0 until output.size) {
      output(i) = new Array[Double](vectorSize)
      if(i >= size + 2) {
        for(j <- 0 until vectorSize) {
          output(i)(j) = data(i - (size + 2))(j)
        }
      }
    }

    (input, output)
  }


  def main(args: Array[String]) {
    val vectorSize = 8
    val h1Size = 100
    val numHeads = 1
    val n = 128
    val m = 20
    val c = ntm.Controller.NewEmptyController(vectorSize + 2, vectorSize, h1Size, numHeads, n, m)
    val weights = c.WeightsValVec()
    for(i <- 0 until weights.size; j <- 0 until weights(i).size) {
      weights(i)(j) = 1 * (Math.random - 0.5)
    }

    var losses = Array[Double]()

    val rmsp = ntm.RMSProp.NewRMSProp(c)
    println(s"numweights: ${c.WeightsValVec().size}")
    for(i <- 1 to 100000) {
      val (x, y) = GenSeq((Math.random * 20).toInt + 1, vectorSize)
      val model = new ntm.LogisticModel(Y = y)
      val machines = rmsp.Train(x, model, 0.95, 0.5, 1e-3, 1e-3)
      val l = model.Loss(ntm.NTM.Predictions(machines))
      if(i % 1000 == 0) {
        val bpc = l / (y.size * y.head.size)
        losses :+= bpc
        println(s"$i, bpc: $bpc, seq length: ${y.size}")
      }
    }
    // val seqLens = Array(10, 20, 30, 50, 120)
    // var runs = Array[Run]()
    // for(seql <- seqLens) {
    //   val (x, y) = GenSeq(seql, vectorSize)
    //   val model = new ntm.LogisticModel(Y = y)
    //   val machines = ntm.NTM.ForwardBackward(c, x, model)
    //   val l = model.Loss(ntm.NTM.Predictions(machines))
    //   val bps = l / (y.size * y.head.size)
    //   println(s"sequence length: $seql, loss: $bps")

    //   val r = new Run(
    //     SeqLen = seql,
    //     BitsPerSeq = bps,
    //     X = x,
    //     Y = y,
    //     Predictions = ntm.NTM.Predictions(machines),
    //     HeadWeights = ntm.NTM.HeadWeights(machines)
    //   )
    //   runs :+= r
    //   //println(s"x: $x")
    //   //println(s"y: $y")
    //   //println(s"predictions: ${ntm.Sprint2(ntm.Predictions(machines))}")
    // }
  }
}

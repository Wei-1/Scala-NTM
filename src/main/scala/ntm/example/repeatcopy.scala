package ntm.example

object repeatcopy {

  // binary on time
  def GenSeqBT(repeat: Int, seqlen: Int): (Array[Array[Double]], Array[Array[Double]]) = {
    val data = randData(seqlen)
    val vectorSize = data.head.size
    val inputSize = vectorSize + 4
    val outputSize = vectorSize + 1

    var input = Array[Array[Double]]()
    var marker = new Array[Double](inputSize)
    marker(vectorSize) = 1
    input :+= marker

    input ++= data.map(_.take(inputSize))

    marker = new Array[Double](inputSize)
    marker(vectorSize + 1) = 1
    input :+= marker

    // Encode repeat times as little endian.
    val repeatBin = repeat.toBinaryString
    for(i <- repeatBin.size - 1 to 0 by -1) {
      val v = new Array[Double](inputSize)
      if(repeatBin(i) == '1') {
        v(vectorSize + 2) = 1
      } else {
        v(vectorSize + 2) = 0
      }
      input :+= v
    }

    var v = new Array[Double](inputSize)
    v(vectorSize + 3) = 1
    input :+= v

    var output = new Array[Array[Double]](input.size)
    for(i <- 0 until input.size) {
      output(i) = new Array[Double](outputSize)
    }
    for(i <- 0 until repeat) {
      for(datum <- data) {
        input :+= new Array[Double](inputSize)
        var tmpOut = datum.take(outputSize)
        if(outputSize > datum.size)
          tmpOut ++= new Array[Double](outputSize - datum.size)
        output :+= tmpOut
      }
    }

    input :+= new Array[Double](inputSize)
    marker = new Array[Double](outputSize)
    marker(vectorSize) = 1
    output :+= marker

    (input, output)
  }

  // linear on time
  def GenSeqLT(repeat: Int, seqlen: Int): (Array[Array[Double]], Array[Array[Double]]) = {
    val data = randData(seqlen)
    val vectorSize = data.head.size
    val inputSize = vectorSize + 3
    val outputSize = vectorSize + 1

    var input = Array[Array[Double]]()
    var marker = new Array[Double](inputSize)
    marker(vectorSize) = 1
    input :+= marker

    input ++= data.map(_.take(inputSize))

    // Encode repeat times as repititions.
    for(i <- 0 until repeat) {
      val v =  new Array[Double](inputSize)
      v(vectorSize + 1) = 1
      input :+= v
    }

    val v = new Array[Double](inputSize)
    v(vectorSize + 2) = 1
    input :+= v

    var output = new Array[Array[Double]](input.size)
    for(i <- 0 until input.size) {
      output(i) = new Array[Double](outputSize)
    }
    for(i <- 0 until repeat) {
      for(datum <- data) {
        input :+= new Array[Double](inputSize)
        var tmpOut = datum.take(outputSize)
        if(outputSize > datum.size)
          tmpOut ++= new Array[Double](outputSize - datum.size)
        output :+= tmpOut
      }
    }

    input :+= new Array[Double](inputSize)
    marker = new Array[Double](outputSize)
    marker(vectorSize) = 1
    output :+= marker

    (input, output)
  }

  // GenSeq generates a sequence with the number of repitions specified as a scaler.
  def GenSeq(repeat: Int, seqlen: Int): (Array[Array[Double]], Array[Array[Double]]) = {
    val data = randData(seqlen)
    val vectorSize = data.head.size
    val inputSize = vectorSize + 2
    val outputSize = vectorSize + 1

    var input = Array[Array[Double]]()
    var marker = new Array[Double](inputSize)
    marker(vectorSize) = 1
    input :+= marker

    input ++= data.map(_.take(inputSize))

    // Encode repeat times as a scalar.
    val v = new Array[Double](inputSize)
    v(vectorSize + 1) = repeat.toDouble
    input :+= v

    var output = new Array[Array[Double]](input.size)
    for(i <- 0 until input.size) {
      output(i) = new Array[Double](outputSize)
    }
    for(i <- 0 until repeat) {
      for(datum <- data) {
        input :+= new Array[Double](inputSize)
        var tmpOut = datum.take(outputSize)
        if(outputSize > datum.size)
          tmpOut ++= new Array[Double](outputSize - datum.size)
        output :+= tmpOut
      }
    }

    input :+= new Array[Double](inputSize)
    marker = new Array[Double](outputSize)
    marker(vectorSize) = 1
    output :+= marker

    (input, output)
  }

  def randData(size: Int): Array[Array[Double]] = {
    val vectorSize = 3
    val data = new Array[Array[Double]](size)
    for(i <- 0 until data.size) {
      data(i) = new Array[Double](vectorSize)
      for(j <- 0 until data(i).size) {
        data(i)(j) = (math.random * 2).toInt
      }
    }
    data
  }

  def main(args: Array[String]) {

    val (ox, oy) = repeatcopy.GenSeqBT(1, 1)
    val h1Size = 12
    val numHeads = 2
    val n = 20
    val m = 8
    val c = ntm.Controller.NewEmptyController(ox.head.size, oy.head.size, h1Size, numHeads, n, m)
    val weights = c.WeightsValVec()
    for(i <- 0 until weights.size; j <- 0 until weights(i).size)
      weights(i)(j) = 1 * (Math.random - 0.5)

    var losses = Array[Double]()

    val rmsp = ntm.RMSProp.NewRMSProp(c)
    println("Training -")
    println(s"numweights: ${c.WeightsValVec().size}")
    for(i <- 0 to 1000) {
      val (x, y) = repeatcopy.GenSeqBT((Math.random * 5).toInt + 1, (Math.random * 5).toInt + 1)
      val model = new ntm.LogisticModel(Y = y)
      val machines = rmsp.Train(x, model, 0.95, 0.5, 1e-3, 1e-3)
      val l = model.Loss(ntm.NTM.Predictions(machines))
      if(i % 1000 == 0) {
        val bpc = l / (y.size * y.head.size)
        losses :+= bpc
        println(s"$i, bpc: $bpc, seq length: ${y.size}")
      }
    }

    println("Predicting -")

    val confs = Array(1 -> 1, 2 -> 3, 3 -> 4, 7 -> 7)
    for((repeat, len) <- confs) {
      val (x, y) = repeatcopy.GenSeqBT(repeat, len)
      val model = new ntm.LogisticModel(Y = y)
      val machines = ntm.NTM.ForwardBackward(c, x, model)
      val predicts = ntm.NTM.Predictions(machines)
      val hWeights = ntm.NTM.HeadWeights(machines)
      val l = model.Loss(predicts)
      val bps = l / (y.size * y.head.size)
      println(s"repeat: $repeat, length: $len, loss: $bps")
      println(s"x: ${ntm.math.Sprint2(x)}, y: ${ntm.math.Sprint2(y)}")
      println(s"predictions: ${ntm.math.Sprint2(predicts)}")
    }

    println("Print Weights -")
    val finalWeights = c.WeightsValVec()
    finalWeights.foreach(arr => println(arr.mkString(",")))
  }
}

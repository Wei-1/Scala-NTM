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
        output :+= datum.take(outputSize)
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
        output :+= datum.take(outputSize)
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
        output :+= datum.take(outputSize)
      }
    }

    input :+= new Array[Double](inputSize)
    marker = new Array[Double](outputSize)
    marker(vectorSize) = 1
    output :+= marker

    (input, output)
  }

  def randData(size: Int): Array[Array[Double]] = {
    val vectorSize = 6
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
  }
}

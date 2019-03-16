package ntm.example

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

  
}

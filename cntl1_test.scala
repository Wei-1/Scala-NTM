package ntm

import math._

object cntl1_test {
  def TestLogisticModel(t: T) {
    val times = 9
    val x = makeTensor2(times, 4)
    for(i <- 0 until x.size) {
      for(j <- 0 until x(i).size) {
        x(i)(j) = 0.0001 * i * j + 0.1//Math.random
      }
    }
    val y = makeTensor2(times, 4)
    for(i <- 0 until y.size) {
      for(j <- 0 until y(i).size) {
        y(i)(j) = 0.0001 * i * j + 0.1//Math.random
      }
    }
    val n = 3
    val m = 2
    val h1Size = 3
    val numHeads = 2
    val c = controller1.NewEmptyController1(x(0).size, y(0).size, h1Size, numHeads, n, m)
    val weights = c.WeightsVal()
    for(i <- 0 until weights.size) {
      weights(i) = 2 *    0.001 * i + 0.2//Math.random
    }
    // println(" - cm " + c.WeightsGrad().mkString(","))
    val model = new LogisticModel(Y = y)
    NTM.ForwardBackward(c, x, model)
    // println(" - cm " + c.WeightsGrad().mkString(","))
    checkGradients(t, c, ControllerForward, x, model)
  }

  def TestMultinomialModel(t: T) {
    val times = 9
    val x = makeTensor2(times, 4)
    for(i <- 0 until x.size) {
      for(j <- 0 until x(i).size) {
        x(i)(j) = 0.0001 * i * j + 0.1//Math.random
      }
    }
    val outputSize = 4
    val y= new Array[Int](times)
    for(i <- 0 until y.size) {
      y(i) = ((0.001 * i + 0.1/*Math.random*/) * outputSize).toInt
    }
    val n = 3
    val m = 2
    val h1Size = 3
    val numHeads = 2
    val c = controller1.NewEmptyController1(x(0).size, outputSize, h1Size, numHeads, n, m)
    val weights = c.WeightsVal()
    for(i <- 0 until weights.size) {
      weights(i) = 2 *    0.001 * i + 0.2//Math.random
    }

    val model = new MultinomialModel(Y = y)
    NTM.ForwardBackward(c, x, model)
    checkGradients(t, c, ControllerForward, x, model)
  }

  // A ControllerForward is a ground truth implementation of the forward pass of a controller.
  def ControllerForward(c1: controller1, reads: Array[Array[Double]], x: Array[Double]): (Array[Double], Array[Head]) = {
    val c = c1
    var readX = Array[Double]()
    for(read <- reads) {
      for(r <- read) {
        readX :+= r
      }
    }
    for(xi <- x) {
      readX :+= xi
    }
    readX :+= 1.0
    var h1 = new Array[Double](c.h1Size)
    val wh1 = c.wh1Val()
    for(i <- 0 until h1.size) {
      var v: Double = 0.0
      for(j <- 0 until readX.size) {
        v += wh1(i)(j) * readX(j)
      }
      h1(i) = Sigmoid(v)
    }

    val out = new Array[Double](c.wyRows())
    val wy = c.wyVal()
    h1 :+= 1.0
    for(i <- 0 until out.size) {
      var v: Double = 0.0
      for(j <- 0 until h1.size) {
        v += wy(i)(j) * h1(j)
      }
      out(i) = v
    }
    val prediction = new Array[Double](c.ySize)
    for(i <- 0 until prediction.size) {
      prediction(i) = out(i)
    }
    val heads = new Array[Head](c.numHeads)
    for(i <- 0 until heads.size) {
      heads(i) = Head.NewHead(c.memoryM)
      val hul = Head.headUnitsLen(c.MemoryM())
      heads(i).vals = new Array[Double](hul)
      heads(i).grads = new Array[Double](hul)
      for(j <- 0 until heads(i).vals.size) {
        heads(i).vals(j) += out(c.ySize + i * hul + j)
      }
    }

    (prediction, heads)
  }

  def loss(
    c: controller1,
    forward: (controller1, Array[Array[Double]], Array[Double]) => (Array[Double], Array[Head]),
    in: Array[Array[Double]],
    model: DensityModel
  ): Double = {
    // Initialize memory as in the function ForwardBackward
    var mem = unit.makeTensorUnit2(c.MemoryN(), c.MemoryM())
    for(i <- 0 until mem.size) {
      for(j <- 0 until mem(i).size) {
        mem(i)(j).Val = c.Mtm1BiasVal()(i * c.MemoryM() + j)
      }
    }
    var wtm1s = new Array[refocus](c.NumHeads())
    for(i <- 0 until wtm1s.size) {
      wtm1s(i) = new refocus(
        TopVal = new Array[Double](c.MemoryN()),
        TopGrad = new Array[Double](c.MemoryN())
      )
      val bs = c.Wtm1BiasVal().take((i + 1) * c.MemoryN()).drop(i * c.MemoryN())
      var sum: Double = 0
      for(j <- 0 until bs.size) {
        wtm1s(i).TopVal(j) = Math.exp(bs(j))
        sum += wtm1s(i).TopVal(j)
      }
      for(j <- 0 until bs.size) {
        wtm1s(i).TopVal(j) = wtm1s(i).TopVal(j) / sum
      }
    }
    var reads = makeTensor2(c.NumHeads(), c.MemoryM())
    for(i <- 0 until reads.size) {
      for(j <- 0 until reads(i).size) {
        var v: Double = 0
        for(k <- 0 until mem.size) {
          v += wtm1s(i).TopVal(k) * mem(k)(j).Val
        }
        reads(i)(j) = v
      }
    }

    val prediction = new Array[Array[Double]](in.size)
    var heads: Array[Head] = null
    for(j <- 0 until in.size) {
      val (p, h) = forward(c, reads, in(j))
      prediction(j) = p
      heads = h
      prediction(j) = computeDensity(j, prediction(j), model)
      for(i <- 0 until heads.size) {
        heads(i).Wtm1 = wtm1s(i)
      }
      val (wsDouble, readsDouble, memDouble) = addressing_test.doAddressing(heads, mem)
      wtm1s = transformWSDouble(wsDouble)
      reads = readsDouble
      mem = transformMemDouble(memDouble)
    }

    model.Loss(prediction)
  }

  def computeDensity(timestep: Int, pred: Array[Double], model: DensityModel): Array[Double] = {
    val den = pred.clone
    model.Model(timestep, den, new Array[Double](pred.size))
    den
  }

  def checkGradients(
    t: T,
    c: controller1,
    forward: (controller1, Array[Array[Double]], Array[Double]) => (Array[Double], Array[Head]),
    in: Array[Array[Double]],
    model: DensityModel
  ) {
    val lx = loss(c, forward, in, model)

    for(i <- 0 until c.WeightsVal().size) {
      val x = c.WeightsVal()(i)
      val h = machineEpsilonSqrt * Math.max(Math.abs(x), 1)
      val xph = x + h
      c.WeightsVal()(i) = xph
      val lxph = loss(c, forward, in, model)
      c.WeightsVal()(i) = x
      val grad = (lxph - lx) / (xph - x)

      val wGrad = c.WeightsGrad()(i)
      val tag = c.WeightsDesc(i)
      if(grad.isNaN || Math.abs(grad-wGrad) > 1e-5) {
        t.Errorf(s"[CNTL] wrong $tag gradient expected $grad, got $wGrad")
      } else {
        t.Logf(s"[CNTL] OK $tag gradient expected $grad, got $wGrad")
      }
    }
  }

  def transformMemDouble(memDouble: Array[Array[Double]]): Array[Array[unit]] = {
    val mem = unit.makeTensorUnit2(memDouble.size, memDouble(0).size)
    for(i <- 0 until mem.size) {
      for(j <- 0 until mem(0).size) {
        mem(i)(j).Val = memDouble(i)(j)
      }
    }
    mem
  }

  def transformWSDouble(wsDouble: Array[Array[Double]]): Array[refocus] = {
    val wtm1s = new Array[refocus](wsDouble.size)
    for(i <- 0 until wtm1s.size) {
      wtm1s(i) = new refocus(
        TopVal = new Array[Double](wsDouble(i).size),
        TopGrad = new Array[Double](wsDouble(i).size)
      )
      for(j <- 0 until wtm1s(i).TopVal.size) {
        wtm1s(i).TopVal(j) = wsDouble(i)(j)
      }
    }
    wtm1s
  }
}

package com.scalaml.ntm

import math._

object ctrl_test {
  def TestSaveLoad(t: T) {
    val times = 9
    val inputSize = 10
    val outputSize = 4
    val n = 3
    val m = 2
    val h1Size = 3
    val numHeads = 2
    val c1 = Controller.NewEmptyController(inputSize, outputSize, h1Size, numHeads, n, m)
    val weights = c1.WeightsValVec()
    for(i <- 0 until weights.size; j <- 0 until weights(i).size)
      weights(i)(j) = 2 * Math.random
    c1.Save("TestWeightFile")
    val c2 = Controller.NewEmptyController(inputSize, outputSize, h1Size, numHeads, n, m)
    c2.Load("TestWeightFile")
    val testWeights = c2.WeightsValVec()
    var check = true
    for(i <- 0 until weights.size; j <- 0 until weights(i).size) {
      check &= weights(i)(j) == testWeights(i)(j)
    }
    if(check) {
      t.Logf(s"[CTRL] OK save load weights")
    } else {
      t.Errorf(s"[CTRL] wrong save load weights")
    }
  }
  def TestLogisticModel(t: T) {
    val times = 9
    val x = makeTensor2(times, 4)
    for(i <- 0 until x.size; j <- 0 until x(i).size)
      x(i)(j) = Math.random
    val y = makeTensor2(times, 4)
    for(i <- 0 until y.size; j <- 0 until y(i).size)
      y(i)(j) = Math.random
    val n = 3
    val m = 2
    val h1Size = 3
    val numHeads = 2
    val c = Controller.NewEmptyController(x(0).size, y(0).size, h1Size, numHeads, n, m)
    val weights = c.WeightsValVec()
    for(i <- 0 until weights.size; j <- 0 until weights(i).size)
      weights(i)(j) = 2 * Math.random
    val model = new LogisticModel(Y = y)
    // println(" weights " + weights.map(_.mkString(",")).mkString(";"))
    // println(" - cm " + c.WeightsGradVec().map(_.mkString(",")).mkString(";"))
    NTM.ForwardBackward(c, x, model)
    // println(" - cm " + c.WeightsGradVec().map(_.mkString(",")).mkString(";"))
    checkGradients(t, c, ControllerForward, x, model)
  }

  def TestMultinomialModel(t: T) {
    val times = 9
    val x = makeTensor2(times, 4)
    for(i <- 0 until x.size; j <- 0 until x(i).size)
      x(i)(j) = Math.random
    val outputSize = 4
    val y = new Array[Int](times)
    for(i <- 0 until y.size)
      y(i) = (Math.random * outputSize).toInt
    val n = 3
    val m = 2
    val h1Size = 3
    val numHeads = 2
    val c = Controller.NewEmptyController(x(0).size, outputSize, h1Size, numHeads, n, m)
    val weights = c.WeightsValVec()
    for(i <- 0 until weights.size; j <- 0 until weights(i).size)
      weights(i)(j) = 2 * Math.random

    val model = new MultinomialModel(Y = y)
    NTM.ForwardBackward(c, x, model)
    checkGradients(t, c, ControllerForward, x, model)
  }

  // A ControllerForward is a ground truth implementation of the forward pass of a controller.
  def ControllerForward(c1: Controller, reads: Array[Array[Double]], x: Array[Double]): (Array[Double], Array[Head]) = {
    // println(reads.map(_.mkString(",")).mkString(" "))
    val c = c1
    var readX = Array[Double]()
    for(read <- reads; r <- read) readX :+= r
    for(xi <- x) readX :+= xi
    readX :+= 1.0
    var h1 = new Array[Double](c.h1Size)
    val wh1 = c.wh1ValVec()
    for(i <- 0 until h1.size) {
      var v: Double = 0.0
      for(j <- 0 until readX.size) v += wh1(i)(j) * readX(j)
      h1(i) = Sigmoid(v)
    }
    h1 :+= 1.0
    // println(h1.mkString(" ")) // <-- CORRECT
    val out = Controller.newOutMemory(c)//new Array[Double](c.wyRows())
    val wy = c.wyValVec()
    // println(wy.map(_.map(_.mkString(",")).mkString(" ")).mkString("   "))
    for(i <- 0 until out.size; j <- 0 until out(i).size) {
      var v: Double = 0.0
      for(k <- 0 until h1.size) v += wy(k)(i)(j) * h1(k)
      out(i)(j) = v
    }
    // println(out.map(_.mkString(",")).mkString(" ")) // <-- ERROR -> FIXED
    val prediction = new Array[Double](c.ySize)
    for(i <- 0 until prediction.size) {
      prediction(i) = out.head(i)
    }
    val heads = new Array[Head](c.numHeads)
    for(i <- 0 until heads.size) {
      heads(i) = Head.NewHead(c.memoryM)
      val hul = Head.headUnitsLen(c.MemoryM())
      heads(i).vals = new Array[Double](hul)
      heads(i).grads = new Array[Double](hul)
      for(j <- 0 until heads(i).vals.size) heads(i).vals(j) += out(i + 1)(j)
    }

    (prediction, heads)
  }

  def loss(
    c: Controller,
    forward: (Controller, Array[Array[Double]], Array[Double]) => (Array[Double], Array[Head]),
    in: Array[Array[Double]],
    model: DensityModel
  ): Double = {
    // Initialize memory as in the function ForwardBackward
    var mem = unit.makeTensorUnit2(c.MemoryN(), c.MemoryM())
    for(i <- 0 until mem.size; j <- 0 until mem(i).size)
      mem(i)(j).Val = c.Mtm1BiasValVec()(i)(j)
    // println(mem.map(_.map(_.Val).mkString(",")).mkString(" "))
    var wtm1s = new Array[refocus](c.NumHeads())
    for(i <- 0 until wtm1s.size) {
      wtm1s(i) = new refocus(
        TopVal = new Array[Double](c.MemoryN()),
        TopGrad = new Array[Double](c.MemoryN())
      )
      val bs = c.Wtm1BiasValVec()(i)
      var sum: Double = 0
      for(j <- 0 until bs.size) {
        wtm1s(i).TopVal(j) = Math.exp(bs(j))
        sum += wtm1s(i).TopVal(j)
      }
      // println(wtm1s(i).TopVal.mkString(" "))
      // print(sum + " ")
      for(j <- 0 until bs.size) wtm1s(i).TopVal(j) = wtm1s(i).TopVal(j) / sum
      // println(wtm1s(i).TopVal.mkString(" "))
    }
    // println(wtm1s.map(_.TopVal.mkString(",")).mkString(" "))
    var reads = makeTensor2(c.NumHeads(), c.MemoryM())
    for(i <- 0 until reads.size; j <- 0 until reads(i).size) {
      var v: Double = 0
      for(k <- 0 until mem.size) v += wtm1s(i).TopVal(k) * mem(k)(j).Val
      reads(i)(j) = v
    }
    // println(reads.map(_.mkString(",")).mkString(" "))
    val prediction = new Array[Array[Double]](in.size)
    var heads: Array[Head] = null
    for(j <- 0 until in.size) {
      val (p, h) = forward(c, reads, in(j))
      // println(p.mkString(","))
      prediction(j) = p
      heads = h
      prediction(j) = computeDensity(j, prediction(j), model)
      for(i <- 0 until heads.size) heads(i).Wtm1 = wtm1s(i)
      val (wsDouble, readsDouble, memDouble) = addressing_test.doAddressing(heads, mem)
      wtm1s = transformWSDouble(wsDouble)
      reads = readsDouble
      mem = transformMemDouble(memDouble)
    }
    // println(prediction.map(_.mkString(",")).mkString(" "))
    model.Loss(prediction)
  }

  def computeDensity(timestep: Int, pred: Array[Double], model: DensityModel): Array[Double] = {
    val den = pred.clone
    model.Model(timestep, den, new Array[Double](pred.size))
    den
  }

  def checkGradients(
    t: T,
    c: Controller,
    forward: (Controller, Array[Array[Double]], Array[Double]) => (Array[Double], Array[Head]),
    in: Array[Array[Double]],
    model: DensityModel
  ) {
    // println(in.map(_.mkString(",")).mkString(" "))
    val lx = loss(c, forward, in, model)
    // println(lx)
    val value = c.WeightsValVec()
    for(i <- 0 until value.size; j <- 0 until value(i).size) {
      val x = value(i)(j)
      val h = machineEpsilonSqrt * Math.max(Math.abs(x), 1)
      val xph = x + h
      value(i)(j) = xph
      val lxph = loss(c, forward, in, model)
      value(i)(j) = x
      val grad = (lxph - lx) / (xph - x)

      val wGrad = c.WeightsGradVec()(i)(j)
      val tag = c.WeightsDesc(i, j)
      // print(x + " " + h + " " + xph + " " + lxph)
      if(grad.isNaN || Math.abs(grad - wGrad) > 1e-4) {
        t.Errorf(s"[CTRL] wrong $tag gradient expected $grad, got $wGrad")
      } else {
        t.Logf(s"[CTRL] OK $tag gradient expected $grad, got $wGrad")
      }
    }
  }

  def transformMemDouble(memDouble: Array[Array[Double]]): Array[Array[unit]] = {
    val mem = unit.makeTensorUnit2(memDouble.size, memDouble(0).size)
    for(i <- 0 until mem.size; j <- 0 until mem(0).size) {
      mem(i)(j).Val = memDouble(i)(j)
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

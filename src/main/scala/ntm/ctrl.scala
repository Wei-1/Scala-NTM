package com.scalaml.ntm

import org.apache.mxnet._
import math._

class Controller(
  var weightsValVec: Array[Array[Double]] = null,
  var weightsGradVec: Array[Array[Double]] = null,

  var Reads: Array[memRead] = null,
  var X: Array[Double] = null,
  var ReadsXVal: Array[Double] = null,

  var H1Val: Array[Double] = null,
  var H1Grad: Array[Double] = null,

  var heads: Array[Head] = null,
  var outValVec: Array[Array[Double]] = null,
  var outGradVec: Array[Array[Double]] = null,

  val numHeads: Int,
  val memoryM: Int,
  val memoryN: Int,
  val xSize: Int,
  val h1Size: Int,
  val ySize: Int
) {
  def wh1Cols(): Int = numHeads * memoryM + xSize + 1

  def wyRowsVec(): Int = 1 + numHeads

  def wyOffsetVec(): Int = h1Size

  def wtm1OffsetVec(): Int = wyOffsetVec() + wyRowsVec() * (h1Size + 1)

  def mtm1OffsetVec(): Int = wtm1OffsetVec() + numHeads

  def numWeightsVec(): Int = mtm1OffsetVec() + memoryN

  def wh1Vec(w: Array[Array[Double]]): Array[Array[Double]] =
    w.take(wyOffsetVec())

  def wh1ValVec(): Array[Array[Double]] = wh1Vec(weightsValVec)

  def wh1GradVec(): Array[Array[Double]] = wh1Vec(weightsGradVec)

  def wyVec(w: Array[Array[Double]]): Array[Array[Array[Double]]] =
    w.drop(wyOffsetVec()).take(wyRowsVec() * (h1Size + 1)).grouped(wyRowsVec()).toArray

  def wyValVec(): Array[Array[Array[Double]]] = wyVec(weightsValVec)

  def wyGradVec(): Array[Array[Array[Double]]] = wyVec(weightsGradVec)

  def Wtm1BiasValVec(): Array[Array[Double]] =
    weightsValVec.drop(wtm1OffsetVec()).take(numHeads)

  def Wtm1BiasGradVec(): Array[Array[Double]] =
    weightsGradVec.drop(wtm1OffsetVec()).take(numHeads)

  def Mtm1BiasValVec(): Array[Array[Double]] =
    weightsValVec.drop(mtm1OffsetVec())

  def Mtm1BiasGradVec(): Array[Array[Double]] =
    weightsGradVec.drop(mtm1OffsetVec())

  def Heads(): Array[Head] = heads

  def YValVec(): Array[Double] = outValVec.head

  def YGradVec(): Array[Double] = outGradVec.head

  def Forward(reads: Array[memRead], x: Array[Double]): Controller = {
    val c = new Controller(
      weightsValVec = weightsValVec,
      weightsGradVec = weightsGradVec,
      Reads = reads,
      X = x,
      H1Val = new Array[Double](h1Size + 1),
      H1Grad = new Array[Double](h1Size + 1),
      heads = new Array[Head](reads.size),
      outValVec = Controller.newOutMemory(this),
      outGradVec = Controller.newOutMemory(this),

      numHeads = numHeads,
      memoryM = memoryM,
      memoryN = memoryN,
      xSize = xSize,
      h1Size = h1Size,
      ySize = ySize,
    )

    val ud = new Array[Double](c.wh1Cols())
    for(i <- 0 until reads.size; j <- 0 until reads(i).TopVal.size)
        ud(i * c.memoryM + j) = reads(i).TopVal(j)
    for(j <- 0 until c.X.size)
      ud(c.numHeads * c.memoryM + j) = c.X(j)
    ud(c.numHeads * c.memoryM + c.xSize) = 1

    c.ReadsXVal = ud

    val h1 = c.H1Val

    for(i <- 0 until c.wh1Cols(); j <- 0 until h1Size) // special Gemv handler
      h1(j) += c.wh1ValVec()(j)(i) * c.ReadsXVal(i)
    for(i <- 0 until c.h1Size)
      h1(i) = Sigmoid(h1(i))
    h1(c.h1Size) = 1

    for(i <- 0 until c.outValVec.size; j <- 0 until c.outValVec(i).size;
      k <- 0 until h1.size) // special Gemv handler
      c.outValVec(i)(j) += c.wyValVec()(k)(i)(j) * h1(k)

    val hul = Head.headUnitsLen(c.memoryM)
    for(i <- 0 until c.heads.size) {
      val head = Head.NewHead(c.memoryM)
      c.heads(i) = head
      head.vals = c.outValVec(i + 1)
      head.grads = c.outGradVec(i + 1)
    }
    c
  }

  def Backward() {
    val out = outGradVec
    val h1Val = H1Val
    val h1Grad = H1Grad
    for(i <- 0 until h1Size + 1; j <- 0 until wyRowsVec(); k <- 0 until out(j).size)
      h1Grad(i) += wyValVec()(i)(j)(k) * out(j)(k)
    for(i <- 0 until h1Size + 1; j <- 0 until wyRowsVec(); k <- 0 until out(j).size)
      wyGradVec()(i)(j)(k) += out(j)(k) * h1Val(i)

    for(i <- 0 until h1Size)
      h1Grad(i) *= h1Val(i) * (1 - h1Val(i))

    val u = new Array[Double](wh1Cols())
    for(i <- 0 until u.size; j <- 0 until h1Size) // limit to h1Size
      u(i) += wh1ValVec()(j)(i) * h1Grad(j)
    for(i <- 0 until wh1Cols(); j <- 0 until h1Size) // limit to h1Size
      wh1GradVec()(j)(i) += h1Grad(j) * ReadsXVal(i)

    for(i <- 0 until Reads.size; j <- 0 until Reads(i).TopGrad.size)
      Reads(i).TopGrad(j) = u(i * memoryM + j)
  }

  def WeightsValVec(): Array[Array[Double]] = weightsValVec
  def WeightsGradVec(): Array[Array[Double]] = weightsGradVec

  def WeightsDesc(i: Int, j: Int): String = s"[$i][$j]"

  def NumHeads(): Int = numHeads

  def MemoryN(): Int = memoryN

  def MemoryM(): Int = memoryM

  def Save(file: String): Unit = {
    val wStr = WeightsValVec.map(_.mkString(",")).mkString(";")
    val writer = new java.io.FileWriter(file)
    writer.write(wStr)
    writer.close
  }

  def Load(file: String): Unit = {
    val wStr = scala.io.Source.fromFile(file).getLines.mkString
    weightsValVec = wStr.split(';').map(_.split(',').map(_.toDouble))
  }
}

// NewEmptyController returns a new controller1 which is a single layer feedforward network.
// The returned Controller is empty in that all its network weights are initialized as 0.
object Controller {
  def newOutMemory(c: Controller): Array[Array[Double]] = new Array[Double](c.ySize) +:
        Array.ofDim[Double](c.numHeads, Head.headUnitsLen(c.memoryM))
  def newControllerMemory(c: Controller): Array[Array[Double]] =
    Array.ofDim[Double](c.h1Size, c.wh1Cols()) ++
    Array.fill(c.h1Size + 1)(newOutMemory(c)).flatten.asInstanceOf[Array[Array[Double]]] ++
    Array.ofDim[Double](c.numHeads, c.memoryN) ++
    Array.ofDim[Double](c.memoryN, c.memoryM)

  def NewEmptyController(xSize: Int, ySize: Int, h1Size: Int, numHeads: Int, n: Int, m: Int): Controller = {
    val c = new Controller(
      numHeads = numHeads,
      memoryM = m,
      memoryN = n,
      xSize = xSize,
      h1Size = h1Size,
      ySize = ySize
    )
    c.weightsValVec = newControllerMemory(c)
    c.weightsGradVec = newControllerMemory(c)
    c
  }
}

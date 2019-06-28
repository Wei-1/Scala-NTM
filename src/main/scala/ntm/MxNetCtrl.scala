package com.scalaml.ntm

import org.apache.mxnet._

import com.scalaml.ntm.MxNetMath._

class MxNetCtrl(
  var weightsValVec: Array[NDArray] = null, // wh1ValVec, wyValVec, Wtm1BiasValVec, Mtm1BiasValVec
  var weightsGradVec: Array[NDArray] = null, // wh1GradVec, wyGradVec, Wtm1BiasGradVec, Mtm1BiasGradVec

  var Reads: Array[memRead] = null,
  var X: NDArray = null,
  var ReadsXVal: NDArray = null,

  var H1Val: NDArray = null,
  var H1Grad: NDArray = null,

  var heads: Array[Head] = null,
  var outValVec: Array[NDArray] = null, // YValVec, outValMemory
  var outGradVec: Array[NDArray] = null, // YGradVec, outGradMemory

  val numHeads: Int,
  val memoryM: Int,
  val memoryN: Int,
  val xSize: Int,
  val h1Size: Int,
  val ySize: Int
) {
  def wh1Cols: Int = numHeads * memoryM + xSize + 1
  def wyVecSize: Int = (h1Size + 1) * 2
  // def wyRowsVec(): Int = 1 + numHeads
  // def wyOffsetVec(): Int = h1Size
  // def wtm1OffsetVec(): Int = wyOffsetVec() + wyRowsVec() * (h1Size + 1)
  // def mtm1OffsetVec(): Int = wtm1OffsetVec() + numHeads

  def wh1ValVec(): NDArray = weightsValVec(0) // w.take(wyOffsetVec())
  def wh1GradVec(): NDArray = weightsGradVec(0) // w.take(wyOffsetVec())

  // w.drop(wyOffsetVec()).take(wyRowsVec() * (h1Size + 1)).grouped(wyRowsVec()).toArray
  def wyValVec(): Array[Array[NDArray]] = weightsValVec.drop(1).take(wyVecSize).grouped(2).toArray
  def wyGradVec(): Array[Array[NDArray]] = weightsGradVec.drop(1).take(wyVecSize).grouped(2).toArray

  def Wtm1BiasValVec(): NDArray = weightsValVec(wyVecSize + 1) // w.drop(wtm1OffsetVec()).take(numHeads)
  def Wtm1BiasGradVec(): NDArray = weightsGradVec(wyVecSize + 1) // w.drop(wtm1OffsetVec()).take(numHeads)

  def Mtm1BiasValVec(): NDArray = weightsValVec(wyVecSize + 2) // w.drop(mtm1OffsetVec())
  def Mtm1BiasGradVec(): NDArray = weightsGradVec(wyVecSize + 2) // w.drop(mtm1OffsetVec())

  def Heads(): Array[Head] = heads
  def YValVec(): NDArray = outValVec(0)
  def YGradVec(): NDArray = outGradVec(0)

  def WeightsDesc(i: Int, j: Int): String = s"[$i][$j]"

  def WeightsValVec(): Array[NDArray] = weightsValVec
  def WeightsGradVec(): Array[NDArray] = weightsGradVec

  def NumHeads(): Int = numHeads
  def MemoryN(): Int = memoryN
  def MemoryM(): Int = memoryM

  def Save(file: String): Unit = NDArray.save(file, WeightsValVec)

  def Load(file: String): Unit = weightsValVec = NDArray.load(file)._2

//   def Forward(reads: Array[memRead], x: NDArray): Controller = {
//     val c = new Controller(
//       weightsValVec = weightsValVec,
//       weightsGradVec = weightsGradVec,
//       Reads = reads,
//       X = x,
//       H1Val = NDArray.zeros(h1Size + 1, 1),
//       H1Grad = NDArray.zeros(h1Size + 1, 1),
//       heads = new Array[Head](reads.size),
//       outValVec = Controller.newOutMemory(this),
//       outGradVec = Controller.newOutMemory(this),

//       numHeads = numHeads,
//       memoryM = memoryM,
//       memoryN = memoryN,
//       xSize = xSize,
//       h1Size = h1Size,
//       ySize = ySize
//     )

//     val ud = new Array[Double](c.wh1Cols)
//     for(i <- 0 until reads.size; j <- 0 until reads(i).TopVal.size)
//         ud(i * c.memoryM + j) = reads(i).TopVal(j)
//     for(j <- 0 until c.X.size)
//       ud(c.numHeads * c.memoryM + j) = c.X(j)
//     ud(c.numHeads * c.memoryM + c.xSize) = 1

//     c.ReadsXVal = NDArray.array(ud, Shape(c.wh1Cols, 1))

//     val h1 = c.H1Val

//     for(i <- 0 until c.wh1Cols; j <- 0 until h1Size) // special Gemv handler
//       h1(j) += c.wh1ValVec()(j)(i) * c.ReadsXVal(i)
//     for(i <- 0 until c.h1Size)
//       h1(i) = Sigmoid(h1(i))
//     h1(c.h1Size) = 1

//     for(i <- 0 until c.outValVec.size; j <- 0 until c.outValVec(i).size;
//       k <- 0 until h1.size) // special Gemv handler
//       c.outValVec(i)(j) += c.wyValVec()(k)(i)(j) * h1(k)

//     val hul = Head.headUnitsLen(c.memoryM)
//     for(i <- 0 until c.heads.size) {
//       val head = Head.NewHead(c.memoryM)
//       c.heads(i) = head
//       head.vals = c.outValVec(i + 1)
//       head.grads = c.outGradVec(i + 1)
//     }
//     c
//   }

//   def Backward() {
//     val out = outGradVec
//     val h1Val = H1Val
//     val h1Grad = H1Grad
//     for(i <- 0 until h1Size + 1; j <- 0 until wyRowsVec(); k <- 0 until out(j).size)
//       h1Grad(i) += wyValVec()(i)(j)(k) * out(j)(k)
//     for(i <- 0 until h1Size + 1; j <- 0 until wyRowsVec(); k <- 0 until out(j).size)
//       wyGradVec()(i)(j)(k) += out(j)(k) * h1Val(i)

//     for(i <- 0 until h1Size)
//       h1Grad(i) *= h1Val(i) * (1 - h1Val(i))

//     val u = new Array[Double](wh1Cols)
//     for(i <- 0 until u.size; j <- 0 until h1Size) // limit to h1Size
//       u(i) += wh1ValVec()(j)(i) * h1Grad(j)
//     for(i <- 0 until wh1Cols; j <- 0 until h1Size) // limit to h1Size
//       wh1GradVec()(j)(i) += h1Grad(j) * ReadsXVal(i)

//     for(i <- 0 until Reads.size; j <- 0 until Reads(i).TopGrad.size)
//       Reads(i).TopGrad(j) = u(i * memoryM + j)
//   }
}

// NewEmptyController returns a new controller1 which is a single layer feedforward network.
// The returned MxNetCtrl is empty in that all its network weights are initialized as 0.
object MxNetCtrl {
  def newOutMemory(c: MxNetCtrl): Array[NDArray] = Array(
    NDArray.zeros(c.ySize, 1),
    NDArray.zeros(c.numHeads, Head.headUnitsLen(c.memoryM))
  )
  def newControllerMemory(c: MxNetCtrl): Array[NDArray] =
    Array(NDArray.zeros(c.h1Size, c.wh1Cols)) ++
    Array.fill(c.h1Size + 1)(newOutMemory(c)).flatten.asInstanceOf[Array[NDArray]] ++
    Array(NDArray.zeros(c.numHeads, c.memoryN), NDArray.zeros(c.memoryN, c.memoryM))
  def NewEmptyController(xSize: Int, ySize: Int, h1Size: Int, numHeads: Int, n: Int, m: Int): MxNetCtrl = {
    val c = new MxNetCtrl(
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

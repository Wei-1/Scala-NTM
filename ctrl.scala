package ntm

import math._

class Controller(
  // var weightsVal: Array[Double] = null,
  var weightsValVec: Array[Array[Double]] = null,
  // var weightsGrad: Array[Double] = null,
  var weightsGradVec: Array[Array[Double]] = null,

  var Reads: Array[memRead] = null,
  var X: Array[Double] = null,
  var ReadsXVal: Array[Double] = null,

  var H1Val: Array[Double] = null,
  var H1Grad: Array[Double] = null,

  var heads: Array[Head] = null,
  // var outVal: Array[Double] = null,
  var outValVec: Array[Array[Double]] = null,
  // var outGrad: Array[Double] = null,
  var outGradVec: Array[Array[Double]] = null,

  val numHeads: Int,
  val memoryM: Int,
  val memoryN: Int,
  val xSize: Int,
  val h1Size: Int,
  val ySize: Int
) {
  def wh1Cols(): Int = numHeads * memoryM + xSize + 1

  // def wyRows(): Int = ySize + numHeads * Head.headUnitsLen(memoryM)
  def wyRowsVec(): Int = 1 + numHeads

  // def wyOffset(): Int = h1Size * wh1Cols()
  def wyOffsetVec(): Int = h1Size

  // def wtm1Offset(): Int = wyOffset() + wyRows() * (h1Size + 1)
  def wtm1OffsetVec(): Int = wyOffsetVec() + wyRowsVec() * (h1Size + 1)

  // def mtm1Offset(): Int = wtm1Offset() + numHeads * memoryN
  def mtm1OffsetVec(): Int = wtm1OffsetVec() + numHeads

  // def numWeights(): Int = mtm1Offset() + memoryN * memoryM
  def numWeightsVec(): Int = mtm1OffsetVec() + memoryN

  // def wh1(w: Array[Double]): Array[Array[Double]] =
  //   w.take(wyOffset()).grouped(wh1Cols()).toArray
  def wh1Vec(w: Array[Array[Double]]): Array[Array[Double]] =
    w.take(wyOffsetVec())

  // def wh1Val(): Array[Array[Double]] = wh1(weightsVal)
  def wh1ValVec(): Array[Array[Double]] = wh1Vec(weightsValVec)

  // def wh1Grad(): Array[Array[Double]] = wh1(weightsGrad)
  def wh1GradVec(): Array[Array[Double]] = wh1Vec(weightsGradVec)

  // def wy(w: Array[Double]): Array[Array[Double]] =
  //   w.take(wtm1Offset()).drop(wyOffset()).grouped(h1Size + 1).toArray
  def wyVec(w: Array[Array[Double]]): Array[Array[Array[Double]]] =
    w.take(wtm1OffsetVec()).drop(wyOffsetVec()).grouped(wyRowsVec()).toArray

  // def wyVal(): Array[Array[Double]] = wy(weightsVal)
  def wyValVec(): Array[Array[Array[Double]]] = wyVec(weightsValVec)

  // def wyGrad(): Array[Array[Double]] = wy(weightsGrad)
  def wyGradVec(): Array[Array[Array[Double]]] = wyVec(weightsGradVec)

  // def Wtm1BiasVal(): Array[Double] =
  //   weightsVal.take(mtm1Offset()).drop(wtm1Offset())
  def Wtm1BiasValVec(): Array[Array[Double]] =
    weightsValVec.take(mtm1OffsetVec()).drop(wtm1OffsetVec())

  // def Wtm1BiasGrad(): Array[Double] =
  //   weightsGrad.take(mtm1Offset()).drop(wtm1Offset())
  def Wtm1BiasGradVec(): Array[Array[Double]] =
    weightsGradVec.take(mtm1OffsetVec()).drop(wtm1OffsetVec())

  // def Mtm1BiasVal(): Array[Double] = weightsVal.drop(mtm1Offset())
  def Mtm1BiasValVec(): Array[Array[Double]] =
    weightsValVec.drop(mtm1OffsetVec())

  // def Mtm1BiasGrad(): Array[Double] = weightsGrad.drop(mtm1Offset())
  def Mtm1BiasGradVec(): Array[Array[Double]] =
    weightsGradVec.drop(mtm1OffsetVec())

  def Heads(): Array[Head] = heads

  // def YVal(): Array[Double] = outVal.take(ySize)
  def YValVec(): Array[Double] = outValVec.head

  // def YGrad(): Array[Double] = outGrad.take(ySize)
  def YGradVec(): Array[Double] = outGradVec.head

  def Forward(reads: Array[memRead], x: Array[Double]): Controller = {
    val c = new Controller(
      // weightsVal = weightsVal,
      weightsValVec = weightsValVec,
      // weightsGrad = weightsGrad,
      weightsGradVec = weightsGradVec,
      Reads = reads,
      X = x,
      H1Val = new Array[Double](h1Size + 1),
      H1Grad = new Array[Double](h1Size + 1),
      heads = new Array[Head](reads.size),
      // outVal = new Array[Double](wyRows()),
      outValVec = Controller.newOutMemory(this),
      // outGrad = new Array[Double](wyRows()),
      outGradVec = Controller.newOutMemory(this),

      numHeads = numHeads,
      memoryM = memoryM,
      memoryN = memoryN,
      xSize = xSize,
      h1Size = h1Size,
      ySize = ySize,
    )

    val ud = new Array[Double](c.wh1Cols())
    for(i <- 0 until reads.size) {
      for(j <- 0 until reads(i).TopVal.size)
        ud(i * c.memoryM + j) = reads(i).TopVal(j)
    }
    for(j <- 0 until c.X.size)
      ud(c.numHeads * c.memoryM + j) = c.X(j)
    ud(c.numHeads * c.memoryM + c.xSize) = 1
    c.ReadsXVal = ud

    val h1 = c.H1Val
    // Gemv(blas.NoTrans, 1, c.wh1Val(), c.ReadsXVal, 1, h1)
    for(i <- 0 until c.wh1Cols(); j <- 0 until h1Size) // special Gemv handler
      h1(j) += c.wh1ValVec()(j)(i) * c.ReadsXVal(i)
    for(i <- 0 until c.h1Size)
      h1(i) = Sigmoid(h1(i))
    h1(c.h1Size) = 1

    // Gemv(blas.NoTrans, 1, c.wyVal(), h1, 1, c.outVal)
    for(i <- 0 until c.outValVec.size; j <- 0 until c.wyRowsVec();
      k <- 0 until c.outValVec.head.size) // special Gemv handler
      c.outValVec(i)(k) += c.wyValVec()(j)(i)(k) * h1(j)

    val hul = Head.headUnitsLen(c.memoryM)
    for(i <- 0 until c.heads.size) {
      val head = Head.NewHead(c.memoryM)
      c.heads(i) = head
      // val start = c.ySize + i * hul
      // head.vals = c.outVal.take(start + hul).drop(start)
      head.vals = c.outValVec(i + 1)
      // head.grads = c.outGrad.take(start + hul).drop(start)
      head.grads = c.outGradVec(i + 1)
    }
    c
  }

  def Backward() {
    // val out = outGrad
    val out = outGradVec
    val h1Val = H1Val
    val h1Grad = H1Grad
    // Gemv(blas.Trans, 1, wyVal(), out, 1, h1Grad)
    for(i <- 0 until h1Size + 1; j <- 0 until wyRowsVec(); k <- 0 until out(j).size)
      h1Grad(i) += wyValVec()(i)(j)(k) * out(j)(k)
    // Ger(1, out, h1Val, wyGrad())
    for(i <- 0 until h1Size + 1; j <- 0 until wyRowsVec(); k <- 0 until out(j).size)
      wyGradVec()(i)(j)(k) += out(j)(k) * h1Val(i)

    // h1Val = Vector{Inc: 1, Data: H1Val[0:h1Size]}
    // h1Grad = Vector{Inc: 1, Data: H1Grad[0:h1Size]}
    for(i <- 0 until h1Size)
      h1Grad(i) *= h1Val(i) * (1 - h1Val(i))

    val u = new Array[Double](wh1Cols())
    // Gemv(blas.Trans, 1, wh1Val(), h1Grad, 1, u)
    for(i <- 0 until u.size; j <- 0 until h1Size) // limit to h1Size
      u(i) += wh1ValVec()(j)(i) * h1Grad(j)
    // Ger(1, h1Grad, ReadsXVal, wh1Grad())
    for(i <- 0 until wh1Cols(); j <- 0 until h1Size) // limit to h1Size
      wh1GradVec()(j)(i) += h1Grad(j) * ReadsXVal(i)

    for(i <- 0 until Reads.size; j <- 0 until Reads(i).TopGrad.size)
      Reads(i).TopGrad(j) = u(i * memoryM + j)
  }

  // def WeightsVal(): Array[Double] = weightsVal
  // def WeightsGrad(): Array[Double] = weightsGrad
  def WeightsValVec(): Array[Array[Double]] = weightsValVec
  def WeightsGradVec(): Array[Array[Double]] = weightsGradVec

  def WeightsDesc(i: Int): String = i.toString

  def NumHeads(): Int = numHeads

  def MemoryN(): Int = memoryN

  def MemoryM(): Int = memoryM

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
    // c.weightsVal = new Array[Double](c.numWeights()) // <--- not finished
    // c.weightsGrad = new Array[Double](c.numWeights()) // <--- not finished
    c.weightsValVec = newControllerMemory(c)
    c.weightsGradVec = newControllerMemory(c)
    c
  }
}

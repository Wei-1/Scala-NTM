package ntm

import math._

class controller1(
  var weightsVal: Array[Double] = null,
  var weightsGrad: Array[Double] = null,

  var Reads: Array[memRead] = null,
  var X: Array[Double] = null,
  var ReadsXVal: Array[Double] = null,

  var H1Val: Array[Double] = null,
  var H1Grad: Array[Double] = null,

  var heads: Array[Head] = null,
  var outVal: Array[Double] = null,
  var outGrad: Array[Double] = null,

  val numHeads: Int,
  val memoryM: Int,
  val memoryN: Int,
  val xSize: Int,
  val h1Size: Int,
  val ySize: Int
) extends Controller {
  def wh1Cols(): Int = {
    numHeads * memoryM + xSize + 1
  }

  def wyRows(): Int = {
    ySize + numHeads * Head.headUnitsLen(memoryM)
  }

  def wyOffset(): Int = {
    h1Size * wh1Cols()
  }

  def wtm1Offset(): Int = {
    wyOffset() + wyRows() * (h1Size + 1)
  }

  def mtm1Offset(): Int = {
    wtm1Offset() + numHeads * memoryN
  }

  def numWeights(): Int = {
    mtm1Offset() + memoryN * memoryM
  }

  def wh1(w: Array[Double]): Array[Array[Double]] = {
    w.take(wyOffset()).grouped(wh1Cols()).toArray
  }

  def wh1Val(): Array[Array[Double]] = {
    wh1(weightsVal)
  }

  def wh1Grad(): Array[Array[Double]] = {
    wh1(weightsGrad)
  }

  def wy(w: Array[Double]): Array[Array[Double]] = {
    w.take(wtm1Offset()).drop(wyOffset()).grouped(h1Size + 1).toArray
  }

  def wyVal(): Array[Array[Double]] = {
    wy(weightsVal)
  }

  def wyGrad(): Array[Array[Double]] = {
    wy(weightsGrad)
  }

  def Wtm1BiasVal(): Array[Double] = {
    weightsVal.take(mtm1Offset()).drop(wtm1Offset())
  }

  def Wtm1BiasGrad(): Array[Double] = {
    weightsGrad.take(mtm1Offset()).drop(wtm1Offset())
  }

  def Mtm1BiasVal(): Array[Double] = {
    weightsVal.drop(mtm1Offset())
  }

  def Mtm1BiasGrad(): Array[Double] = {
    weightsGrad.drop(mtm1Offset())
  }

  def Heads(): Array[Head] = {
    heads
  }

  def YVal(): Array[Double] = {
    outVal.take(ySize)
  }

  def YGrad(): Array[Double] = {
    outGrad.take(ySize)
  }

  def Forward(reads: Array[memRead], x: Array[Double]): Controller = {
    val c = new controller1(
      weightsVal = weightsVal,
      weightsGrad = weightsGrad,
      Reads = reads,
      X = x,
      H1Val = new Array[Double](h1Size + 1),
      H1Grad = new Array[Double](h1Size + 1),
      heads = new Array[Head](reads.size),
      outVal = new Array[Double](wyRows()),
      outGrad = new Array[Double](wyRows()),

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

    val h1 = c.H1Val //.take(c.h1Size)
    // Gemv(t Trans, alpha f64, A General, x Vec, beta f64, y Vec)
    // y = alpha * A * x + beta * y; if t == blas.NoTrans
    // blas64.Gemv(blas.NoTrans, 1, c.wh1Val(), c.ReadsXVal, 1, h1)
    for(i <- 0 until c.h1Size; j <- 0 until wh1Cols()) {
      h1(i) += c.wh1Val()(i)(j) * c.ReadsXVal(j)
    }
    for(i <- 0 until c.h1Size) {
      c.H1Val(i) = Sigmoid(c.H1Val(i))
    }

    c.H1Val(c.h1Size) = 1
    // h1 = c.H1Val
    val outV = c.outVal
    // blas64.Gemv(blas.NoTrans, 1, c.wyVal(), h1, 1, outV)
    for(i <- 0 until outV.size; j <- 0 until h1Size + 1) {
      outV(i) += c.wyVal()(i)(j) * h1(j)
    }

    val hul = Head.headUnitsLen(c.memoryM)
    for(i <- 0 until c.heads.size) {
      val head = Head.NewHead(c.memoryM)
      c.heads(i) = head
      val start = c.ySize + i * hul
      head.vals = c.outVal.take(start + hul).drop(start)
      head.grads = c.outGrad.take(start + hul).drop(start)
    }
    c
  }

  def Backward() {
    val out = outGrad
    val h1Val = H1Val
    val h1Grad = H1Grad
    // blas64.Gemv(blas.Trans, 1, wyVal(), out, 1, h1Grad)
    for(i <- 0 until h1Grad.size; j <- 0 until out.size)
      h1Grad(i) += wyVal()(j)(i) * out(j)
    // Ger(alpha f64, x, y Vec, A General)
    // A += alpha * x * y^T
    // blas64.Ger(1, out, h1Val, wyGrad())
    // weightsGrad.take(wtm1Offset()).drop(wyOffset()).grouped(h1Size + 1)
    for(i <- 0 until h1Size + 1; j <- 0 until wyRows())
      weightsGrad(wyOffset() + j * (h1Size + 1) + i) += out(j) * h1Val(i)

    // h1Val = blas64.Vector{Inc: 1, Data: H1Val[0:h1Size]}
    // h1Grad = blas64.Vector{Inc: 1, Data: H1Grad[0:h1Size]}
    for(i <- 0 until h1Val.size) {
      h1Grad(i) *= h1Val(i) * (1 - h1Val(i))
    }

    val u = new Array[Double](wh1Cols())
    // blas64.Gemv(blas.Trans, 1, wh1Val(), h1Grad, 1, u)
    for(i <- 0 until u.size; j <- 0 until h1Size)
      u(i) += wh1Val()(j)(i) * h1Grad(j)
    // Ger(alpha f64, x, y Vec, A General)
    // A += alpha * x * y^T
    // blas64.Ger(1, h1Grad, ReadsXVal, wh1Grad())
    // weightsGrad.take(wyOffset()).grouped(wh1Cols())
    for(i <- 0 until wh1Cols(); j <- 0 until h1Size + 1) {
      weightsGrad(j * wh1Cols() + i) += h1Grad(j) + ReadsXVal(i)
    }

    for(i <- 0 until Reads.size; j <- 0 until Reads(i).TopGrad.size) {
      Reads(i).TopGrad(j) = u(i * memoryM + j)
    }
  }

  def WeightsVal(): Array[Double] = {
    weightsVal
  }

  def WeightsGrad(): Array[Double] = {
    weightsGrad
  }

  def WeightsDesc(i: Int): String = {
    if(i < wyOffset()) {
      s"wh1[${i / wh1Cols()}][${i % wh1Cols()}]"
    } else if(i < wtm1Offset()) {
      val j = i - wyOffset()
      val cols = h1Size + 1
      s"wy[${j / cols}][${i % cols}]"
    } else if(i < mtm1Offset()) {
      val j = i - wtm1Offset()
      s"wtm1[${j / memoryN}][${j % memoryN}]"
    } else {
      val j = i - mtm1Offset()
      s"mtm1[${j / memoryM}][${j % memoryM}]"
    }
  }

  def NumHeads(): Int = numHeads

  def MemoryN(): Int = memoryN

  def MemoryM(): Int = memoryM

}

// NewEmptyController1 returns a new controller1 which is a single layer feedforward network.
// The returned controller1 is empty in that all its network weights are initialized as 0.
object controller1 {
  def NewEmptyController1(xSize: Int, ySize: Int, h1Size: Int, numHeads: Int, n: Int, m: Int): controller1 = {
    val c = new controller1(
      numHeads = numHeads,
      memoryM = m,
      memoryN = n,
      xSize = xSize,
      h1Size = h1Size,
      ySize = ySize,
    )
    c.weightsVal = new Array[Double](c.numWeights())
    c.weightsGrad = new Array[Double](c.numWeights())
    c
  }
}

/*
Package ntm implements the Neural Turing Machine architecture as described in A.Graves,
G. Wayne, and I. Danihelka. arXiv preprint arXiv:1410.5401, 2014.

Using this package along its subpackages, the "copy", "repeatcopy" and "ngram" tasks
mentioned in the paper were verified.
For each of these tasks, the successfully trained models are saved under the filenames
"seedA_B", where A is the number indicating the seed provided to rand.Seed in the
training process, and B is the iteration number in which the trained weights converged.
*/
package ntm

import math._
// A Head is a read write head on a memory bank.
// It carriess information that is required to operate on a memory bank according
// to the NTM architecture.
class Head(
  // size of a row in the memory
  val M: Int
) {
  // the weights at time t-1
  var Wtm1: refocus = null
  var vals: Array[Double] = null
  var grads: Array[Double] = null

  // EraseVector returns the erase vector of a memory head.
  def EraseVal(): Array[Double] = vals.take(M)
  def EraseGrad(): Array[Double] = grads.take(M)

  // AddVector returns the add vector of a memory head.
  def AddVal(): Array[Double] = vals.drop(M).take(M)
  def AddGrad(): Array[Double] = grads.drop(M).take(M)

  // K returns a head's key vector, which is the target data in the content addressing
  // step.
  def KVal(): Array[Double] = vals.drop(2 * M).take(M)
  def KGrad(): Array[Double] = grads.drop(2 * M).take(M)

  // Beta returns the key strength of a content addressing step.
  def BetaVal(): Double = vals(3 * M)
  def BetaGrad(): Double = grads(3 * M)

  // G returns the degree in which we want to choose content-addressing over
  // location-based-addressing.
  def GVal(): Double = vals(3 * M + 1)
  def GGrad(): Double = grads(3 * M + 1)

  // S returns a value indicating how much the weightings are rotated in a
  // location-based-addressing step.
  def SVal(): Double = vals(3 * M + 2)
  def SGrad(): Double = grads(3 * M + 2)

  // Gamma returns the degree in which the addressing weights are sharpened.
  def GammaVal(): Double = vals(3 * M + 3)
  def GammaGrad(): Double = grads(3 * M + 3)
}

object Head {
  // NewHead creates a new memory head.
  def NewHead(m: Int): Head = new Head(M = m)
  def headUnitsLen(m: Int): Int = 3 * m + 4
}

// The Controller Interface is implemented by NTM controller networks that wish to
// operate with memory banks in a NTM.
trait Controller {
  // Heads returns the emitted memory heads.
  def Heads(): Array[Head]
  // YVal returns the values of the output of the Controller.
  def YVal(): Array[Double]
  // YVal returns the gradients of the output of the Controller.
  def YGrad(): Array[Double]

  // Forward creates a new Controller which shares the same Internal weights,
  // and performs a forward pass whose results can be retrived by Heads and Y.
  def Forward(reads: Array[memRead], x: Array[Double]): Controller
  // Backward performs a backward pass,
  // assuming the gradients on Heads and Y are already set.
  def Backward(): Unit

  // Wtm1BiasVal returns the values of the bias of the previous weight.
  // The layout is |-- 1st head weights (size memoryN) --|-- 2nd head --|-- ... --|
  // The length of the returned slice is numHeads * memoryN.
  def Wtm1BiasVal(): Array[Double]
  def Wtm1BiasGrad(): Array[Double]

  // M1mt1BiasVal returns the values of the bias of the memory bank.
  // The returned matrix is in row major order.
  def Mtm1BiasVal(): Array[Double]
  def Mtm1BiasGrad(): Array[Double]

  // WeightsVal returns the values of all weights.
  def WeightsVal(): Array[Double]
  // WeightsGrad returns the gradients of all weights.
  def WeightsGrad(): Array[Double]
  // WeightsDesc returns the descriptions of a weight.
  def WeightsDesc(i: Int): String

  // NumHeads returns the number of memory heads of a controller.
  def NumHeads(): Int
  // MemoryN returns the number of vectors of the memory bank of a controller.
  def MemoryN(): Int
  // MemoryM returns the size of a vector in the memory bank of a controller.
  def MemoryM(): Int
}

// A NTM is a neural turing machine as described in A.Graves, G. Wayne, and I. Danihelka.
// arXiv preprint arXiv:1410.5401, 2014.
class NTM(val Controller: Controller, var memOp: memOp = null) {
  def backward(): Unit = {
    memOp.Backward()
    Controller.Backward()
  }
}

object NTM {
  // NewNTM creates a new NTM.
  def NewNTM(old: NTM, x: Array[Double]): NTM = {
    val m = new NTM(Controller = old.Controller.Forward(old.memOp.R, x))
    for(i <- 0 until m.Controller.Heads().size) {
      m.Controller.Heads()(i).Wtm1 = old.memOp.W(i)
    }
    m.memOp = memOp.newMemOp(m.Controller.Heads(), old.memOp.WM)
    m
  }

  // ForwardBackward computes a controller's prediction and gradients with respect to the
  // given ground truth input and output values.
  def ForwardBackward(c: Controller, in: Array[Array[Double]], out: DensityModel): Array[NTM] = {
    val weights = c.WeightsGrad()
    for(i <- 0 until weights.size) {
      weights(i) = 0
    }

    // Set the empty NTM's memory and head weights to their bias values.
    val (empty, reads, cas) = makeEmptyNTM(c)
    val machines = new Array[NTM](in.size)

    // Backpropagation through time.
    machines(0) = NewNTM(empty, in(0))
    for(t <- 1 until in.size) {
      machines(t) = NewNTM(machines(t-1), in(t))
    }
    for(t <- in.size - 1 to 0 by -1) {
      val m = machines(t)
      out.Model(t, m.Controller.YVal(), m.Controller.YGrad())
      m.backward()
    }

    // Compute gradients for the bias values of the initial memory and weights.
    for(i <- 0 until reads.size) {
      reads(i).Backward()
      for(j <- 0 until reads(i).W.TopGrad.size) {
        cas(i).Top(j).Grad += reads(i).W.TopGrad(j)
      }
      cas(i).Backward()
    }

    // Copy gradients to the controller.
    val cwtm1 = c.Wtm1BiasGrad()
    for(i <- 0 until cas.size) {
      for(j <- 0 until cas(i).Units.size) {
        cwtm1(i * c.MemoryN() + j) = cas(i).Units(j).Top.Grad
      }
    }

    machines
  }

  // MakeEmptyNTM makes a NTM with its memory and head weights set to their bias values,
  // based on the controller.
  def MakeEmptyNTM(c: Controller): NTM = {
    val (machine, _, _) = makeEmptyNTM(c)
    machine
  }

  def makeEmptyNTM(c: Controller): (NTM, Array[memRead], Array[contentAddressing]) = {
    val cwtm1 = c.Wtm1BiasVal()
    val unws = new Array[Array[betaSimilarity]](c.NumHeads())
    for(i <- 0 until unws.size) {
      unws(i) = new Array[betaSimilarity](c.MemoryN())
      for(j <- 0 until unws(i).size) {
        val v = cwtm1(i * c.MemoryN() + j)
        unws(i)(j) = new betaSimilarity(Top = new unit(Val = v))
      }
    }

    val mtm1 = new writtenMemory(
      N = c.MemoryN(),
      TopVal = c.Mtm1BiasVal(),
      TopGrad = c.Mtm1BiasGrad()
    )

    val wtm1s = new Array[refocus](c.NumHeads())
    val reads = new Array[memRead](c.NumHeads())
    val cas = new Array[contentAddressing](c.NumHeads())
    for(i <- 0 until reads.size) {
      cas(i) = contentAddressing.newContentAddressing(unws(i))
      wtm1s(i) = new refocus(
        TopVal = new Array[Double](c.MemoryN()),
        TopGrad = new Array[Double](c.MemoryN())
      )
      for(j <- 0 until wtm1s(i).TopVal.size) {
        wtm1s(i).TopVal(j) = cas(i).Top(j).Val
      }
      reads(i) = memRead.newMemRead(wtm1s(i), mtm1)
    }

    val empty = new NTM(
      Controller = c,
      memOp = new memOp(W = wtm1s, R = reads, WM = mtm1)
    )

    (empty, reads, cas)
  }

  // Predictions returns the predictions of a NTM across time.
  def Predictions(machines: Array[NTM]): Array[Array[Double]] = {
    val pdts = new Array[Array[Double]](machines.size)
    for(t <- 0 until pdts.size) {
      pdts(t) = machines(t).Controller.YVal()
    }
    pdts
  }

  // HeadWeights returns the addressing weights of all memory heads across time.
  // The top level elements represent each head.
  // The second level elements represent every time instant.
  def HeadWeights(machines: Array[NTM]): Array[Array[Array[Double]]] = {
    val hws = new Array[Array[Array[Double]]](machines(0).memOp.W.size)
    for(i <- 0 until hws.size) {
      hws(i) = new Array[Array[Double]](machines.size)
      for(t <- 0 until machines.size) {
        hws(i)(t) = new Array[Double](machines(t).memOp.W(i).TopVal.size)
        for(j <- 0 until machines(t).memOp.W(i).TopVal.size) {
          hws(i)(t)(j) = machines(t).memOp.W(i).TopVal(j)
        }
      }
    }
    hws
  }
}

// SGDMomentum implements stochastic gradient descent with momentum.
class SGDMomentum (
  val C: Controller,
  val PrevD: Array[Double]
){
  def Train(x: Array[Array[Double]], y: DensityModel, alpha: Double, mt: Double): Array[NTM] = {
    val machines = NTM.ForwardBackward(C, x, y)
    val weights = C.WeightsVal()
    for(i <- 0 until C.WeightsGrad().size) {
      val d = -alpha * C.WeightsGrad()(i) + mt * PrevD(i)
      weights(i) += d
      PrevD(i) = d
    }
    machines
  }
}

object SGDMomentum {
  def NewSGDMomentum(c: Controller): SGDMomentum = {
    new SGDMomentum(
      C = c,
      PrevD = new Array[Double](c.WeightsVal().size)
    )
  }
}

// RMSProp implements the rmsprop algorithm. The detailed updating equations are given in
// Graves, Alex (2013). Generating sequences with recurrent neural networks. arXiv
// preprint arXiv:1308.0850.
class RMSProp(
  val C: Controller,
  val N: Array[Double],
  val G: Array[Double],
  val D: Array[Double]
){
  def Train(x: Array[Array[Double]], y: DensityModel, a: Double, b: Double, c: Double, d: Double): Array[NTM] = {
    val machines = NTM.ForwardBackward(C, x, y)
    update(a, b, c, d)
    machines
  }

  def update(a: Double, b: Double, c: Double, d: Double): Unit = {
    val grad = C.WeightsGrad()
    val value = C.WeightsVal()
    for(i <- 0 until grad.size) {
      val grad2i = grad(i) * grad(i)
      N(i) = a * N(i) + (1 - a) * grad2i
      G(i) = a * G(i) + (1 - a) * grad(i)
      val rmsi = grad(i) / Math.sqrt(N(i) - G(i) * G(i) + d)
      D(i) = b * D(i) - c * rmsi
      value(i) += D(i)
    }
  }
}

object RMSProp {
  def NewRMSProp(c: Controller): RMSProp = {
    new RMSProp(
      C = c,
      N = new Array[Double](c.WeightsVal().size),
      G = new Array[Double](c.WeightsVal().size),
      D = new Array[Double](c.WeightsVal().size)
    )
  }
}

// A DensityModel is a model of how the last layer of a network gets transformed Into the
// final output.
trait DensityModel {
  // Model sets the value and gradient of Units of the output layer.
  def Model(t: Int, yHVal: Array[Double], yHGrad: Array[Double]): Unit

  // Loss is the loss definition of this model.
  def Loss(output: Array[Array[Double]]): Double
}

// A LogisticModel models its outputs as logistic sigmoids.
class LogisticModel(
  // Y is the strength of the output unit at each time step.
  val Y: Array[Array[Double]]
) extends DensityModel {
  // Model sets the values and gradients of the output units.
  override def Model(t: Int, yHVal: Array[Double], yHGrad: Array[Double]): Unit = {
    val ys = Y(t)
    for(i <- 0 until yHVal.size) {
      val newYhv = Sigmoid(yHVal(i))
      yHVal(i) = newYhv
      yHGrad(i) = newYhv - ys(i)
    }
  }

  // Loss returns the cross entropy loss.
  override def Loss(output: Array[Array[Double]]): Double = {
    var l: Double = 0
    for(t <- 0 until output.size) {
      for(i <- 0 until output(t).size) {
        val p = output(t)(i)
        val y = Y(t)(i)
        l += y * Math.log(p) + (1 - y) * Math.log(1 - p)
      }
    }
    -l
  }
}

// A MultinomialModel models its outputs as following the multinomial distribution.
class MultinomialModel(
  // Y is the class of the output at each time step.
  val Y: Array[Int]
) extends DensityModel {
  // Model sets the values and gradients of the output units.
  override def Model(t: Int, yHVal: Array[Double], yHGrad: Array[Double]): Unit = {
    var sum: Double = 0
    for(i <- 0 until yHVal.size) {
      val v = Math.exp(yHVal(i))
      yHVal(i) = v
      sum += v
    }

    val k = Y(t)
    for(i <- 0 until yHVal.size) {
      val newYhv = yHVal(i) / sum
      yHVal(i) = newYhv
      yHGrad(i) = newYhv - delta(i, k)
    }
  }

  override def Loss(output: Array[Array[Double]]): Double = {
    var l: Double = 0
    for(t <- 0 until output.size) {
      l += Math.log(output(t)(Y(t)))
    }
    -l
  }
}

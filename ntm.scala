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
  def getEraseVal(i: Int): Double = vals(i)
  def setEraseVal(i: Int, v: Double): Unit = vals(i) = v
  def addEraseVal(i: Int, v: Double): Unit = vals(i) += v
  def getEraseGrad(i: Int): Double = grads(i)
  def setEraseGrad(i: Int, v: Double): Unit = grads(i) = v
  def addEraseGrad(i: Int, v: Double): Unit = grads(i) += v

  // AddVector returns the add vector of a memory head.
  def getAddVal(i: Int): Double = vals(M + i)
  def setAddVal(i: Int, v: Double): Unit = vals(M + i) = v
  def addAddVal(i: Int, v: Double): Unit = vals(M + i) += v
  def getAddGrad(i: Int): Double = grads(M + i)
  def setAddGrad(i: Int, v: Double): Unit = grads(M + i) = v
  def addAddGrad(i: Int, v: Double): Unit = grads(M + i) += v

  // K returns a head's key vector,
  // which is the target data in the content addressing step.
  def getKVal(i: Int): Double = vals(2 * M + i)
  def setKVal(i: Int, v: Double): Unit = vals(2 * M + i) = v
  def addKVal(i: Int, v: Double): Unit = vals(2 * M + i) += v
  def getKGrad(i: Int): Double = grads(2 * M + i)
  def setKGrad(i: Int, v: Double): Unit = grads(2 * M + i) = v
  def addKGrad(i: Int, v: Double): Unit = grads(2 * M + i) += v

  // Beta returns the key strength of a content addressing step.
  def getBetaVal(): Double = vals(3 * M)
  def setBetaVal(v: Double): Unit = vals(3 * M) = v
  def addBetaVal(v: Double): Unit = vals(3 * M) += v
  def getBetaGrad(): Double = grads(3 * M)
  def setBetaGrad(v: Double): Unit = grads(3 * M) = v
  def addBetaGrad(v: Double): Unit = grads(3 * M) += v

  // G returns the degree in which we want to choose content-addressing over
  // location-based-addressing.
  def getGVal(): Double = vals(3 * M + 1)
  def setGVal(v: Double): Unit = vals(3 * M + 1) = v
  def addGVal(v: Double): Unit = vals(3 * M + 1) += v
  def getGGrad(): Double = grads(3 * M + 1)
  def setGGrad(v: Double): Unit = grads(3 * M + 1) = v
  def addGGrad(v: Double): Unit = grads(3 * M + 1) += v

  // S returns a value indicating how much the weightings are rotated in a
  // location-based-addressing step.
  def getSVal(): Double = vals(3 * M + 2)
  def setSVal(v: Double): Unit = vals(3 * M + 2) = v
  def addSVal(v: Double): Unit = vals(3 * M + 2) += v
  def getSGrad(): Double = grads(3 * M + 2)
  def setSGrad(v: Double): Unit = grads(3 * M + 2) = v
  def addSGrad(v: Double): Unit = grads(3 * M + 2) += v

  // Gamma returns the degree in which the addressing weights are sharpened.
  def getGammaVal(): Double = vals(3 * M + 3)
  def setGammaVal(v: Double): Unit = vals(3 * M + 3) = v
  def addGammaVal(v: Double): Unit = vals(3 * M + 3) += v
  def getGammaGrad(): Double = grads(3 * M + 3)
  def setGammaGrad(v: Double): Unit = grads(3 * M + 3) = v
  def addGammaGrad(v: Double): Unit = grads(3 * M + 3) += v
}

object Head {
  // NewHead creates a new memory head.
  def NewHead(m: Int): Head = new Head(M = m)
  def headUnitsLen(m: Int): Int = 3 * m + 4
}

// A NTM is a neural turing machine as described in A.Graves, G. Wayne, and I. Danihelka.
// arXiv preprint arXiv:1410.5401, 2014.
class NTM(val Controller: Controller, var memOp: memOp = null) {
  def backward(): Unit = {
    memOp.Backward()
    Controller.Backward() // <--- ERROR
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
    val weights = c.WeightsGradVec()
    for(i <- 0 until weights.size; j <- 0 until weights(i).size) {
      weights(i)(j) = 0
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
      out.Model(t, m.Controller.YValVec(), m.Controller.YGradVec())
      m.backward() //  <---- ERROR HERE
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
    val cwtm1 = c.Wtm1BiasGradVec()
    for(i <- 0 until cas.size; j <- 0 until cas(i).Units.size) {
      cwtm1(i)(j) = cas(i).Units(j).Top.Grad
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
    val cwtm1 = c.Wtm1BiasValVec()
    val unws = new Array[Array[betaSimilarity]](c.NumHeads())
    for(i <- 0 until unws.size) {
      unws(i) = new Array[betaSimilarity](c.MemoryN())
      for(j <- 0 until unws(i).size) {
        val v = cwtm1(i)(j)
        unws(i)(j) = new betaSimilarity(Top = new unit(Val = v))
      }
    }

    val mtm1 = new writtenMemory(
      N = c.MemoryN(),
      TopVal = c.Mtm1BiasValVec(),
      TopGrad = c.Mtm1BiasGradVec()
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
      pdts(t) = machines(t).Controller.YValVec()
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
  val PrevD: Array[Array[Double]]
){
  def Train(x: Array[Array[Double]], y: DensityModel, alpha: Double, mt: Double): Array[NTM] = {
    val machines = NTM.ForwardBackward(C, x, y)
    val grad = C.WeightsGradVec()
    val value = C.WeightsValVec()
    for(i <- 0 until grad.size; j <- 0 until grad(i).size) {
      val d = -alpha * grad(i)(j) + mt * PrevD(i)(j)
      value(i)(j) += d
      PrevD(i)(j) = d
    }
    machines
  }
}

object SGDMomentum {
  def NewSGDMomentum(c: Controller): SGDMomentum = {
    new SGDMomentum(
      C = c,
      // PrevD = new Array[Double](c.WeightsVal().size)
      PrevD = Controller.newControllerMemory(c)
    )
  }
}

// RMSProp implements the rmsprop algorithm. The detailed updating equations are given in
// Graves, Alex (2013). Generating sequences with recurrent neural networks. arXiv
// preprint arXiv:1308.0850.
class RMSProp(
  val C: Controller,
  val N: Array[Array[Double]],
  val G: Array[Array[Double]],
  val D: Array[Array[Double]]
){
  def Train(x: Array[Array[Double]], y: DensityModel, a: Double, b: Double, c: Double, d: Double): Array[NTM] = {
    val machines = NTM.ForwardBackward(C, x, y)
    update(a, b, c, d)
    machines
  }

  def update(a: Double, b: Double, c: Double, d: Double): Unit = {
    val grad = C.WeightsGradVec()
    val value = C.WeightsValVec()
    for(i <- 0 until grad.size; j <- 0 until grad(i).size) {
      val grad2i = grad(i)(j) * grad(i)(j)
      N(i)(j) = a * N(i)(j) + (1 - a) * grad2i
      G(i)(j) = a * G(i)(j) + (1 - a) * grad(i)(j)
      val rmsi = grad(i)(j) / Math.sqrt(N(i)(j) - G(i)(j) * G(i)(j) + d)
      D(i)(j) = b * D(i)(j) - c * rmsi
      value(i)(j) += D(i)(j)
    }
  }
}

object RMSProp {
  def NewRMSProp(c: Controller): RMSProp = {
    new RMSProp(
      C = c,
      // N = new Array[Double](c.WeightsVal().size),
      N = Controller.newControllerMemory(c),
      // G = new Array[Double](c.WeightsVal().size),
      G = Controller.newControllerMemory(c),
      // D = new Array[Double](c.WeightsVal().size)
      D = Controller.newControllerMemory(c)
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

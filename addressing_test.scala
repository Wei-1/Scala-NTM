package ntm

import math._

object addressing_test {
  // outputGradient is the gradient of all units at the output of a Circuit,
  // except for the output weights[1][1].
  // The gradient of weights[1][1], gw11, needs to be different from the other weights(i)(j),
  // because the value of the `addressing` function to be dependent on gw11.
  // If gw11 == gwij, then since weights(i)(j) always sum up to 1, gwij has no effect on `addressing`.
  val outputGradient    = 1.234
  val w11OutputGradient = 0.987

  def TestCircuit(t: T) {
    val t = new T

    val n = 3
    val m = 2
    val memory = new writtenMemory(
      N = n,
      TopVal = new Array[Double](n * m),
      TopGrad = new Array[Double](n * m),
    )
    for(i <- 0 until n) {
      for(j <- 0 until m) {
        memory.TopVal(i * m + j) = 0.001 * i * j + 0.1//Math.random
      }
    }
    val hul = Head.headUnitsLen(m)
    val heads = new Array[Head](2)
    for(i <- 0 until heads.size) {
      heads(i) = Head.NewHead(m)
      heads(i).vals = new Array[Double](hul)
      heads(i).grads = new Array[Double](hul)
      heads(i).Wtm1 = randomRefocus(n)
      for(j <- 0 until hul) {
        heads(i).vals(j) = 0.001 * i * j + 0.1//Math.random
      }
    }
    // We want to check the case where Beta > 0 and Gamma > 1.
    heads(0).vals(3 * heads(0).M) = 0.137350
    heads(0).vals(3 * heads(0).M + 3) = 1.9876

    val circuit = memOp.newMemOp(heads, memory)
    for(i <- 0 until circuit.W.size) {
      for(j <- 0 until circuit.W(i).TopGrad.size) {
        if(i == 0 && j == 0) {
          circuit.W(i).TopGrad(j) += w11OutputGradient
        } else {
          circuit.W(i).TopGrad(j) += outputGradient
        }
      }
    }
    for(i <- 0 until circuit.R.size) {
      for(j <- 0 until circuit.R(i).TopGrad.size) {
        circuit.R(i).TopGrad(j) += outputGradient
      }
    }
    for(i <- 0 until n) {
      for(j <- 0 until m) {
        circuit.WM.TopGrad(i * m + j) += outputGradient
      }
    }
    circuit.Backward()

    val memoryTop = unit.makeTensorUnit2(n, m)
    for(i <- 0 until n) {
      for(j <- 0 until m) {
        memoryTop(i)(j).Val = memory.TopVal(i * m + j)
        memoryTop(i)(j).Grad = memory.TopGrad(i * m + j)
      }
    }
    val ax = addressing(heads, memoryTop)
    checkGamma(t, heads, memoryTop, ax)
    checkS(t, heads, memoryTop, ax)
    checkG(t, heads, memoryTop, ax)
    checkWtm1(t, heads, memoryTop, ax)
    checkBeta(t, heads, memoryTop, ax)
    checkK(t, heads, memoryTop, ax)
    checkMemory(t, heads, memoryTop, ax)
  }

  def addressing(heads: Array[Head], memory: Array[Array[unit]]): Double = {
    val (weights, reads, newMem) = doAddressing(heads, memory)
    addressingLoss(weights, reads, newMem)
  }

  def doAddressing(heads: Array[Head], memory: Array[Array[unit]]): (Array[Array[Double]], Array[Array[Double]], Array[Array[Double]]) = {
    val weights = makeTensor2(heads.size, memory.size)
    for(i <- 0 until heads.size) {
      val h = heads(i)
      // Content-based addressing
      val beta = Math.exp(h.BetaVal())
      val wc = new Array[Double](memory.size)
      var sum: Double = 0
      for(j <- 0 until wc.size) {
        wc(j) = Math.exp(beta * cosineSimilarity(h.KVal(), unitVals(memory(j))))
        sum += wc(j)
      }
      for(j <- 0 until wc.size) {
        wc(j) = wc(j) / sum
      }

      // Content-based, location-based addressing gate
      val g = Sigmoid(h.GVal())
      for(j <- 0 until wc.size) {
        wc(j) = g * wc(j) + (1 - g) * h.Wtm1.TopVal(j)
      }

      // Location-based addressing
      val n = weights(i).size
      val s = (2 * Sigmoid(h.SVal()) - 1) - ((2 * Sigmoid(h.SVal()) - 1) / n).toInt * n
      for(j <- 0 until n) {
        val imj = (j + s.toInt) % n
        val simj = 1 - (s - Math.floor(s))
        weights(i)(j) = wc(imj)*simj + wc((imj + 1) % n) * (1 - simj)
      }

      // Refocusing
      val gamma = Math.log(Math.exp(h.GammaVal()) + 1) + 1
      sum = 0.0
      for(j <- 0 until weights(i).size) {
        weights(i)(j) = Math.pow(weights(i)(j), gamma)
        sum += weights(i)(j)
      }
      for(j <- 0 until weights(i).size) {
        weights(i)(j) = weights(i)(j) / sum
      }
    }

    val reads = makeTensor2(heads.size, memory(0).size)
    for(i <- 0 until weights.size) {
      val r = reads(i)
      for(j <- 0 until r.size) {
        for(k <- 0 until weights(i).size) {
          r(j) += weights(i)(k) * memory(k)(j).Val
        }
      }
    }

    val erase = makeTensor2(heads.size, memory(0).size)
    val add = makeTensor2(heads.size, memory(0).size)
    for(k <- 0 until heads.size) {
      val eraseVec = heads(k).EraseVal()
      for(i <- 0 until erase(k).size) {
        erase(k)(i) = Sigmoid(eraseVec(i))
      }
      val addVec = heads(k).AddVal()
      for(i <- 0 until add(k).size) {
        add(k)(i) = Sigmoid(addVec(i))
      }
    }
    val newMem = makeTensor2(memory.size, memory(0).size)
    for(i <- 0 until newMem.size) {
      for(j <- 0 until newMem(i).size) {
        newMem(i)(j) = memory(i)(j).Val
        for(k <- 0 until heads.size) {
          newMem(i)(j) = newMem(i)(j) * (1 - weights(k)(i)*erase(k)(j))
        }
        for(k <- 0 until heads.size) {
          newMem(i)(j) += weights(k)(i) * add(k)(j)
        }
      }
    }
    (weights, reads, newMem)
  }

  def addressingLoss(weights: Array[Array[Double]], reads: Array[Array[Double]], newMem: Array[Array[Double]]): Double = {
    var res: Double = 0
    for(i <- 0 until weights.size) {
      val w = weights(i)
      for(j <- 0 until w.size) {
        if(i == 0 && j == 0) {
          res += w(j) * w11OutputGradient
        } else {
          res += w(j) * outputGradient
        }
      }
    }
    for(r <- reads) {
      for(rr <- r) {
        res += rr * outputGradient
      }
    }
    for(i <- 0 until newMem.size) {
      for(j <- 0 until newMem(i).size) {
        res += newMem(i)(j) * outputGradient
      }
    }
    res
  }

  def checkMemory(t: T, heads: Array[Head], memory: Array[Array[unit]], ax: Double) {
    for(i <- 0 until memory.size) {
      for(j <- 0 until memory(i).size) {
        val x = memory(i)(j).Val
        val h = machineEpsilonSqrt * Math.max(Math.abs(x), 1)
        val xph = x + h
        memory(i)(j).Val = xph
        val dx = xph - x
        val axph = addressing(heads, memory)
        val grad = (axph - ax) / dx
        memory(i)(j).Val = x

        if(grad.isNaN || Math.abs(grad - memory(i)(j).Grad) > 1e-5) {
          t.Fatalf(s"[ADDRESS] wrong memory gradient expected $grad, got ${memory(i)(j).Grad}")
        } else {
          t.Logf(s"[ADDRESS] OK memory[$i][$j] gradient expected $grad, got ${memory(i)(j).Grad}")
        }
      }
    }
  }

  def checkK(t: T, heads: Array[Head], memory: Array[Array[unit]], ax: Double) {
    for(k <- 0 until heads.size) {
      val hd = heads(k)
      for(i <- 0 until hd.KVal().size) {
        val x = hd.KVal()(i)
        val h = machineEpsilonSqrt * Math.max(Math.abs(x), 1)
        val xph = x + h
        hd.KVal()(i) = xph
        val dx = xph - x
        val axph = addressing(heads, memory)
        val grad = (axph - ax) / dx
        hd.KVal()(i) = x

        if(grad.isNaN || Math.abs(grad - hd.KGrad()(i)) > 1e-5) {
          t.Fatalf(s"[ADDRESS] wrong beta[$i] gradient expected $grad, got ${hd.KGrad()(i)}")
        } else {
          t.Logf(s"[ADDRESS] OK K[$k][$i] gradient expected $grad, got ${hd.KGrad()(i)}")
        }
      }
    }
  }

  def checkBeta(t: T, heads: Array[Head], memory: Array[Array[unit]], ax: Double) {
    for(k <- 0 until heads.size) {
      val hd = heads(k)
      val x = hd.BetaVal()
      val h = machineEpsilonSqrt * Math.max(Math.abs(x), 1)
      val xph = x + h
      hd.vals(3 * hd.M) = xph
      val dx = xph - x
      val axph = addressing(heads, memory)
      val grad = (axph - ax) / dx
      hd.vals(3 * hd.M) = x

      if(grad.isNaN || Math.abs(grad - hd.BetaGrad()) > 1e-5) {
        t.Fatalf(s"[ADDRESS] wrong beta gradient expected $grad, got ${hd.BetaGrad()}")
      } else {
        t.Logf(s"[ADDRESS] OK beta[$k] gradient expected $grad, got ${hd.BetaGrad()}")
      }
    }
  }

  def checkWtm1(t: T, heads: Array[Head], memory: Array[Array[unit]], ax: Double) {
    for(k <- 0 until heads.size) {
      val hd = heads(k)
      for(i <- 0 until hd.Wtm1.TopVal.size) {
        val x = hd.Wtm1.TopVal(i)
        val h = machineEpsilonSqrt * Math.max(Math.abs(x), 1)
        val xph = x + h
        hd.Wtm1.TopVal(i) = xph
        val dx = xph - x
        val axph = addressing(heads, memory)
        val grad = (axph - ax) / dx
        hd.Wtm1.TopVal(i) = x

        if(grad.isNaN || Math.abs(grad - hd.Wtm1.TopGrad(i)) > 1e-5) {
          t.Fatalf(s"[ADDRESS] wrong wtm1[$i] gradient expected $grad, got ${hd.Wtm1.TopGrad(i)}")
        } else {
          t.Logf(s"[ADDRESS] OK wtm1[$k][$i] gradient expected $grad, got ${hd.Wtm1.TopGrad(i)}")
        }
      }
    }
  }

  def checkG(t: T, heads: Array[Head], memory: Array[Array[unit]], ax: Double) {
    for(k <- 0 until heads.size) {
      val hd = heads(k)
      val x = hd.GVal()
      val h = machineEpsilonSqrt * Math.max(Math.abs(x), 1)
      val xph = x + h
      hd.vals(3 * hd.M + 1) = xph
      val dx = xph - x
      val axph = addressing(heads, memory)
      val grad = (axph - ax) / dx
      hd.vals(3 * hd.M + 1) = x

      if(grad.isNaN || Math.abs(grad - hd.GGrad()) > 1e-5) {
        t.Fatalf(s"[ADDRESS] wrong G gradient expected $grad, got ${hd.GGrad()}")
      } else {
        t.Logf(s"[ADDRESS] OK G[$k] agradient expected $grad, got ${hd.GGrad()}")
      }
    }
  }

  def checkS(t: T, heads: Array[Head], memory: Array[Array[unit]], ax: Double) {
    for(k <- 0 until heads.size) {
      val hd = heads(k)
      val x = hd.SVal()
      val h = machineEpsilonSqrt * Math.max(x.abs, 1)
      val xph = x + h
      hd.vals(3 * hd.M + 2) = xph
      val dx = xph - x
      val axph = addressing(heads, memory)
      val grad = (axph - ax) / dx
      hd.vals(3 * hd.M + 2) = x

      if(grad.isNaN || Math.abs(grad - hd.SGrad()) > 1e-5) {
        t.Fatalf(s"[ADDRESS] wrong S gradient expected $grad, got ${hd.SGrad()}")
      } else {
        t.Logf(s"[ADDRESS] OK S[$k] agradient expected $grad, got ${hd.SGrad()}")
      }
    }
  }

  def checkGamma(t: T, heads: Array[Head], memory: Array[Array[unit]], ax: Double) {
    for(k <- 0 until heads.size) {
      val hd = heads(k)
      val x = hd.GammaVal()
      val h = machineEpsilonSqrt * Math.max(x.abs, 1)
      val xph = x + h
      hd.vals(3 * hd.M + 3) = xph
      val dx = xph - x
      val axph = addressing(heads, memory)
      val grad = (axph - ax) / dx
      hd.vals(3 * hd.M + 3) = x

      if(grad.isNaN || Math.abs(grad - hd.GammaGrad()) > 1e-5) {
        t.Fatalf(s"[ADDRESS] wrong gamma gradient expected $grad, got ${hd.GammaGrad()}")
      } else {
        t.Logf(s"[ADDRESS] OK gamma[$k] gradient expected $grad, got ${hd.GammaGrad()}")
      }
    }
  }

  def randomRefocus(n: Int): refocus = {
    val w = new Array[Double](n)
    var sum: Double = 0
    for(i <- 0 until w.size) {
      w(i) = Math.abs(0.01 * i + 0.1)//Math.random)
      sum += w(i)
    }
    for(i <- 0 until w.size) {
      w(i) = w(i) / sum
    }
    new refocus(
      TopVal = w,
      TopGrad = new Array[Double](n),
    )
  }

  def unitVals(units: Array[unit]): Array[Double] = {
    units.map(_.Val)
  }
}

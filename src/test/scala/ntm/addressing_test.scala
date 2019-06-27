package com.scalaml.ntm

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

    val n = 3
    val m = 2
    val memory = new writtenMemory(
      N = n,
      TopVal = Array.ofDim[Double](n, m),
      TopGrad = Array.ofDim[Double](n, m)
    )
    for(i <- 0 until n; j <- 0 until m) {
      memory.TopVal(i)(j) = Math.random
    }

    val hul = Head.headUnitsLen(m)
    val heads = new Array[Head](2)
    for(i <- 0 until heads.size) {
      heads(i) = Head.NewHead(m)
      heads(i).vals = new Array[Double](hul)
      heads(i).grads = new Array[Double](hul)
      heads(i).Wtm1 = randomRefocus(n)
      for(j <- 0 until hul) {
        heads(i).vals(j) = Math.random
      }
    }
    // We want to check the case where Beta > 0 and Gamma > 1.
    heads(0).setBetaVal(0.137350)
    heads(0).setGammaVal(1.9876)
    // println("memory TV " + memory.TopVal.mkString(","))

    val circuit = memOp.newMemOp(heads, memory)
    for(i <- 0 until circuit.W.size; j <- 0 until circuit.W(i).TopGrad.size) {
      if(i == 0 && j == 0) {
        circuit.W(i).TopGrad(j) += w11OutputGradient
      } else {
        circuit.W(i).TopGrad(j) += outputGradient
      }
    }
    for(i <- 0 until circuit.R.size) {
      for(j <- 0 until circuit.R(i).TopGrad.size) {
        circuit.R(i).TopGrad(j) += outputGradient
      }
    }
    for(i <- 0 until n; j <- 0 until m) {
      circuit.WM.TopGrad(i)(j) += outputGradient
    }

    circuit.Backward() // Major Computation Here

    val memoryTop = unit.makeTensorUnit2(n, m)
    for(i <- 0 until n; j <- 0 until m) {
      memoryTop(i)(j).Val = memory.TopVal(i)(j)
      memoryTop(i)(j).Grad = memory.TopGrad(i)(j)
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
      val beta = Math.exp(h.getBetaVal())
      val wc = new Array[Double](memory.size)
      val kVal = (0 until h.M).map(i => h.getKVal(i)).toArray
      var sum: Double = 0
      for(j <- 0 until wc.size) {
        wc(j) = Math.exp(beta * cosineSimilarity(kVal, unitVals(memory(j))))
        sum += wc(j)
      }
      for(j <- 0 until wc.size) {
        wc(j) = wc(j) / sum
      }

      // Content-based, location-based addressing gate
      val g = Sigmoid(h.getGVal())
      for(j <- 0 until wc.size) {
        wc(j) = g * wc(j) + (1 - g) * h.Wtm1.TopVal(j)
      }

      // Location-based addressing
      val n = weights(i).size
      val s = (2 * Sigmoid(h.getSVal()) - 1) - ((2 * Sigmoid(h.getSVal()) - 1) / n).toInt * n
      for(j <- 0 until n) {
        val imj = (j + s.toInt) % n
        val simj = 1 - (s - Math.floor(s))
        weights(i)(j) = wc(imj)*simj + wc((imj + 1) % n) * (1 - simj)
      }

      // Refocusing
      val gamma = Math.log(Math.exp(h.getGammaVal()) + 1) + 1
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
      val head = heads(k)
      for(i <- 0 until erase(k).size) {
        erase(k)(i) = Sigmoid(head.getEraseVal(i))
      }
      for(i <- 0 until add(k).size) {
        add(k)(i) = Sigmoid(head.getAddVal(i))
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
    for(i <- 0 until memory.size; j <- 0 until memory(i).size) {
      val x = memory(i)(j).Val
      val h = machineEpsilonSqrt * Math.max(Math.abs(x), 1)
      val xph = x + h
      memory(i)(j).Val = xph
      val dx = xph - x
      val axph = addressing(heads, memory)
      val grad = (axph - ax) / dx
      memory(i)(j).Val = x

      if(grad.isNaN || Math.abs(grad - memory(i)(j).Grad) > 1e-5) {
        t.Fatalf(s"[ADDRESS] wrong memory[$i][$j] gradient expected $grad, got ${memory(i)(j).Grad}")
      } else {
        t.Logf(s"[ADDRESS] OK memory[$i][$j] gradient expected $grad, got ${memory(i)(j).Grad}")
      }
    }
  }

  def checkK(t: T, heads: Array[Head], memory: Array[Array[unit]], ax: Double) {
    for(k <- 0 until heads.size) {
      val hd = heads(k)
      for(i <- 0 until hd.M) {
        val x = hd.getKVal(i)
        val h = machineEpsilonSqrt * Math.max(Math.abs(x), 1)
        val xph = x + h
        hd.setKVal(i, xph)
        val dx = xph - x
        val axph = addressing(heads, memory)
        val grad = (axph - ax) / dx
        hd.setKVal(i, x)

        if(grad.isNaN || Math.abs(grad - hd.getKGrad(i)) > 1e-5) {
          t.Fatalf(s"[ADDRESS] wrong beta[$k][$i] gradient expected $grad, got ${hd.getKGrad(i)}")
        } else {
          t.Logf(s"[ADDRESS] OK K[$k][$i] gradient expected $grad, got ${hd.getKGrad(i)}")
        }
      }
    }
  }

  def checkBeta(t: T, heads: Array[Head], memory: Array[Array[unit]], ax: Double) {
    for(k <- 0 until heads.size) {
      val hd = heads(k)
      val x = hd.getBetaVal()
      val h = machineEpsilonSqrt * Math.max(Math.abs(x), 1)
      val xph = x + h
      hd.setBetaVal(xph)
      val dx = xph - x
      val axph = addressing(heads, memory)
      val grad = (axph - ax) / dx
      hd.setBetaVal(x)

      if(grad.isNaN || Math.abs(grad - hd.getBetaGrad()) > 1e-5) {
        t.Fatalf(s"[ADDRESS] wrong beta[$k] gradient expected $grad, got ${hd.getBetaGrad()}")
      } else {
        t.Logf(s"[ADDRESS] OK beta[$k] gradient expected $grad, got ${hd.getBetaGrad()}")
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
          t.Fatalf(s"[ADDRESS] wrong wtm1[$k][$i] gradient expected $grad, got ${hd.Wtm1.TopGrad(i)}")
        } else {
          t.Logf(s"[ADDRESS] OK wtm1[$k][$i] gradient expected $grad, got ${hd.Wtm1.TopGrad(i)}")
        }
      }
    }
  }

  def checkG(t: T, heads: Array[Head], memory: Array[Array[unit]], ax: Double) {
    for(k <- 0 until heads.size) {
      val hd = heads(k)
      val x = hd.getGVal()
      val h = machineEpsilonSqrt * Math.max(Math.abs(x), 1)
      val xph = x + h
      hd.setGVal(xph)
      val dx = xph - x
      val axph = addressing(heads, memory)
      val grad = (axph - ax) / dx
      hd.setGVal(x)

      if(grad.isNaN || Math.abs(grad - hd.getGGrad()) > 1e-5) {
        t.Fatalf(s"[ADDRESS] wrong G[$k] gradient expected $grad, got ${hd.getGGrad()}")
      } else {
        t.Logf(s"[ADDRESS] OK G[$k] agradient expected $grad, got ${hd.getGGrad()}")
      }
    }
  }

  def checkS(t: T, heads: Array[Head], memory: Array[Array[unit]], ax: Double) {
    for(k <- 0 until heads.size) {
      val hd = heads(k)
      val x = hd.getSVal()
      val h = machineEpsilonSqrt * Math.max(x.abs, 1)
      val xph = x + h
      hd.setSVal(xph)
      val dx = xph - x
      val axph = addressing(heads, memory)
      val grad = (axph - ax) / dx
      hd.setSVal(x)

      if(grad.isNaN || Math.abs(grad - hd.getSGrad()) > 1e-5) {
        t.Fatalf(s"[ADDRESS] wrong S[$k] gradient expected $grad, got ${hd.getSGrad()}")
      } else {
        t.Logf(s"[ADDRESS] OK S[$k] agradient expected $grad, got ${hd.getSGrad()}")
      }
    }
  }

  def checkGamma(t: T, heads: Array[Head], memory: Array[Array[unit]], ax: Double) {
    for(k <- 0 until heads.size) {
      val hd = heads(k)
      val x = hd.getGammaVal()
      val h = machineEpsilonSqrt * Math.max(x.abs, 1)
      val xph = x + h
      hd.setGammaVal(xph)
      val dx = xph - x
      val axph = addressing(heads, memory)
      val grad = (axph - ax) / dx
      hd.setGammaVal(x)

      if(grad.isNaN || Math.abs(grad - hd.getGammaGrad()) > 1e-5) {
        t.Fatalf(s"[ADDRESS] wrong gamma[$k] gradient expected $grad, got ${hd.getGammaGrad()}")
      } else {
        t.Logf(s"[ADDRESS] OK gamma[$k] gradient expected $grad, got ${hd.getGammaGrad()}")
      }
    }
  }

  def randomRefocus(n: Int): refocus = {
    val w = new Array[Double](n)
    var sum: Double = 0
    for(i <- 0 until w.size) {
      w(i) = Math.random
      sum += w(i)
    }
    for(i <- 0 until w.size) {
      w(i) = w(i) / sum
    }
    new refocus(
      TopVal = w,
      TopGrad = new Array[Double](n)
    )
  }

  def unitVals(units: Array[unit]): Array[Double] = {
    units.map(_.Val)
  }
}

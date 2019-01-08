package ntm

import math._

class similarityCircuit(
  val UVal: Array[Double],
  val UGrad: Array[Double],
  val VVal: Array[Double],
  val VGrad: Array[Double]
){
  var UV: Double = UVal.zip(VVal).map { case (ui, vi) => ui * vi }.sum
  var Unorm: Double = Math.sqrt(UVal.map(v => v * v).sum)
  var Vnorm: Double = Math.sqrt(VVal.map(v => v * v).sum)
  var TopVal: Double = UV / (Unorm * Vnorm)
  var TopGrad: Double = 0.0

  def Backward(): Unit = {
    val uvuu = UV / (Unorm * Unorm)
    val uvvv = UV / (Vnorm * Vnorm)
    val uvg = TopGrad / (Unorm * Vnorm)
    for(i <- 0 until UGrad.size) {
      UGrad(i) += uvg * VVal(i) - uvuu*uvg * UVal(i)
      VGrad(i) += uvg * UVal(i) - uvuu*uvg * VVal(i)
    }
  }
}

object similarityCircuit {

  def newSimilarityCircuit(
    uVal: Array[Double],
    uGrad: Array[Double],
    vVal: Array[Double],
    vGrad: Array[Double]
  ): similarityCircuit = {
    new similarityCircuit(
      UVal = uVal,
      UGrad = uGrad,
      VVal = vVal,
      VGrad = vGrad
    )
  }
}

class betaSimilarity(
  var BetaVal: Double = 0.0,
  var BetaGrad: Double = 0.0,
  var S: similarityCircuit = null,
  var b: Double = 0.0,
  var Top: unit = new unit
){
  Top.Val = b * S.TopVal
  def Backward() {
    BetaGrad += S.TopVal * b * Top.Grad
    S.TopGrad += b * Top.Grad
  }
}

object betaSimilarity {
  def newBetaSimilarity(
    betaVal: Double,
    betaGrad: Double,
    s: similarityCircuit
  ): betaSimilarity = {
    new betaSimilarity(
      BetaVal = betaVal,
      BetaGrad = betaGrad,
      S = s,
      // Beta is in the range (-Inf, Inf)
      b = Math.exp(betaVal)
    )
  }
}

class contentAddressing(
  val Units: Array[betaSimilarity],
  val Top: Array[unit]
){
  def Backward(): Unit = {
    var gv: Double = 0
    for(top <- Top) {
      gv += top.Grad * top.Val
    }
    for(i <- 0 until Top.size) {
      Units(i).Top.Grad += (Top(i).Grad - gv) * Top(i).Val
    }
  }
}

object contentAddressing {
  def newContentAddressing(units: Array[betaSimilarity]): contentAddressing = {
    val s = new contentAddressing(
      Units = units,
      Top = Array.fill[unit](units.size)(new unit)
    )
    // Increase numerical stability by subtracting all weights by their max,
    // before computing Math.exp().
    var max: Double = Double.MinValue
    for(u <- s.Units) {
      max = Math.max(max, u.Top.Val)
    }
    var sum: Double = 0
    for(i <- 0 until s.Units.size) {
      val w = Math.exp(s.Units(i).Top.Val - max)
      s.Top(i).Val = w
      sum += w
    }
    for(i <- 0 until s.Top.size) {
      s.Top(i).Val = s.Top(i).Val / sum
    }
    s
  }
}

class gatedWeighting(
  var GVal: Double = 0.0,
  var GGrad: Double = 0.0,
  val WC: contentAddressing,
  // the weights at time t-1
  val Wtm1: refocus,
  val Top: Array[unit]
){
  def Backward(): Unit = {
    val gt = Sigmoid(GVal)

    var grad: Double = 0.0
    for(i <- 0 until Top.size) {
      grad += (WC.Top(i).Val - Wtm1.TopVal(i)) * Top(i).Grad
    }
    GGrad += grad * gt * (1 - gt)

    for(i <- 0 until WC.Top.size) {
      WC.Top(i).Grad += gt * Top(i).Grad
    }

    for(i <- 0 until Wtm1.TopGrad.size) {
      Wtm1.TopGrad(i) += (1 - gt) * Top(i).Grad
    }
  }
}

object gatedWeighting {
  def newGatedWeighting(
    gVal: Double,
    gGrad: Double,
    wc: contentAddressing,
    wtm1: refocus
  ): gatedWeighting = {
    val wg = new gatedWeighting(
      GVal = gVal,
      GGrad = gGrad,
      WC = wc,
      Wtm1 = wtm1,
      Top = Array.fill[unit](wc.Top.size)(new unit),
    )
    val gt = Sigmoid(gVal)
    for(i <- 0 until wg.Top.size) {
      wg.Top(i).Val = gt*wc.Top(i).Val + (1-gt)*wtm1.TopVal(i)
    }
    wg
  }
}

class shiftedWeighting(
  var SVal: Double = 0.0,
  var SGrad: Double = 0.0,
  var Z: Double = 0.0,
  val WG: gatedWeighting,
  val Top: Array[unit]
){
  def Backward(): Unit = {
    var grad: Double = 0
    val n = WG.Top.size
    for(i <- 0 until Top.size) {
      val imj = (i + Z.toInt) % n
      grad += (-WG.Top(imj).Val + WG.Top((imj + 1) % n).Val) * Top(i).Grad
    }
    val sig = Sigmoid(SVal)
    grad = grad * 2 * sig * (1 - sig)
    SGrad += grad

    val simj = 1 - (Z - Math.floor(Z))
    for(i <- 0 until WG.Top.size) {
      val j = (i - Z.toInt + n) % n
      WG.Top(i).Grad += Top(j).Grad * simj + Top((j - 1 + n) % n).Grad * (1 - simj)
    }
  }
}

object shiftedWeighting {
  def newShiftedWeighting(sVal: Double, sGrad: Double, wg: gatedWeighting): shiftedWeighting = {
    val sw = new shiftedWeighting(
      SVal = sVal,
      SGrad = sGrad,
      WG = wg,
      Top = Array.fill[unit](wg.Top.size)(new unit)
    )

    val n = sw.WG.Top.size

    val shift = (2 * Sigmoid(sVal) - 1)
    sw.Z = shift - (shift / n.toDouble).toInt * n

    val simj = 1 - (sw.Z - Math.floor(sw.Z))
    for(i <- 0 until sw.Top.size) {
      val imj = (i + sw.Z.toInt) % n
      sw.Top(i).Val = sw.WG.Top(imj).Val * simj + sw.WG.Top((imj + 1) % n).Val * (1 - simj)
      if(sw.Top(i).Val.isNaN || sw.Top(i).Val < 0) {
        Console.err.println(s"imj: $imj, wg: $sw.WG.Top(imj).Val, simj: $simj, wg+1: $sw.WG.Top((imj + 1) % n).Val")
      }
    }
    sw
  }
}

class refocus(
  var GammaVal: Double = 0,
  var GammaGrad: Double = 0,
  var SW: shiftedWeighting = null,
  val TopVal: Array[Double],
  val TopGrad: Array[Double],
  var g: Double = 0
){
  def backwardSW() {
    var topGV: Double = 0
    for(i <- 0 until TopVal.size) {
      topGV += TopGrad(i) * TopVal(i)
    }
    for(i <- 0 until SW.Top.size if SW.Top(i).Val >= machineEpsilon) {
      SW.Top(i).Grad += (TopGrad(i) - topGV) * g / SW.Top(i).Val * TopVal(i)
    }
  }

  def backwardGamma() {
    val lns = new Array[Double](SW.Top.size)
    var lnexp: Double = 0
    var s: Double = 0
    for(i <- 0 until SW.Top.size if SW.Top(i).Val >= machineEpsilon) {
      lns(i) = Math.log(SW.Top(i).Val)
      val pow = Math.pow(SW.Top(i).Val, g)
      lnexp += lns(i) * pow
      s += pow
    }
    val lnexps = lnexp / s
    var grad: Double = 0
    for(i <- 0 until TopVal.size if SW.Top(i).Val >= machineEpsilon) {
      grad += TopGrad(i) * (TopVal(i) * (lns(i) - lnexps))
    }
    grad = grad / (1 + Math.exp(-(GammaVal)))
    GammaGrad += grad
  }

  def Backward() {
    backwardSW()
    backwardGamma()
  }
}

object refocus {
  def newRefocus(gammaVal: Double, gammaGrad: Double, sw: shiftedWeighting): refocus = {
    val rf = new refocus(
      GammaVal = gammaVal,
      GammaGrad = gammaGrad,
      SW = sw,
      TopVal = new Array[Double](sw.Top.size),
      TopGrad = new Array[Double](sw.Top.size),
      g = Math.log(Math.exp(gammaVal) + 1) + 1
    )
    var sum: Double = 0
    for(i <- 0 until rf.TopVal.size) {
      rf.TopVal(i) = Math.pow(sw.Top(i).Val, rf.g)
      sum += rf.TopVal(i)
    }
    for(i <- 0 until rf.TopVal.size) {
      rf.TopVal(i) = rf.TopVal(i) / sum
    }
    rf
  }
}

class memRead(
  val W: refocus,
  val Memory: writtenMemory,
  val TopVal: Array[Double],
  val TopGrad: Array[Double]
){
  def Backward() {
    val n = Memory.N
    val m = Memory.TopVal.size / n
    // grad.size == m
    val grad = TopGrad
    val memVal = Memory.TopVal
    val weightsGrad = W.TopGrad
    // Gemv(t Trans, alpha f64, A General, x Vec, beta f64, y Vec)
    // y = alpha * A * x + beta * y; if t == blas.NoTrans
    for(k <- 0 until m; j <- 0 until n)
      weightsGrad(j) -= memVal(k * n + j) * grad(k)
    val memGrad = Memory.TopGrad
    val weights = W.TopVal
    // Ger(alpha f64, x, y Vec, A General)
    // A += alpha * x * y^T
    for(i <- 0 until m; j <- 0 until n) {
      memGrad(i * n + j) += weights(j) * grad(i)
    }
  }
}

object memRead {
  def newMemRead(w: refocus, memory: writtenMemory): memRead = {
    val n = memory.N
    val m = memory.TopVal.size / n
    val r = new memRead(
      W = w,
      Memory = memory,
      TopVal = new Array[Double](m),
      TopGrad = new Array[Double](m)
    )
    val weights = w.TopVal
    val mem = memory.TopVal
    val top = r.TopVal
    for(i <- 0 until m; j <- 0 until n) {
      top(i) += mem(i * n + j) * weights(j)
    }
    r
  }
}

class writtenMemory(
  var Ws: Array[refocus] = null,
  // We actually need only the erase and add vectors.
  var Heads: Array[Head] = null,
  // memory at time t-1
  var Mtm1: writtenMemory = null,
  // memoryN
  val N: Int,
  val TopVal: Array[Double],
  val TopGrad: Array[Double],
  var erase: Array[Array[Double]] = null,
  var add: Array[Array[Double]] = null,
  var erasures: Array[Double] = null
){
  def div1MWE(out: Array[Double]) {
    val m = TopVal.size / N
    for(i <- 0 until erasures.size) {
      val mwe = 1 - out(i)
      if(mwe.abs > 1e-6) {
        out(i) = erasures(i) / mwe
      } else {
        val j = i / m
        val k = i % m
        var mtilt = Mtm1.TopVal(j * m + k)
        for(q <- 0 until Ws.size if q != i) {
          mtilt *= (1 - Ws(q).TopVal(j) * erase(q)(k))
        }
        out(i) = mtilt
      }
    }
  }

  def backwardWErase() {
    val n = N
    val m = TopVal.size / n

    val mgrad = new Array[Double](n * m)
    val hEraseGrad = new Array[Double](m)
    for(i <- 0 until Ws.size) {
      val eraseV = erase(i)
      val addV = add(i)
      val weightsVal = Ws(i).TopVal
      for(j <- 0 until n * m) {
        mgrad(j) = 0
      }
      // Ger(alpha f64, x, y Vec, A General)
      // A += alpha * x * y^T
      for(k <- 0 until m; j <- 0 until n)
        mgrad(k * n + j) += weightsVal(j) * eraseV(k)
      div1MWE(mgrad)
      for(j <- 0 until n * m)
        mgrad(j) *= TopGrad(j)

      val weightsV = Ws(i).TopGrad
      // Gemv(t Trans, alpha f64, A General, x Vec, beta f64, y Vec)
      // y = alpha * A * x + beta * y; if t == blas.NoTrans
      for(k <- 0 until m; j <- 0 until n)
        weightsV(j) -= mgrad(k * n + j) * eraseV(k)
      for(k <- 0 until m; j <- 0 until n)
        weightsV(j) += TopGrad(k * n + j) * addV(k)
      
      val hErase = Heads(i).EraseGrad()
      for(j <- 0 until hEraseGrad.size)
        hEraseGrad(j) = 0

      for(k <- 0 until m; j <- 0 until n)
        hEraseGrad(k) -= TopGrad(k * n + j) * weightsVal(k)

      for(j <- 0 until eraseV.size) {
        hErase(j) += hEraseGrad(j) * eraseV(j) * (1 - eraseV(j))
      }
    }
  }

  def backwardAdd() {
    val n = N
    val m = TopVal.size / n
    var grad: Double = 0
    for(k <- 0 until Heads.size) {
      val addV = add(k)
      val ws = Ws(k)
      val hAdd = Heads(k).AddGrad()
      for(i <- 0 until hAdd.size) {
        grad = 0
        for(j <- 0 until n) {
          grad += TopGrad(j * m + i) * ws.TopVal(j)
        }
        val a = addV(i)
        hAdd(i) += grad * a * (1 - a)
      }
    }
  }

  def backwardMtm1() {
    val n = N
    val m = TopVal.size / n
    var grad: Double = 0
    for(i <- 0 until n) {
      for(j <- 0 until m) {
        grad = 1
        for(q <- 0 until Ws.size) {
          grad *= (1 - Ws(q).TopVal(i) * erase(q)(j))
        }
        Mtm1.TopGrad(i * m + j) += grad * TopGrad(i * m + j)
      }
    }
  }

  def Backward() {
    backwardWErase()
    backwardAdd()
    backwardMtm1()
  }
}

object writtenMemory {
  def newWrittenMemory(ws: Array[refocus], heads: Array[Head], mtm1: writtenMemory): writtenMemory = {
    val n = mtm1.N
    val mtn = mtm1.TopVal.size
    val m = mtn / n
    val wm = new writtenMemory(
      Ws = ws,
      Heads = heads,
      Mtm1 = mtm1,
      N = mtm1.N,
      TopVal = new Array[Double](mtn),
      TopGrad = new Array[Double](mtn),
      erase = Array.ofDim[Double](m, m),
      add = Array.ofDim[Double](m, m),
      erasures = new Array[Double](mtn)
    )
    for(i <- 0 until m) {
      val erase = wm.erase(i)
      val add = wm.add(i)
      val addVec = wm.Heads(i).AddVal()
      for(j <- 0 until m) {
        erase(j) = Sigmoid(wm.Heads(i).EraseVal()(j))
        add(j) = Sigmoid(addVec(j))
      }
    }

    for(i <- 0 until mtn) wm.erasures(i) = mtm1.TopVal(i)
    val we = Array.fill[Double](mtn)(1.0)
    for(k <- 0 until m) {
      val weights = wm.Ws(k).TopVal
      val erase = wm.erase(k)
      // Ger(alpha f64, x, y Vec, A General)
      // A += alpha * x * y^T
      for(i <- 0 until m; j <- 0 until n)
        we(i * n + j) += weights(j) * erase(i)
      for(i <- 0 until mtn)
        wm.erasures(i) *= we(i)
    }

    for(i <- 0 until mtn) wm.TopVal(i) = wm.erasures(i)

    val topG = wm.TopVal
    for(k <- 0 until wm.Ws.size) {
      val weights = wm.Ws(k).TopVal
      val add = wm.add(k)
      for(i <- 0 until m; j <- 0 until n)
        topG(i * n + j) += weights(j) * add(i)
    }

    wm
  }
}

class memOp(
  val R: Array[memRead],
  var W: Array[refocus] = null,
  var WM: writtenMemory = null
) {
  def Backward() {
    R.foreach(r => r.Backward())
    WM.Backward()
    WM.Ws.foreach { rf =>
      rf.Backward()
      rf.SW.Backward()
      rf.SW.WG.Backward()
      rf.SW.WG.WC.Backward()
      rf.SW.WG.WC.Units.foreach { bs =>
        bs.Backward()
        bs.S.Backward()
      }
    }
  }
}

object memOp {
  def newMemOp(heads: Array[Head], mtm1: writtenMemory): memOp = {
    val circuit = new memOp(
      R = new Array[memRead](heads.size)
    )
    circuit.W = new Array[refocus](heads.size)
    for(wi <- 0 until heads.size) {
      val h = heads(wi)
      val ss = new Array[betaSimilarity](mtm1.N)
      for(i <- 0 until mtm1.N) {
        val m = mtm1.TopVal.size / mtm1.N
        val s = similarityCircuit.newSimilarityCircuit(h.KVal(), h.KGrad(), mtm1.TopVal.drop(i * m).take(m), mtm1.TopGrad.drop(i * m).take(m))
        ss(i) = betaSimilarity.newBetaSimilarity(h.BetaVal(), h.BetaGrad(), s)
      }
      val wc = contentAddressing.newContentAddressing(ss)
      val wg = gatedWeighting.newGatedWeighting(h.GVal(), h.GGrad(), wc, h.Wtm1)
      val ws = shiftedWeighting.newShiftedWeighting(h.SVal(), h.SGrad(), wg)
      circuit.W(wi) = refocus.newRefocus(h.GammaVal(), h.GammaGrad(), ws)
      circuit.R(wi) = memRead.newMemRead(circuit.W(wi), mtm1)
    }
    circuit.WM = writtenMemory.newWrittenMemory(circuit.W, heads, mtm1)
    circuit
  }
}

package ntm

import math._

class similarityCircuit(
  val UVal: Array[Double],
  val UGrad: Array[Double],
  val VVal: Array[Double],
  val VGrad: Array[Double],
  val index: Int,
  val m: Int
){
  val VVali = VVal.drop(index * m).take(m)
  var UV: Double = UVal.zip(VVali).map { case (ui, vi) => ui * vi }.sum
  var Unorm: Double = Math.sqrt(UVal.map(v => v * v).sum)
  var Vnorm: Double = Math.sqrt(VVali.map(v => v * v).sum)
  var TopVal: Double = UV / (Unorm * Vnorm)
  var TopGrad: Double = 0.0

  def Backward(): Unit = {
    val uvuu = UV / (Unorm * Unorm)
    val uvvv = UV / (Vnorm * Vnorm)
    val uvg = TopGrad / (Unorm * Vnorm)
    // println("uvuu " + uvuu + " uvvv " + uvvv + " uvg " + uvg)
    for(i <- 0 until UGrad.size) {
      UGrad(i) += uvg * VVali(i) - uvuu * uvg * UVal(i)
      VGrad(index * m + i) += uvg * UVal(i) - uvvv * uvg * VVali(i)
    }
    // println("~~~ " + VGrad.mkString(","))
  }
}

object similarityCircuit {
  def newSimilarityCircuit(
    uVal: Array[Double],
    uGrad: Array[Double],
    vVal: Array[Double],
    vGrad: Array[Double],
    index: Int,
    m: Int
  ): similarityCircuit = {
    new similarityCircuit(
      UVal = uVal,
      UGrad = uGrad,
      VVal = vVal,
      VGrad = vGrad,
      index = index,
      m = m
    )
  }
}

class betaSimilarity(
  var h: Head = null,
  var S: similarityCircuit = null,
  var b: Double = 0.0,
  var Top: unit = new unit
){
  def BetaVal: Double = h.BetaVal()
  def BetaGrad: Double = h.BetaGrad()
  if(S != null) Top.Val = b * S.TopVal
  def Backward() {
    h.grads(3 * h.M) += S.TopVal * b * Top.Grad
    S.TopGrad += b * Top.Grad
  }
}

object betaSimilarity {
  def newBetaSimilarity(
    h: Head,
    s: similarityCircuit
  ): betaSimilarity = {
    new betaSimilarity(
      h = h,
      S = s,
      // Beta is in the range (-Inf, Inf)
      b = Math.exp(h.BetaVal)
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
  var h: Head = null,
  val WC: contentAddressing,
  // the weights at time t-1
  val Wtm1: refocus,
  val Top: Array[unit]
){
  def GVal: Double = h.GVal()
  def GGrad: Double = h.GGrad()
  def Backward(): Unit = {
    val gt = Sigmoid(GVal)

    var grad: Double = 0.0
    for(i <- 0 until Top.size) {
      grad += (WC.Top(i).Val - Wtm1.TopVal(i)) * Top(i).Grad
    }
    h.grads(3 * h.M + 1) += grad * gt * (1 - gt)

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
    h: Head,
    wc: contentAddressing,
    wtm1: refocus
  ): gatedWeighting = {
    val wg = new gatedWeighting(
      h = h,
      WC = wc,
      Wtm1 = wtm1,
      Top = Array.fill[unit](wc.Top.size)(new unit),
    )
    val gt = Sigmoid(h.GVal)
    for(i <- 0 until wg.Top.size) {
      wg.Top(i).Val = gt*wc.Top(i).Val + (1-gt)*wtm1.TopVal(i)
    }
    wg
  }
}

class shiftedWeighting(
  var h: Head = null,
  var Z: Double = 0.0,
  val WG: gatedWeighting,
  val Top: Array[unit]
){
  def SVal: Double = h.SVal()
  def SGrad: Double = h.SGrad()
  def Backward(): Unit = {
    var grad: Double = 0
    val n = WG.Top.size
    for(i <- 0 until Top.size) {
      val imj = (i + Z.toInt) % n
      grad += (-WG.Top(imj).Val + WG.Top((imj + 1) % n).Val) * Top(i).Grad
    }
    val sig = Sigmoid(SVal)
    grad = grad * 2 * sig * (1 - sig)
    h.grads(3 * h.M + 2) += grad

    val simj = 1 - (Z - Math.floor(Z))
    for(i <- 0 until WG.Top.size) {
      val j = (i - Z.toInt + n) % n
      WG.Top(i).Grad += Top(j).Grad * simj + Top((j - 1 + n) % n).Grad * (1 - simj)
    }
  }
}

object shiftedWeighting {
  def newShiftedWeighting(h: Head, wg: gatedWeighting): shiftedWeighting = {
    val sw = new shiftedWeighting(
      h = h,
      WG = wg,
      Top = Array.fill[unit](wg.Top.size)(new unit)
    )

    val n = sw.WG.Top.size

    val shift = (2 * Sigmoid(h.SVal) - 1)
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
  var h: Head = null,
  var SW: shiftedWeighting = null,
  val TopVal: Array[Double],
  val TopGrad: Array[Double],
  var g: Double = 0
){
  // println("----" + TopGrad.mkString(","))
  def GammaVal: Double = h.GammaVal()
  def GammaGrad: Double = h.GammaGrad()
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
    // println("----" + TopGrad.mkString(","))
    val lns = new Array[Double](SW.Top.size)
    var lnexp: Double = 0
    var s: Double = 0
    for(i <- 0 until SW.Top.size if SW.Top(i).Val >= machineEpsilon) {
      lns(i) = Math.log(SW.Top(i).Val)
      val pow = Math.pow(SW.Top(i).Val, g)
      lnexp += lns(i) * pow
      s += pow
    }
    // Console.err.println(" - s " + s)
    // Console.err.println(" - lnexp " + lnexp)
    val lnexps = lnexp / s
    var grad: Double = 0
    for(i <- 0 until TopVal.size if SW.Top(i).Val >= machineEpsilon) {
      // println(" - TopVal " + TopVal(i)) //  <---- Correct
      // println(" - TopGrad " + TopGrad(i)) //  <---- Error
      grad += TopGrad(i) * (TopVal(i) * (lns(i) - lnexps))
    }
    // println(" - grad " + grad)
    grad = grad / (1 + Math.exp(-GammaVal))
    h.grads(3 * h.M + 3) += grad
  }

  def Backward() {
    // println("----" + TopGrad.mkString(","))
    backwardSW()
    // println("----" + TopGrad.mkString(","))
    backwardGamma()
    // println("----" + TopGrad.mkString(","))
  }
}

object refocus {
  def newRefocus(h: Head, sw: shiftedWeighting): refocus = {
    val rf = new refocus(
      h = h,
      SW = sw,
      TopVal = new Array[Double](sw.Top.size),
      TopGrad = new Array[Double](sw.Top.size),
      g = Math.log(Math.exp(h.GammaVal()) + 1) + 1
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
    // println(" memory TopVal " + memVal.mkString(","))
    // Gemv(t Trans, alpha f64, A General, x Vec, beta f64, y Vec)
    // y = alpha * A * x + beta * y; if t == blas.NoTrans
    // println(" refocus TopGrad " + W.TopGrad.mkString(","))

    // println("  - memVal " + memVal.mkString(","))
    // println("  - grad " + grad.mkString(","))

    // for(k <- 0 until m; j <- 0 until n) {
    //   weightsGrad(j) += memVal(j * m + k) * grad(k)
    // }
    Gemv(false, 1, memVal, grad, 1, weightsGrad)

    // println(" refocus TopGrad " + W.TopGrad.mkString(","))

    val memGrad = Memory.TopGrad
    val weights = W.TopVal
    // Ger(alpha f64, x, y Vec, A General)
    // A += alpha * x * y^T
    // println(memGrad.mkString(","))
    // for(i <- 0 until m; j <- 0 until n) {
    //   memGrad(j * m + i) += weights(j) * grad(i)
    // }
    Ger(1, weights, grad, memGrad)
    // println(memGrad.mkString(","))
    // println(" refocus TopGrad " + W.TopGrad.mkString(","))
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
    Gemv(true, 1, mem, weights, 1, top)
    // for(i <- 0 until m; j <- 0 until n) {
    //   top(i) += mem(i * n + j) * weights(j)
    // }
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
        // print(mtilt + ",")
        out(i) = mtilt
      }
    }
    // println("--------- ----")
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
      // for(k <- 0 until m; j <- 0 until n)
      //   mgrad(k * n + j) += weightsVal(j) * eraseV(k)
      Ger(1, weightsVal, eraseV, mgrad)
      // println("-erasure- " + erasures.mkString(","))
      // println("-mgrad 0- " + mgrad.mkString(","))
      div1MWE(mgrad) // <---- FIXED
      // println("-mgrad 1- " + mgrad.mkString(","))

      for(j <- 0 until n * m)
        mgrad(j) *= TopGrad(j)

      val weightsV = Ws(i).TopGrad

      // println("-#- " + weightsVal.mkString(","))
      // println("-!2- " + mgrad.mkString(","))
      // println("-@- " + eraseV.mkString(","))
      // Gemv(t Trans, alpha f64, A General, x Vec, beta f64, y Vec)
      // y = alpha * A * x + beta * y; if t == blas.NoTrans
      // println("-~- " + weightsV.mkString(","))
      // for(k <- 0 until m; j <- 0 until n)
      //   weightsV(j) -= mgrad(j * m + k) * eraseV(k)
      Gemv(false, -1, mgrad, eraseV, 1, weightsV)
      // println("-~- " + weightsV.mkString(","))
      // for(k <- 0 until m; j <- 0 until n)
      //   weightsV(j) += TopGrad(j * m + k) * addV(k)
      Gemv(false, 1, TopGrad, addV, 1, weightsV)
      // println("-~- " + weightsV.mkString(","))
      val hErase = Heads(i).EraseGrad()
      for(j <- 0 until hEraseGrad.size)
        hEraseGrad(j) = 0
      // for(k <- 0 until m; j <- 0 until n)
      //   hEraseGrad(k) -= mgrad(k * n + j) * weightsVal(k)
      Gemv(true, -1, mgrad, weightsVal, 1, hEraseGrad)

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
    // println(Ws.map(_.TopGrad.mkString(",")).mkString(" ; "))
    backwardWErase()
    // println(Ws.map(_.TopGrad.mkString(",")).mkString(" ; "))
    backwardAdd()
    // println(Ws.map(_.TopGrad.mkString(",")).mkString(" ;; "))
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

    // println("-$0- " + wm.erasures.mkString(","))

    for(k <- 0 until m) {
      val we = Array.fill[Double](mtn)(1.0)
      val weights = wm.Ws(k).TopVal
      val erase = wm.erase(k)
      // Ger(alpha f64, x, y Vec, A General)
      // A += alpha * x * y^T
      // for(i <- 0 until m; j <- 0 until n)
      //   we(j * m + i) -= weights(j) * erase(i)
      Ger(-1.0, weights, erase, we)
      // println("+--- " + weights.mkString(","))
      // println("-+-- " + erase.mkString(","))
      // println("--+- " + we.mkString(","))
      for(i <- 0 until mtn)
        wm.erasures(i) *= we(i)
    }

    // println("-$1- " + wm.erasures.mkString(","))

    for(i <- 0 until mtn) wm.TopVal(i) = wm.erasures(i)

    val topG = wm.TopVal
    for(k <- 0 until wm.Ws.size) {
      val weights = wm.Ws(k).TopVal
      val add = wm.add(k)
      // for(i <- 0 until m; j <- 0 until n)
      //   topG(i * n + j) += weights(j) * add(i)
      Ger(1, weights, add, topG)
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
    // println(" -0- " + WM.Mtm1.TopGrad.mkString(","))
    // println(WM.Ws.map(_.TopGrad.mkString(",")).mkString(" ; "))
    R.foreach(r => r.Backward()) //  <---- ERROR HERE -> FIXED
    // println(" -1- " + WM.Mtm1.TopGrad.mkString(","))
    WM.Backward() //  <---- ERROR HERE -> FIXED
    // println(" -2- " + WM.Mtm1.TopGrad.mkString(","))
    // println(WM.Ws.map(_.TopGrad.mkString(",")).mkString(" ; "))
    WM.Ws.foreach { rf =>
      // println(" - CHECK gamma[$k] " + WM.Heads.map(_.GammaGrad()).mkString(","))
      rf.Backward()
      // println(" - CHECK gamma[$k] " + WM.Heads.map(_.GammaGrad()).mkString(","))
      rf.SW.Backward()
      rf.SW.WG.Backward()
      rf.SW.WG.WC.Backward()
      // println(" -330- " + WM.Mtm1.TopGrad.mkString(","))
      rf.SW.WG.WC.Units.foreach { bs =>
        bs.Backward()
        // println(" -3300- " + WM.Mtm1.TopGrad.mkString(","))
        bs.S.Backward() // <---- WHERER THINGS WENT WRONG -> FIXED
      }
      // println(" -331- " + WM.Mtm1.TopGrad.mkString(","))
    }
    // println(" -3- " + WM.Mtm1.TopGrad.mkString(","))
  }
}

object memOp {
  def newMemOp(heads: Array[Head], mtm1: writtenMemory): memOp = {
    // println("mtm1 TV " + mtm1.TopVal.mkString(","))

    val circuit = new memOp(
      R = new Array[memRead](heads.size)
    )
    circuit.W = new Array[refocus](heads.size)
    for(wi <- 0 until heads.size) {
      val h = heads(wi)
      val ss = new Array[betaSimilarity](mtm1.N)
      for(i <- 0 until mtm1.N) {
        val m = mtm1.TopVal.size / mtm1.N
        val s = similarityCircuit.newSimilarityCircuit(h.KVal(), h.KGrad(), mtm1.TopVal, mtm1.TopGrad, i, m)
        ss(i) = betaSimilarity.newBetaSimilarity(h, s)
      }
      val wc = contentAddressing.newContentAddressing(ss)
      val wg = gatedWeighting.newGatedWeighting(h, wc, h.Wtm1)
      val ws = shiftedWeighting.newShiftedWeighting(h, wg)
      circuit.W(wi) = refocus.newRefocus(h, ws)
      circuit.R(wi) = memRead.newMemRead(circuit.W(wi), mtm1)
    }
    circuit.WM = writtenMemory.newWrittenMemory(circuit.W, heads, mtm1)
    circuit
  }
}

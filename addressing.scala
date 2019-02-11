package ntm

import math._

class similarityCircuit(
  val h: Head = null,
  // val UVal: Array[Double],
  // val UGrad: Array[Double],
  val VVal: Array[Array[Double]],
  val VGrad: Array[Array[Double]],
  val index: Int,
  val m: Int
){
  var UV: Double = (0 until m).map { i =>
    h.getKVal(i) * VVal(index)(i)
  }.sum
  var Unorm: Double = Math.sqrt(
    (0 until m).map(i => Math.pow(h.getKVal(i), 2)).sum
  )
  var Vnorm: Double = Math.sqrt(
    (0 until m).map(i => Math.pow(VVal(index)(i), 2)).sum
  )
  var TopVal: Double = UV / (Unorm * Vnorm)
  var TopGrad: Double = 0.0

  def Backward(): Unit = {
    val uvuu = UV / (Unorm * Unorm)
    val uvvv = UV / (Vnorm * Vnorm)
    val uvg = TopGrad / (Unorm * Vnorm)
    for(i <- 0 until m) {
      h.addKGrad(i, uvg * VVal(index)(i) - uvuu * uvg * h.getKVal(i))
      VGrad(index)(i) += uvg * h.getKVal(i) - uvvv * uvg * VVal(index)(i)
    }
  }
}

object similarityCircuit {
  def newSimilarityCircuit(
    h: Head,
    vVal: Array[Array[Double]],
    vGrad: Array[Array[Double]],
    index: Int,
    m: Int
  ): similarityCircuit = {
    new similarityCircuit(
      h = h,
      // UVal = uVal,
      // UGrad = uGrad,
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
  def BetaVal: Double = h.getBetaVal()
  def BetaGrad: Double = h.getBetaGrad()
  if(S != null) Top.Val = b * S.TopVal
  def Backward() {
    h.addBetaGrad(S.TopVal * b * Top.Grad)
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
      b = Math.exp(h.getBetaVal()) // Beta is in the range (-Inf, Inf)
    )
  }
}

class contentAddressing(
  val Units: Array[betaSimilarity],
  val Top: Array[unit]
){
  def Backward(): Unit = {
    var gv: Double = 0
    for(top <- Top) gv += top.Grad * top.Val
    for(i <- 0 until Top.size) Units(i).Top.Grad += (Top(i).Grad - gv) * Top(i).Val
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
    for(u <- s.Units) max = Math.max(max, u.Top.Val)
    
    var sum: Double = 0
    for(i <- 0 until s.Units.size) {
      val w = Math.exp(s.Units(i).Top.Val - max)
      s.Top(i).Val = w
      sum += w
    }
    for(i <- 0 until s.Top.size) s.Top(i).Val = s.Top(i).Val / sum

    s
  }
}

class gatedWeighting(
  var h: Head = null,
  val WC: contentAddressing,
  val Wtm1: refocus, // the weights at time t-1
  val Top: Array[unit]
){
  def GVal: Double = h.getGVal()
  def GGrad: Double = h.getGGrad()
  def Backward(): Unit = {
    val gt = Sigmoid(GVal)

    var grad: Double = 0.0
    for(i <- 0 until Top.size)
      grad += (WC.Top(i).Val - Wtm1.TopVal(i)) * Top(i).Grad

    h.addGGrad(grad * gt * (1 - gt))

    for(i <- 0 until WC.Top.size) WC.Top(i).Grad += gt * Top(i).Grad
    for(i <- 0 until Wtm1.TopGrad.size)
      Wtm1.TopGrad(i) += (1 - gt) * Top(i).Grad
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
      Top = Array.fill[unit](wc.Top.size)(new unit)
    )
    val gt = Sigmoid(h.getGVal())
    for(i <- 0 until wg.Top.size)
      wg.Top(i).Val = gt * wc.Top(i).Val + (1 - gt) * wtm1.TopVal(i)
    wg
  }
}

class shiftedWeighting(
  var h: Head = null,
  var Z: Double = 0.0,
  val WG: gatedWeighting,
  val Top: Array[unit]
){
  def SVal: Double = h.getSVal()
  def SGrad: Double = h.getSGrad()
  def Backward(): Unit = {
    var grad: Double = 0.0
    val n = WG.Top.size
    for(i <- 0 until Top.size) {
      val imj = (i + Z.toInt) % n
      grad += (-WG.Top(imj).Val + WG.Top((imj + 1) % n).Val) * Top(i).Grad
    }
    val sig = Sigmoid(SVal)
    grad = grad * 2 * sig * (1 - sig)
    h.addSGrad(grad)

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

    val shift = (2 * Sigmoid(h.getSVal()) - 1)
    sw.Z = shift - (shift / n.toDouble).toInt * n

    val simj = 1 - (sw.Z - Math.floor(sw.Z))
    for(i <- 0 until sw.Top.size) {
      val imj = (i + sw.Z.toInt) % n
      sw.Top(i).Val = sw.WG.Top(imj).Val * simj + sw.WG.Top((imj + 1) % n).Val * (1 - simj)
      if(sw.Top(i).Val.isNaN || sw.Top(i).Val < 0) {
        Console.err.println(s"imj: $imj, wg: ${sw.WG.Top(imj).Val}, simj: $simj, wg+1: ${sw.WG.Top((imj + 1) % n).Val}")
        System.exit(1)
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
  def GammaVal: Double = h.getGammaVal()
  def GammaGrad: Double = h.getGammaGrad()
  def backwardSW() {
    var topGV: Double = 0
    for(i <- 0 until TopVal.size) topGV += TopGrad(i) * TopVal(i)
    for(i <- 0 until SW.Top.size if SW.Top(i).Val >= machineEpsilon)
      SW.Top(i).Grad += (TopGrad(i) - topGV) * g / SW.Top(i).Val * TopVal(i)
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
    for(i <- 0 until TopVal.size if SW.Top(i).Val >= machineEpsilon)
      grad += TopGrad(i) * (TopVal(i) * (lns(i) - lnexps))
    grad = grad / (1 + Math.exp(-GammaVal))
    h.addGammaGrad(grad)
  }

  def Backward() {
    backwardSW()
    backwardGamma()
  }
}

object refocus {
  def newRefocus(h: Head, sw: shiftedWeighting): refocus = {
    val rf = new refocus(
      h = h,
      SW = sw,
      TopVal = new Array[Double](sw.Top.size),
      TopGrad = new Array[Double](sw.Top.size),
      g = Math.log(Math.exp(h.getGammaVal()) + 1) + 1
    )
    var sum: Double = 0
    for(i <- 0 until rf.TopVal.size) {
      rf.TopVal(i) = Math.pow(sw.Top(i).Val, rf.g)
      sum += rf.TopVal(i)
    }
    for(i <- 0 until rf.TopVal.size) rf.TopVal(i) = rf.TopVal(i) / sum
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
    val grad = TopGrad // grad.size == m
    val memVal = Memory.TopVal
    val weightsGrad = W.TopGrad
    Gemv(false, 1, memVal, grad, 1, weightsGrad)

    val memGrad = Memory.TopGrad
    val weights = W.TopVal
    Ger(1, weights, grad, memGrad)
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
  val TopVal: Array[Array[Double]],
  val TopGrad: Array[Array[Double]],
  var erase: Array[Array[Double]] = null,
  var add: Array[Array[Double]] = null,
  var erasures: Array[Array[Double]] = null
){
  def div1MWE(out: Array[Array[Double]]) {
    val m = TopVal.head.size
    for(i <- 0 until erasures.size; j <- 0 until erasures(i).size) {
      val mwe = 1 - out(i)(j)
      if(mwe.abs > 1e-6) {
        out(i)(j) = erasures(i)(j) / mwe
      } else {
        var mtilt = Mtm1.TopVal(i)(j)
        for(q <- 0 until Ws.size if q != i * m + j)
          mtilt *= (1 - Ws(q).TopVal(i) * erase(q)(j))
        out(i)(j) = mtilt
      }
    }
  }

  def backwardWErase() {
    val n = N
    val m = TopVal.head.size

    val mgrad = Array.ofDim[Double](n, m)
    val hEraseGrad = new Array[Double](m)
    for(i <- 0 until Ws.size) {

      val eraseV = erase(i)
      val addV = add(i)
      val weightsVal = Ws(i).TopVal
      for(j <- 0 until n; k <- 0 until m) mgrad(j)(k) = 0
      Ger(1, weightsVal, eraseV, mgrad)

      div1MWE(mgrad) // <---- FIXED

      for(j <- 0 until n; k <- 0 until m) mgrad(j)(k) *= TopGrad(j)(k)

      val weightsV = Ws(i).TopGrad

      Gemv(false, -1, mgrad, eraseV, 1, weightsV)
      Gemv(false, 1, TopGrad, addV, 1, weightsV)

      val head = Heads(i)
      for(j <- 0 until hEraseGrad.size) hEraseGrad(j) = 0
      Gemv(true, -1, mgrad, weightsVal, 1, hEraseGrad)

      for(j <- 0 until head.M)
        head.addEraseGrad(j, hEraseGrad(j) * eraseV(j) * (1 - eraseV(j)))
    }
  }

  def backwardAdd() {
    val n = N
    val m = TopVal.head.size
    var grad: Double = 0
    for(k <- 0 until Heads.size) {
      val addV = add(k)
      val ws = Ws(k)
      val head = Heads(k)
      for(i <- 0 until head.M) {
        grad = 0
        for(j <- 0 until n) grad += TopGrad(j)(i) * ws.TopVal(j)
        val a = addV(i)
        head.addAddGrad(i, grad * a * (1 - a))
      }
    }
  }

  def backwardMtm1() {
    val n = N
    val m = TopVal.head.size / n
    var grad: Double = 0
    for(i <- 0 until n; j <- 0 until m) {
      grad = 1
      for(q <- 0 until Ws.size) grad *= (1 - Ws(q).TopVal(i) * erase(q)(j))
      Mtm1.TopGrad(i)(j) += grad * TopGrad(i)(j)
    }
  }

  def Backward() {
    backwardWErase()
    backwardAdd()
    backwardMtm1()
  }
}

object writtenMemory {
  def newWrittenMemory(
    ws: Array[refocus],
    heads: Array[Head],
    mtm1: writtenMemory
  ): writtenMemory = {
    val n = mtm1.N
    val m = mtm1.TopVal.head.size
    val mtn = m * n
    val wm = new writtenMemory(
      Ws = ws,
      Heads = heads,
      Mtm1 = mtm1,
      N = mtm1.N,
      // TopVal = new Array[Double](mtn),
      TopVal = Array.ofDim[Double](n, m),
      // TopGrad = new Array[Double](mtn),
      TopGrad = Array.ofDim[Double](n, m),
      erase = Array.ofDim[Double](m, m),
      add = Array.ofDim[Double](m, m),
      erasures = Array.ofDim[Double](n, m)
    )
    for(i <- 0 until m) {
      val erase = wm.erase(i)
      val add = wm.add(i)
      val head = wm.Heads(i)
      for(j <- 0 until m) {
        erase(j) = Sigmoid(head.getEraseVal(j))
        add(j) = Sigmoid(head.getAddVal(j))
      }
    }

    for(i <- 0 until n; j <- 0 until m) wm.erasures(i)(j) = mtm1.TopVal(i)(j)

    for(k <- 0 until m) {
      // val we = Array.fill[Double](mtn)(1.0)
      val we = Array.ofDim[Double](n, m)
      for(i <- 0 until n; j <- 0 until m) we(i)(j) = 1.0
      val weights = wm.Ws(k).TopVal
      val erase = wm.erase(k)
      Ger(-1.0, weights, erase, we)
      for(i <- 0 until n; j <- 0 until m) wm.erasures(i)(j) *= we(i)(j)
    }

    for(i <- 0 until n; j <- 0 until m) wm.TopVal(i)(j) = wm.erasures(i)(j)

    val topG = wm.TopVal
    for(k <- 0 until wm.Ws.size) {
      val weights = wm.Ws(k).TopVal
      val add = wm.add(k)
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
      // println(" - CHECK gamma[$k] " + WM.Heads.map(_.getGammaGrad()).mkString(","))
      rf.Backward()
      // println(" - CHECK gamma[$k] " + WM.Heads.map(_.getGammaGrad()).mkString(","))
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
    val circuit = new memOp(
      R = new Array[memRead](heads.size)
    )
    circuit.W = new Array[refocus](heads.size)
    for(wi <- 0 until heads.size) {
      val h = heads(wi)
      val ss = new Array[betaSimilarity](mtm1.N)
      for(i <- 0 until mtm1.N) {
        val m = mtm1.TopVal.head.size
        val s = similarityCircuit.newSimilarityCircuit(h, mtm1.TopVal, mtm1.TopGrad, i, m)
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

package com.scalaml.ntm

import org.apache.mxnet._

import MxNetMath._

class MNHead(
  // size of a row in the memory
  val M: Int
) {
  // the weights at time t-1
  // var Wtm1: refocus = null
  var vals: NDArray = null
  var grads: NDArray = null
  // EraseVector returns the erase vector of a memory head.
  val eraseVal: NDArray = vals.slice(0, M)
  val eraseGrad: NDArray = grads.slice(0, M)
  // AddVector returns the add vector of a memory head.
  val addVal: NDArray = vals.slice(M, 2 * M)
  val addGrad: NDArray = grads.slice(M, 2 * M)
  // K returns a head's key vector,
  // which is the target data in the content addressing step.
  val kVal: NDArray = vals.slice(2 * M, 3 * M)
  val kGrad: NDArray = grads.slice(2 * M, 3 * M)
  // Beta returns the key strength of a content addressing step.
  val betaVal: NDArray = vals.slice(3 * M, 3 * M + 1)
  val betaGrad: NDArray = grads.slice(3 * M, 3 * M + 1)
  // G returns the degree in which we want to choose content-addressing over
  // location-based-addressing.
  val gVal: NDArray = vals.slice(3 * M + 1, 3 * M + 2)
  val gGrad: NDArray = grads.slice(3 * M + 1, 3 * M + 2)
  // S returns a value indicating how much the weightings are rotated in a
  // location-based-addressing step.
  val sVal: NDArray = vals.slice(3 * M + 2, 3 * M + 3)
  val sGrad: NDArray = grads.slice(3 * M + 2, 3 * M + 3)
  // Gamma returns the degree in which the addressing weights are sharpened.
  val gammaVal: NDArray = vals.slice(3 * M + 3, 3 * M + 4)
  val gammaGrad: NDArray = grads.slice(3 * M + 3, 3 * M + 4)
}

object MNHead {
  // NewHead creates a new memory head.
  def NewHead(m: Int): MNHead = new MNHead(M = m)
  def headUnitsLen(m: Int): Int = 3 * m + 4
}

class MNSimilarityCircuit(
  val h: MNHead = null,
  val VVal: NDArray,
  val VGrad: NDArray,
  val m: Int
){
  val UV: NDArray = NDArray.dot(h.kVal, VVal) // 1x1 NDArray
  val Unorm: NDArray = NDArray.sqrt(NDArray.sum(h.kVal * h.kVal)) // 1x1 NDArray
  val Vnorm: NDArray = NDArray.sqrt(NDArray.sum(VVal * VVal)) // 1x1 NDArray
  var TopVal: NDArray = UV / (Unorm * Vnorm)
  var TopGrad: NDArray = NDArray.zeros(1, 1)

  def Backward(): Unit = {
    val uvuu = UV / (Unorm * Unorm)
    val uvvv = UV / (Vnorm * Vnorm)
    val uvg = TopGrad / (Unorm * Vnorm)

    h.kGrad += NDArray.dot(VVal, uvg) - NDArray.dot(h.kVal, uvuu * uvg)
    VGrad += NDArray.dot(h.kVal, uvg) - NDArray.dot(VVal, uvvv * uvg)
  }
}

object MNSimilarityCircuit {
  def newSimilarityCircuit(
    h: MNHead,
    vVal: NDArray,
    vGrad: NDArray,
    m: Int
  ): MNSimilarityCircuit = {
    new MNSimilarityCircuit(
      h = h,
      VVal = vVal,
      VGrad = vGrad,
      m = m
    )
  }
}

class MNBetaSimilarity(
  var h: MNHead = null,
  var S: MNSimilarityCircuit = null,
  var b: NDArray = NDArray.zeros(1),
  var Top: MxNetUnit = new MxNetUnit
){
  def BetaVal: NDArray = h.betaVal
  def BetaGrad: NDArray = h.betaGrad
  if(S != null) Top.Val = S.TopVal * b
  def Backward() {
    h.betaGrad += S.TopVal * Top.Grad * b
    S.TopGrad += Top.Grad * b
  }
}

object MNBetaSimilarity {
  def newBetaSimilarity(
    h: MNHead,
    s: MNSimilarityCircuit
  ): MNBetaSimilarity = {
    new MNBetaSimilarity(
      h = h,
      S = s,
      b = NDArray.exp(h.betaVal) // Beta is in the range (-Inf, Inf)
    )
  }
}

class MNContentAddressing(
  val Units: Array[MNBetaSimilarity],
  val Top: Array[MxNetUnit] // <---- NEED REWRITE ----
){
  def Backward(): Unit = {
    val gv: NDArray = NDArray.zeros(1)
    Top.foreach(top => gv += top.Grad * top.Val)
    (0 until Top.size).foreach(i => Units(i).Top.Grad += (Top(i).Grad - gv) * Top(i).Val)
  }
}

object MNContentAddressing {
  def newContentAddressing(units: Array[MNBetaSimilarity]): MNContentAddressing = {
    val s = new MNContentAddressing(
      Units = units,
      Top = Array.fill[MxNetUnit](units.size)(new MxNetUnit)
    )
    // Increase numerical stability by subtracting all weights by their max,
    // before computing Math.exp().
    var max: Float = Float.MinValue
    s.Units.foreach(u => max = Math.max(max, u.Top.Val.toArray(0)))
    
    val sum: NDArray = NDArray.zeros(1)
    (0 until s.Units.size).foreach { i =>
      val w = NDArray.exp(s.Units(i).Top.Val - max)
      s.Top(i).Val = w
      sum += w
    }
    s.Top.foreach(top => top.Val /= sum)
    s
  }
}

// class MNGatedWeighting(
//   var h: MNHead = null,
//   val WC: MNContentAddressing,
//   val Wtm1: MNRefocus, // the weights at time t-1
//   val Top: Array[MxNetUnit]
// ){
//   def GVal: NDArray = h.gVal
//   def GGrad: NDArray = h.gGrad
//   def Backward(): Unit = {
//     val gt = Sigmoid(GVal)
//     val grad: NDArray = NDArray.zeros(1)
//     (0 until Top.size).foreach { i =>
//       grad += (WC.Top(i).Val - Wtm1.TopVal(i)) * Top(i).Grad
//     }
//     h.gGrad += grad * gt * (gt * -1 + 1)
//     (0 until WC.Top.size).foreach { i =>
//       WC.Top(i).Grad += gt * Top(i).Grad
//     }
//     (0 until Wtm1.TopGrad.size).foreach { i =>
//       Wtm1.TopGrad(i) += (gt * -1 + 1) * Top(i).Grad
//     }
//   }
// }

// object MNGatedWeighting {
//   def newGatedWeighting(
//     h: Head,
//     wc: MNContentAddressing,
//     wtm1: refocus
//   ): MNGatedWeighting = {
//     val wg = new MNGatedWeighting(
//       h = h,
//       WC = wc,
//       Wtm1 = wtm1,
//       Top = Array.fill[unit](wc.Top.size)(new unit)
//     )
//     val gt = Sigmoid(h.getGVal())
//     for(i <- 0 until wg.Top.size)
//       wg.Top(i).Val = gt * wc.Top(i).Val + (1 - gt) * wtm1.TopVal(i)
//     wg
//   }
// }

// class MNShiftedWeighting(
//   var h: Head = null,
//   var Z: Double = 0.0,
//   val WG: MNGatedWeighting,
//   val Top: Array[unit]
// ){
//   def SVal: Double = h.getSVal()
//   def SGrad: Double = h.getSGrad()
//   def Backward(): Unit = {
//     var grad: Double = 0.0
//     val n = WG.Top.size
//     for(i <- 0 until Top.size) {
//       val imj = (i + Z.toInt) % n
//       grad += (-WG.Top(imj).Val + WG.Top((imj + 1) % n).Val) * Top(i).Grad
//     }
//     val sig = Sigmoid(SVal)
//     grad = grad * 2 * sig * (1 - sig)
//     h.addSGrad(grad)

//     val simj = 1 - (Z - Math.floor(Z))
//     for(i <- 0 until WG.Top.size) {
//       val j = (i - Z.toInt + n) % n
//       WG.Top(i).Grad += Top(j).Grad * simj + Top((j - 1 + n) % n).Grad * (1 - simj)
//     }
//   }
// }

// object MNShiftedWeighting {
//   def newShiftedWeighting(h: Head, wg: MNGatedWeighting): MNShiftedWeighting = {
//     val sw = new MNShiftedWeighting(
//       h = h,
//       WG = wg,
//       Top = Array.fill[unit](wg.Top.size)(new unit)
//     )

//     val n = sw.WG.Top.size

//     val shift = (2 * Sigmoid(h.getSVal()) - 1)
//     sw.Z = shift - (shift / n.toDouble).toInt * n

//     val simj = 1 - (sw.Z - Math.floor(sw.Z))
//     for(i <- 0 until sw.Top.size) {
//       val imj = (i + sw.Z.toInt) % n
//       sw.Top(i).Val = sw.WG.Top(imj).Val * simj + sw.WG.Top((imj + 1) % n).Val * (1 - simj)
//       if(sw.Top(i).Val.isNaN || sw.Top(i).Val < 0) {
//         Console.err.println(s"imj: $imj, wg: ${sw.WG.Top(imj).Val}, simj: $simj, wg+1: ${sw.WG.Top((imj + 1) % n).Val}")
//         System.exit(1)
//       }
//     }
//     sw
//   }
// }

// class refocus(
//   var h: Head = null,
//   var SW: MNShiftedWeighting = null,
//   val TopVal: Array[Double],
//   val TopGrad: Array[Double],
//   var g: Double = 0
// ){
//   def GammaVal: Double = h.getGammaVal()
//   def GammaGrad: Double = h.getGammaGrad()
//   def backwardSW() {
//     var topGV: Double = 0
//     for(i <- 0 until TopVal.size) topGV += TopGrad(i) * TopVal(i)
//     for(i <- 0 until SW.Top.size if SW.Top(i).Val >= machineEpsilon)
//       SW.Top(i).Grad += (TopGrad(i) - topGV) * g / SW.Top(i).Val * TopVal(i)
//   }

//   def backwardGamma() {
//     val lns = new Array[Double](SW.Top.size)
//     var lnexp: Double = 0
//     var s: Double = 0
//     for(i <- 0 until SW.Top.size if SW.Top(i).Val >= machineEpsilon) {
//       lns(i) = Math.log(SW.Top(i).Val)
//       val pow = Math.pow(SW.Top(i).Val, g)
//       lnexp += lns(i) * pow
//       s += pow
//     }
//     val lnexps = lnexp / s
//     var grad: Double = 0
//     for(i <- 0 until TopVal.size if SW.Top(i).Val >= machineEpsilon)
//       grad += TopGrad(i) * (TopVal(i) * (lns(i) - lnexps))
//     grad = grad / (1 + Math.exp(-GammaVal))
//     h.addGammaGrad(grad)
//   }

//   def Backward() {
//     backwardSW()
//     backwardGamma()
//   }
// }

// object refocus {
//   def newRefocus(h: Head, sw: MNShiftedWeighting): refocus = {
//     val rf = new refocus(
//       h = h,
//       SW = sw,
//       TopVal = new Array[Double](sw.Top.size),
//       TopGrad = new Array[Double](sw.Top.size),
//       g = Math.log(Math.exp(h.getGammaVal()) + 1) + 1
//     )
//     var sum: Double = 0
//     for(i <- 0 until rf.TopVal.size) {
//       rf.TopVal(i) = Math.pow(sw.Top(i).Val, rf.g)
//       sum += rf.TopVal(i)
//     }
//     for(i <- 0 until rf.TopVal.size) rf.TopVal(i) = rf.TopVal(i) / sum
//     rf
//   }
// }

// class memRead(
//   val W: refocus,
//   val Memory: writtenMemory,
//   val TopVal: Array[Double],
//   val TopGrad: Array[Double]
// ){
//   def Backward() {
//     val n = Memory.N
//     val grad = TopGrad // grad.size == Memory.TopVal.last.size
//     val memVal = Memory.TopVal
//     val weightsGrad = W.TopGrad
//     // println("grad " + grad.mkString(","))
//     // println("memVal " + memVal.map(_.mkString(",")).mkString(";"))
//     // println("weightsGrad " + weightsGrad.mkString(","))
//     Gemv(false, 1, memVal, grad, 1, weightsGrad)
//     // println("weightsGrad " + weightsGrad.mkString(","))
//     val memGrad = Memory.TopGrad
//     val weights = W.TopVal
//     Ger(1, weights, grad, memGrad)
//   }
// }

// object memRead {
//   def newMemRead(w: refocus, memory: writtenMemory): memRead = {
//     val n = memory.N
//     val m = memory.TopVal.last.size
//     val r = new memRead(
//       W = w,
//       Memory = memory,
//       TopVal = new Array[Double](m),
//       TopGrad = new Array[Double](m)
//     )
//     val weights = w.TopVal
//     val mem = memory.TopVal
//     val top = r.TopVal
//     Gemv(true, 1, mem, weights, 1, top)
//     r
//   }
// }

// class writtenMemory(
//   var Ws: Array[refocus] = null,
//   // We actually need only the erase and add vectors.
//   var Heads: Array[Head] = null,
//   // memory at time t-1
//   var Mtm1: writtenMemory = null,
//   // memoryN
//   val N: Int,
//   val TopVal: Array[Array[Double]],
//   val TopGrad: Array[Array[Double]],
//   var erase: Array[Array[Double]] = null,
//   var add: Array[Array[Double]] = null,
//   var erasures: Array[Array[Double]] = null
// ){
//   def div1MWE(out: Array[Array[Double]]) {
//     val m = TopVal.last.size
//     for(i <- 0 until erasures.size; j <- 0 until erasures(i).size) {
//       val mwe = 1 - out(i)(j)
//       if(mwe.abs > 1e-6) {
//         out(i)(j) = erasures(i)(j) / mwe
//       } else {
//         var mtilt = Mtm1.TopVal(i)(j)
//         for(q <- 0 until Ws.size if q != i * m + j)
//           mtilt *= (1 - Ws(q).TopVal(i) * erase(q)(j))
//         out(i)(j) = mtilt
//       }
//     }
//   }

//   def backwardWErase() {
//     val n = N
//     val m = TopVal.last.size

//     val mgrad = Array.ofDim[Double](n, m)
//     val hEraseGrad = new Array[Double](m)
//     for(i <- 0 until Ws.size) {

//       val eraseV = erase(i)
//       val addV = add(i)
//       val weightsVal = Ws(i).TopVal
//       for(j <- 0 until n; k <- 0 until m) mgrad(j)(k) = 0
//       Ger(1, weightsVal, eraseV, mgrad)

//       div1MWE(mgrad) // <---- FIXED

//       for(j <- 0 until n; k <- 0 until m) mgrad(j)(k) *= TopGrad(j)(k)

//       val weightsV = Ws(i).TopGrad

//       Gemv(false, -1, mgrad, eraseV, 1, weightsV)
//       Gemv(false, 1, TopGrad, addV, 1, weightsV)

//       val head = Heads(i)
//       for(j <- 0 until hEraseGrad.size) hEraseGrad(j) = 0
//       Gemv(true, -1, mgrad, weightsVal, 1, hEraseGrad)

//       for(j <- 0 until head.M)
//         head.addEraseGrad(j, hEraseGrad(j) * eraseV(j) * (1 - eraseV(j)))
//     }
//   }

//   def backwardAdd() {
//     val n = N
//     var grad: Double = 0
//     for(k <- 0 until Heads.size) {
//       val addV = add(k)
//       val ws = Ws(k)
//       val head = Heads(k)
//       for(i <- 0 until head.M) {
//         grad = 0
//         for(j <- 0 until n) grad += TopGrad(j)(i) * ws.TopVal(j)
//         val a = addV(i)
//         head.addAddGrad(i, grad * a * (1 - a))
//       }
//     }
//   }

//   def backwardMtm1() {
//     val n = N
//     val m = TopVal.last.size
//     var grad: Double = 0
//     for(i <- 0 until n; j <- 0 until m) {
//       grad = 1
//       for(q <- 0 until Ws.size) grad *= (1 - Ws(q).TopVal(i) * erase(q)(j))
//       // println("grad " + grad)
//       Mtm1.TopGrad(i)(j) += grad * TopGrad(i)(j)
//     }
//   }

//   def Backward() {
//     backwardWErase()
//     // println(" -1.1- " + Mtm1.TopGrad.map(_.mkString(",")).mkString(";"))
//     backwardAdd()
//     // println(" -1.2- " + Mtm1.TopGrad.map(_.mkString(",")).mkString(";"))
//     backwardMtm1() // <-- ERROR HERE -> FIXED
//   }
// }

// object writtenMemory {
//   def newWrittenMemory(
//     ws: Array[refocus],
//     heads: Array[Head],
//     mtm1: writtenMemory
//   ): writtenMemory = {
//     val n = mtm1.N
//     val m = mtm1.TopVal.head.size
//     val numHeads = heads.size
//     val mtn = m * n
//     val wm = new writtenMemory(
//       Ws = ws,
//       Heads = heads,
//       Mtm1 = mtm1,
//       N = mtm1.N,
//       // TopVal = new Array[Double](mtn),
//       TopVal = Array.ofDim[Double](n, m),
//       // TopGrad = new Array[Double](mtn),
//       TopGrad = Array.ofDim[Double](n, m),
//       erase = Array.ofDim[Double](numHeads, m),
//       add = Array.ofDim[Double](numHeads, m),
//       erasures = Array.ofDim[Double](n, m)
//     )

//     for(i <- 0 until wm.Heads.size) {
//       val erase = wm.erase(i)
//       val add = wm.add(i)
//       val head = wm.Heads(i)
//       for(j <- 0 until numHeads) {
//         erase(j) = Sigmoid(head.getEraseVal(j))
//         add(j) = Sigmoid(head.getAddVal(j))
//       }
//     }

//     for(i <- 0 until n; j <- 0 until m) wm.erasures(i)(j) = mtm1.TopVal(i)(j)

//     for(k <- 0 until wm.Ws.size) {
//       // val we = Array.fill[Double](mtn)(1.0)
//       val we = Array.ofDim[Double](n, m)
//       for(i <- 0 until n; j <- 0 until m) we(i)(j) = 1.0
//       val weights = wm.Ws(k).TopVal
//       val erase = wm.erase(k)
//       Ger(-1.0, weights, erase, we)
//       for(i <- 0 until n; j <- 0 until m) wm.erasures(i)(j) *= we(i)(j)
//     }

//     for(i <- 0 until n; j <- 0 until m) wm.TopVal(i)(j) = wm.erasures(i)(j)

//     val topG = wm.TopVal
//     for(k <- 0 until wm.Ws.size) {
//       val weights = wm.Ws(k).TopVal
//       val add = wm.add(k)
//       Ger(1, weights, add, topG)
//     }

//     wm
//   }
// }

// class memOp(
//   val R: Array[memRead],
//   var W: Array[refocus] = null,
//   var WM: writtenMemory = null
// ) {
//   def Backward() {
//     // println(" -0- " + WM.Mtm1.TopGrad.map(_.mkString(",")).mkString(";"))
//     // println(WM.Ws.map(_.TopGrad.mkString(",")).mkString(" ; "))
//     R.foreach(r => r.Backward()) //  <---- ERROR HERE -> FIXED
//     // println(" -1- " + WM.Mtm1.TopGrad.map(_.mkString(",")).mkString(";"))
//     WM.Backward() //  <---- ERROR HERE -> FIXED
//     // println(" -2- " + WM.Mtm1.TopGrad.map(_.mkString(",")).mkString(";"))
//     // println(WM.Ws.map(_.TopGrad.mkString(",")).mkString(" ; "))
//     WM.Ws.foreach { rf =>
//       // println(" - CHECK gamma[$k] " + WM.Heads.map(_.getGammaGrad()).mkString(","))
//       rf.Backward()
//       // println(" - CHECK gamma[$k] " + WM.Heads.map(_.getGammaGrad()).mkString(","))
//       rf.SW.Backward()
//       rf.SW.WG.Backward()
//       rf.SW.WG.WC.Backward()
//       // println(" -330- " + WM.Mtm1.TopGrad.mkString(","))
//       rf.SW.WG.WC.Units.foreach { bs =>
//         bs.Backward()
//         // println(" -3300- " + WM.Mtm1.TopGrad.mkString(","))
//         bs.S.Backward() // <---- WHERER THINGS WENT WRONG -> FIXED
//       }
//       // println(" -331- " + WM.Mtm1.TopGrad.mkString(","))
//     }
//     // println(" -3- " + WM.Mtm1.TopGrad.map(_.mkString(",")).mkString(","))
//   }
// }

// object memOp {
//   def newMemOp(heads: Array[Head], mtm1: writtenMemory): memOp = {
//     val circuit = new memOp(
//       R = new Array[memRead](heads.size)
//     )
//     circuit.W = new Array[refocus](heads.size)
//     for(wi <- 0 until heads.size) {
//       val h = heads(wi)
//       val ss = new Array[MNBetaSimilarity](mtm1.N)
//       for(i <- 0 until mtm1.N) {
//         val m = mtm1.TopVal.head.size
//         val s = MNSimilarityCircuit.newSimilarityCircuit(h, mtm1.TopVal(i), mtm1.TopGrad(i), m)
//         // println(s.TopVal + " -MNSimilarityCircuit")
//         ss(i) = MNBetaSimilarity.newBetaSimilarity(h, s)
//         // println(ss(i).Top.Val + " -MNBetaSimilarity")
//       }
//       val wc = MNContentAddressing.newContentAddressing(ss)
//       val wg = MNGatedWeighting.newGatedWeighting(h, wc, h.Wtm1)
//       val ws = MNShiftedWeighting.newShiftedWeighting(h, wg)
//       circuit.W(wi) = refocus.newRefocus(h, ws)
//       circuit.R(wi) = memRead.newMemRead(circuit.W(wi), mtm1)
//     }
//     circuit.WM = writtenMemory.newWrittenMemory(circuit.W, heads, mtm1)
//     circuit
//   }
// }

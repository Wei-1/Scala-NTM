package ntm

import math._

object ntm_test {
  def TestRMSProp(t: T) {
    val xSize = 1
    val ySize = 1
    val h1Size = 1
    val numHeads = 1
    val n = 1
    val m = 1
    val c = Controller.NewEmptyController(xSize, ySize, h1Size, numHeads, n, m)
    val rms = RMSProp.NewRMSProp(c)

    c.WeightsValVec()(0)(0) = 1.1
    c.WeightsGradVec()(0)(0) = 2.7
    rms.N(0)(0) = 10.3
    rms.G(0)(0) = 1.8
    rms.D(0)(0) = 3.7

    c.WeightsValVec()(0)(1) = 1.2
    c.WeightsGradVec()(0)(1) = 1.9
    rms.N(0)(1) = 14.3
    rms.G(0)(1) = 2.1
    rms.D(0)(1) = 1.7

    c.WeightsValVec().last(c.WeightsValVec().last.size - 1) = 0.9
    c.WeightsGradVec().last(c.WeightsValVec().last.size - 1) = 1.3
    rms.N.last(c.WeightsValVec().last.size - 1) = 12.3
    rms.G.last(c.WeightsValVec().last.size - 1) = 0.8
    rms.D.last(c.WeightsValVec().last.size - 1) = 8.1

    rms.update(0.95, 0.9, 0.0001, 0.0001)

    checkRMS(t, c, rms, 0, 0, 10.1495, 1.845, 3.329896, 4.429896)
    checkRMS(t, c, rms, 0, 1, 13.7655, 2.09, 1.529938, 2.729938)
    checkRMS(t, c, rms, c.WeightsValVec().size - 1, c.WeightsValVec().last.size - 1,
      11.7695, 0.825, 7.289961, 8.189961)
  }

  def checkRMS(t: T, c: Controller, rms: RMSProp, i: Int, j: Int, n: Double, g: Double, d: Double, w: Double) {
    val tol = 1e-6
    if(Math.abs(rms.N(i)(j) - n) > tol) {
      t.Errorf(s"[NTM] wrong rms.N[$i][$j] expected $n, got ${rms.N(i)(j)}")
    } else {
      t.Logf(s"[NTM] OK rms.N[$i][$j] expected $n, got ${rms.N(i)(j)}")
    }
    if(Math.abs(rms.G(i)(j) - g) > tol) {
      t.Errorf(s"[NTM] wrong rms.G[$i][$j] expected $g, got ${rms.G(i)(j)}")
    } else {
      t.Logf(s"[NTM] OK rms.G[$i][$j] expected $g, got ${rms.G(i)(j)}")
    }
    if(Math.abs(rms.D(i)(j) - d) > tol) {
      t.Errorf(s"[NTM] wrong rms.D[$i][$j] expected $d, got ${rms.D(i)(j)}")
    } else {
      t.Logf(s"[NTM] OK rms.D[$i][$j] expected $d, got ${rms.D(i)(j)}")
    }
    if(Math.abs(c.WeightsValVec()(i)(j) - w) > tol) {
      t.Errorf(s"[NTM] wrong w[$i][$j] expected $w, got ${c.WeightsValVec()(i)(j)}")
    } else {
      t.Logf(s"[NTM] OK w[$i][$j] expected $w, got ${c.WeightsValVec()(i)(j)}")
    }
  }
}

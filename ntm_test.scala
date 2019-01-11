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
    val c = controller1.NewEmptyController1(xSize, ySize, h1Size, numHeads, n, m)
    val rms = RMSProp.NewRMSProp(c)

    c.WeightsVal()(0) = 1.1
    c.WeightsGrad()(0) = 2.7
    rms.N(0) = 10.3
    rms.G(0) = 1.8
    rms.D(0) = 3.7

    c.WeightsVal()(1) = 1.2
    c.WeightsGrad()(1) = 1.9
    rms.N(1) = 14.3
    rms.G(1) = 2.1
    rms.D(1) = 1.7

    c.WeightsVal()(c.WeightsVal().size - 1) = 0.9
    c.WeightsGrad()(c.WeightsVal().size - 1) = 1.3
    rms.N(c.WeightsVal().size - 1) = 12.3
    rms.G(c.WeightsVal().size - 1) = 0.8
    rms.D(c.WeightsVal().size - 1) = 8.1

    rms.update(0.95, 0.9, 0.0001, 0.0001)

    checkRMS(t, c, rms, 0, 10.1495, 1.845, 3.329896, 4.429896)
    checkRMS(t, c, rms, 1, 13.7655, 2.09, 1.529938, 2.729938)
    checkRMS(t, c, rms, c.WeightsVal().size - 1, 11.7695, 0.825, 7.289961, 8.189961)
  }

  def checkRMS(t: T, c: Controller, rms: RMSProp, i: Int, n: Double, g: Double, d: Double, w: Double) {
    val tol = 1e-6
    if(Math.abs(rms.N(i) - n) > tol) {
      t.Errorf(s"[NTM] wrong rms.N[$i] expected $n, got ${rms.N(i)}")
    } else {
      t.Logf(s"[NTM] OK rms.N[$i] expected $n, got ${rms.N(i)}")
    }
    if(Math.abs(rms.G(i) - g) > tol) {
      t.Errorf(s"[NTM] wrong rms.G[$i] expected $g, got ${rms.G(i)}")
    } else {
      t.Logf(s"[NTM] OK rms.G[$i] expected $g, got ${rms.G(i)}")
    }
    if(Math.abs(rms.D(i) - d) > tol) {
      t.Errorf(s"[NTM] wrong rms.D[$i] expected $d, got ${rms.D(i)}")
    } else {
      t.Logf(s"[NTM] OK rms.D[$i] expected $d, got ${rms.D(i)}")
    }
    if(Math.abs(c.WeightsVal()(i) - w) > tol) {
      t.Errorf(s"[NTM] wrong w[$i] expected $w, got ${c.WeightsVal()(i)}")
    } else {
      t.Logf(s"[NTM] OK w[$i] expected $w, got ${c.WeightsVal()(i)}")
    }
  }
}

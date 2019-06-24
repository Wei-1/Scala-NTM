package com.scalaml.ntm

import math._

object math_test {
  def TestMath(t: T) {
    def checkEqual(s: String, v: Double): Unit = {
      if(v.abs < machineEpsilon) t.Logf(s"[MATH] OK math $s")
      else t.Errorf(s"[MATH] wrong math $s")
    }

    checkEqual("Sigmoid(0)", Sigmoid(0) - 0.5)
    checkEqual("Sigmoid(1)", Sigmoid(1) - 0.7310585786300049)
    checkEqual("Sigmoid(-1)", Sigmoid(-1) - 0.2689414213699951)
    checkEqual("Sigmoid(10)", Sigmoid(10) - 0.9999546021312976)
    checkEqual("Sigmoid(-10)", Sigmoid(-10) - 4.5397868702434395E-5)

    checkEqual("delta(0, 0)", delta(0, 0) - 1)
    checkEqual("delta(1, 1)", delta(1, 1) - 1)
    checkEqual("delta(-1, -1)", delta(-1, -1) - 1)
    checkEqual("delta(10, 10)", delta(1, 0))
    checkEqual("delta(-10, -10)", delta(-10, 10))

    checkEqual("cosineSimilarity(Array(1, 0), Array(1, 0))", cosineSimilarity(Array(1, 0), Array(1, 0)) - 1.0)
    checkEqual("cosineSimilarity(Array(1, 0), Array(2, 0))", cosineSimilarity(Array(1, 0), Array(2, 0)) - 1.0)
    checkEqual("cosineSimilarity(Array(1, 2), Array(2, 3))", cosineSimilarity(Array(1, 2), Array(2, 3)) - 0.9922778767136677)
    checkEqual("cosineSimilarity(Array(1, 2, 3), Array(2, 3, 4))", cosineSimilarity(Array(1, 2, 3), Array(2, 3, 4)) - 0.9925833339709302)
    checkEqual("cosineSimilarity(Array(1, 2), Array(-2, -1))", cosineSimilarity(Array(1, 2), Array(-2, -1)) - -0.8)

    checkEqual("makeTensor2(2, 3).size", makeTensor2(2, 3).size - 2)
    checkEqual("makeTensor2(2, 3).head.size", makeTensor2(2, 3).head.size - 3)

    var y = Array[Double]()
    y = Array(7.0, 8.0)
    Gemv(false, 0.5, Array(Array(1, 2), Array(3, 4)), Array(5, 6), 0.05, y)
    checkEqual("Gemv(false, 0.5, Array(Array(1, 2), Array(3, 4)), Array(5, 6), 0.05, y) y(0)", y(0) - 6.1425)
    checkEqual("Gemv(false, 0.5, Array(Array(1, 2), Array(3, 4)), Array(5, 6), 0.05, y) y(1)", y(1) - 12.395)

    y = Array(7.0, 8.0)
    Gemv(true, 0.5, Array(Array(1, 2), Array(3, 4)), Array(5, 6), 0.05, y)
    checkEqual("Gemv(true, 0.5, Array(Array(1, 2), Array(3, 4)), Array(5, 6), 0.05, y) y(0)", y(0) - 9.1425)
    checkEqual("Gemv(true, 0.5, Array(Array(1, 2), Array(3, 4)), Array(5, 6), 0.05, y) y(1)", y(1) - 12.27)

    var a = Array[Array[Double]]()
    a = Array(Array(1, 2), Array(3, 4))
    Ger(0.5, Array(5, 6), Array(7, 8), a)
    checkEqual("Ger(0.5, Array(5, 6), Array(7, 8), a) a(0)(0)", a(0)(0) - 18.5)
    checkEqual("Ger(0.5, Array(5, 6), Array(7, 8), a) a(0)(1)", a(0)(1) - 22.0)
    checkEqual("Ger(0.5, Array(5, 6), Array(7, 8), a) a(1)(0)", a(1)(0) - 24.0)
    checkEqual("Ger(0.5, Array(5, 6), Array(7, 8), a) a(1)(1)", a(1)(1) - 28.0)
  }
}

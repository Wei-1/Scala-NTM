package com.scalaml.ntm

import org.apache.mxnet._

import com.scalaml.ntm.mxnet_math._

object math_test {
  def TestMath(t: T): Unit = try {
    def checkEqual(s: String, v: Double): Unit = {
      if(v.abs < Math.sqrt(machineEpsilonSqrt)) t.Logf(s"[MATH] OK math $s")
      else t.Errorf(s"[MATH] wrong math $s")
    }

    checkEqual("Sigmoid(0)", Sigmoid(NDArray.array(Array(0), Shape(1, 1))).toArray(0) - 0.5)
    checkEqual("Sigmoid(1)", Sigmoid(NDArray.array(Array(1), Shape(1, 1))).toArray(0) - 0.7310585786300049)
    checkEqual("Sigmoid(-1)", Sigmoid(NDArray.array(Array(-1), Shape(1, 1))).toArray(0) - 0.2689414213699951)
    checkEqual("Sigmoid(10)", Sigmoid(NDArray.array(Array(10), Shape(1, 1))).toArray(0) - 0.9999546021312976)
    checkEqual("Sigmoid(-10)", Sigmoid(NDArray.array(Array(-10), Shape(1, 1))).toArray(0) - 4.5397868702434395E-5)

    checkEqual("delta(0, 0)", delta(0, 0) - 1)
    checkEqual("delta(1, 1)", delta(1, 1) - 1)
    checkEqual("delta(-1, -1)", delta(-1, -1) - 1)
    checkEqual("delta(10, 10)", delta(1, 0))
    checkEqual("delta(-10, -10)", delta(-10, 10))

    checkEqual("cosineSimilarity(Array(1, 0), Array(1, 0))", cosineSimilarity(NDArray.array(Array(1, 0), Shape(2, 1)), NDArray.array(Array(1, 0), Shape(2, 1))).toArray(0) - 1.0)
    checkEqual("cosineSimilarity(Array(1, 0), Array(2, 0))", cosineSimilarity(NDArray.array(Array(1, 0), Shape(2, 1)), NDArray.array(Array(2, 0), Shape(2, 1))).toArray(0) - 1.0)
    checkEqual("cosineSimilarity(Array(1, 2), Array(2, 3))", cosineSimilarity(NDArray.array(Array(1, 2), Shape(2, 1)), NDArray.array(Array(2, 3), Shape(2, 1))).toArray(0) - 0.9922778767136677)
    checkEqual("cosineSimilarity(Array(1, 2, 3), Array(2, 3, 4))", cosineSimilarity(NDArray.array(Array(1, 2, 3), Shape(3, 1)), NDArray.array(Array(2, 3, 4), Shape(3, 1))).toArray(0) - 0.9925833339709302)
    checkEqual("cosineSimilarity(Array(1, 2), Array(-2, -1))", cosineSimilarity(NDArray.array(Array(1, 2), Shape(2, 1)), NDArray.array(Array(-2, -1), Shape(2, 1))).toArray(0) - -0.8)

    checkEqual("makeTensor2(2, 3).size", makeTensor2(2, 3).shape(0) - 2)
    checkEqual("makeTensor2(2, 3).head.size", makeTensor2(2, 3).shape(1) - 3)

    var y = Array[Float]()
    y = Gemv(false, 0.5f, NDArray.array(Array(1, 2, 3, 4), Shape(2, 2)), NDArray.array(Array(5, 6), Shape(2, 1)), 0.05f, NDArray.array(Array(7, 8), Shape(2, 1))).toArray
    checkEqual("Gemv(false, 0.5, Array(Array(1, 2), Array(3, 4)), Array(5, 6), 0.05, y) y(0)", y(0) - 8.85)
    checkEqual("Gemv(false, 0.5, Array(Array(1, 2), Array(3, 4)), Array(5, 6), 0.05, y) y(1)", y(1) - 19.9)

    y = Gemv(true, 0.5f, NDArray.array(Array(1, 2, 3, 4), Shape(2, 2)), NDArray.array(Array(5, 6), Shape(2, 1)), 0.05f, NDArray.array(Array(7, 8), Shape(2, 1))).toArray
    checkEqual("Gemv(true, 0.5, Array(Array(1, 2), Array(3, 4)), Array(5, 6), 0.05, y) y(0)", y(0) - 11.85)
    checkEqual("Gemv(true, 0.5, Array(Array(1, 2), Array(3, 4)), Array(5, 6), 0.05, y) y(1)", y(1) - 17.4)

    y = Ger(0.5f, NDArray.array(Array(5, 6), Shape(2, 1)), NDArray.array(Array(7, 8), Shape(2, 1)),
      NDArray.array(Array(1, 2, 3, 4), Shape(2, 2))).toArray
    checkEqual("Ger(0.5, Array(5, 6), Array(7, 8), a) a(0)(0)", y(0) - 18.5)
    checkEqual("Ger(0.5, Array(5, 6), Array(7, 8), a) a(0)(1)", y(1) - 22.0)
    checkEqual("Ger(0.5, Array(5, 6), Array(7, 8), a) a(1)(0)", y(2) - 24.0)
    checkEqual("Ger(0.5, Array(5, 6), Array(7, 8), a) a(1)(1)", y(3) - 28.0)
  } catch { case e: Exception =>
    t.Errorf(s"[MATH] wrong math Exception: $e")
  }
}

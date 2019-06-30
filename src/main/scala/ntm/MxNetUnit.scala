package com.scalaml.ntm

import org.apache.mxnet._

// An unit is a node in a neural network, containing fields that are essential to
// efficiently compute gradients in the backward pass of a stochastic gradient
// descent training process.
class MxNetUnit (
  var Val: NDArray = NDArray.zeros(1), // value at node
  var Grad: NDArray = NDArray.zeros(1) // gradient at node
){
  def String(): String = s"{${Val.toArray.mkString(",")} ${Grad.toArray.mkString(",")}}"
}

object MxNetUnit {

  def makeMxNetUnit1(n: Int): MxNetUnit =
    new MxNetUnit(NDArray.zeros(n, 1), NDArray.zeros(n, 1))

  def makeMxNetUnit2(n: Int, m: Int): MxNetUnit =
    new MxNetUnit(NDArray.zeros(n, m), NDArray.zeros(n, m))

  def makeMxNetUnit3(n: Int, m: Int, p: Int): MxNetUnit =
    new MxNetUnit(NDArray.zeros(n, m, p), NDArray.zeros(n, m, p))
  
}
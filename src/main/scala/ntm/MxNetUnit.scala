package com.scalaml.ntm

import org.apache.mxnet._

// An unit is a node in a neural network, containing fields that are essential to
// efficiently compute gradients in the backward pass of a stochastic gradient
// descent training process.
class MxNetUnit (
  var Val: NDArray = NDArray.zeros(1), // value at node
  var Grad: NDArray = NDArray.zeros(1) // gradient at node
){
  def String(): String = s"{$Val $Grad}"
}

object MxNetUnit {

  def makeTensorUnit2(n: Int, m: Int): Array[Array[MxNetUnit]] =
    Array.fill[MxNetUnit](n, m)(new MxNetUnit)

  def makeTensorUnit3(n: Int, m: Int, p: Int): Array[Array[Array[MxNetUnit]]] =
    Array.fill[MxNetUnit](n, m, p)(new MxNetUnit)

  def doUnit1(t: Array[MxNetUnit], f: MxNetUnit => Unit): Unit =
    t.foreach(u => f(u))

  def doUnit2(t: Array[Array[MxNetUnit]], f: MxNetUnit => Unit): Unit =
    t.foreach(a => doUnit1(a, f))

  def doUnit3(t: Array[Array[Array[MxNetUnit]]], f: MxNetUnit => Unit): Unit =
    t.foreach(a => doUnit2(a, f))
}
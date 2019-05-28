package com.scalaml.ntm

// An unit is a node in a neural network, containing fields that are essential to
// efficiently compute gradients in the backward pass of a stochastic gradient
// descent training process.
class unit (
  var Val: Double = 0.0, // value at node
  var Grad: Double = 0.0 // gradient at node
){
  def String(): String = s"{$Val $Grad}"
}

object unit {

  def makeTensorUnit2(n: Int, m: Int): Array[Array[unit]] = {
    val t = new Array[Array[unit]](n)
    for(i <- 0 until t.size) {
      t(i) = Array.fill[unit](m)(new unit)
    }
    t
  }

  def makeTensorUnit3(n: Int, m: Int, p: Int): Array[Array[Array[unit]]] = {
    val t = new Array[Array[Array[unit]]](n)
    for(i <- 0 until t.size) {
      t(i) = makeTensorUnit2(m, p)
    }
    t
  }

  def doUnit1(t: Array[unit], f: unit => Unit): Unit = {
    for(i <- 0 until t.size) {
      f(t(i))
    }
  }

  def doUnit2(t: Array[Array[unit]], f: unit => Unit): Unit = {
    for(v <- t) {
      doUnit1(v, f)
    }
  }

  def doUnit3(t: Array[Array[Array[unit]]], f: unit => Unit): Unit = {
    for(v <- t) {
      doUnit2(v, f)
    }
  }
}
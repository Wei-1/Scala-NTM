# Scala Neural Turning Machine

[![Build Status](https://travis-ci.org/Wei-1/Scala-NTM.svg?branch=master)](https://travis-ci.org/Wei-1/Scala-NTM)

This is a hard coded Neural Turning Machine (NTM) ported from
[fumin/ntm](https://github.com/fumin/ntm).

We use 2D instead of 1D Array for value and gradient memory
so the array objects can replace the pointers in Go.

Since this is ported from Go,
a lot of implementations are still far from optimal in Scala. [(Notes)](#notes)

However, instead of focusing on adapting the Scala style.

I will start adding test cases from the paper first.

## TESTS:

- [x] Unit Tests
- [x] Copy Task
- [x] Repeat Copy
- [x] nGram

## STATUS:

Unit tests are successful,
and had done a 1 to 1 value comparison with fumin's project.
(tested with the same random seed)

## USAGES:

Run the unit tests: `sbt test`

Build the Jar file: `sbt package`

Run examples: `sbt run`

## NOTES:

Currently, the computation performance is 1000 times worse than the [Go version](https://github.com/fumin/ntm).

The comparison is based on the CopyTask example and RepeatCopy example computation time on my laptop.

Therefore, all examples are down scaled for reasonable testing time.

Please don't hesitate to contribute if you find any computation bottleneck.

Possible Bottleneck Improvements:

1. Replace those for loops with while loops

2. Replace Array with IndexSeq or other collection classes

3. Use NDArray with [MxNet](https://github.com/apache/incubator-mxnet) to improve tensor computations. Since this repo is for no dependency, might create another repo.

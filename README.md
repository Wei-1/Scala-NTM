# Scala Neural Turning Machine

[![Build Status](https://travis-ci.org/Wei-1/Scala-NTM.svg?branch=master)](https://travis-ci.org/Wei-1/Scala-NTM)

This is a hard coded Neural Turning Machine (NTM) ported from
[fumin/ntm](https://github.com/fumin/ntm).

We use 2D instead of 1D Array for value and gradient memory
so the array objects can replace the pointers in Go.

Since this is ported from Go,
a lot of implementations are still far from optimal in Scala.

However, instead of focusing on adapting the Scala style.

I will start adding test cases from the paper first.

## TESTS:

- [x] Unit Tests
- [x] Copy Task [main](src/main/scala/ntm/example/copytask.scala)[test](src/test/scala/ntm/example/copytask_test.scala)
- [ ] Repeat Copy - WIP
- [ ] nGram

## STATUS:

Unit tests are successful,
and had done a 1 to 1 value comparison with fumin's project.
(tested with the same random seed)

## USAGES:

Run the unit tests: `sbt test`

Build the Jar file: `sbt package`

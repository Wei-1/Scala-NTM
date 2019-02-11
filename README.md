# Scala Neural Turning Machine

This is a hard coded Neural Turning Machine (NTM) ported from
[fumin/ntm](https://github.com/fumin/ntm).

We use 2D instead of 1D Array for value and gradient memory
so the array objects can replace the pointers in Go.

Since this is ported from Go,
a lot of implementations are still far from optimal in Scala.

However, instead of focusing on adapting the Scala style.

I will start adding test cases from the paper first.

## TODO:

- [ ] Unit Tests (WIP)
- [ ] Copy Task
- [ ] Repeat Copy
- [ ] nGram

## STATUS:

Currently,
without proper tests, one can only check the code with:

```
sbt run
```

Well... at least there is no error from the compiler.

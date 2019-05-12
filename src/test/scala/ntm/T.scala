package ntm

import org.scalatest.FunSuite

class T() {
  var totCount: Int = 0
  var errCount: Int = 0
  val startTime = System.currentTimeMillis
  def Fatalf(s: String): Unit = {
    totCount += 1
    errCount += 1
    Console.err.println(Console.RED + s + Console.RESET)
  }
  def Errorf = Fatalf _
  def Logf(s: String): Unit = {
    totCount += 1
    println(Console.GREEN + s + Console.RESET)
  }
  def endT: Unit = {
    println(
      Console.BLUE + "Time: " +
      ((System.currentTimeMillis - startTime) / 1000.0) +
      " Seconds" + Console.RESET
    )
    if(errCount > 0) {
      Errorf(s"TEST FAILED with $errCount of Errors out of $totCount Tests")
      System.exit(1) 
    } else {
      Logf(s"TEST PASSED with all $totCount Tests")
      System.exit(0)
    }
  }
}

class NTMSuite extends FunSuite {
  val t = new T()
  test("Addressing") {
    addressing_test.TestCircuit(t)
  }
  test("CTRL_SaveLoad") {
    ctrl_test.TestSaveLoad(t)
  }
  test("CTRL_Logistic") {
    ctrl_test.TestLogisticModel(t)
  }
  test("CTRL_Multinomial") {
    ctrl_test.TestMultinomialModel(t)
  }
  test("NTM") {
    ntm_test.TestRMSProp(t)
  }
  test("Example - CopyTask") {
    ntm.example.copytask_test.TestRun(t)
  }
  test("Example - RepeatCopy") {
    ntm.example.repeatcopy_test.TestRun(t)
    t.endT
  }
}

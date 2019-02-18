package ntm

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

object T {
  def main(args: Array[String]) {
    val t = new T()
    t.Logf("Addressing:")
    addressing_test.TestCircuit(t)
    t.Logf("CTRL_Logistic:")
    ctrl_test.TestLogisticModel(t)
    t.Logf("CTRL_Multinomial:")
    ctrl_test.TestMultinomialModel(t)
    t.Logf("NTM:")
    ntm_test.TestRMSProp(t)
    
    t.endT
  }
}

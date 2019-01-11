package ntm

class T() {
  var hasError: Boolean = false
  val startTime = System.currentTimeMillis
  def Fatalf(s: String): Unit = {
    hasError = true
    Console.err.println(Console.RED + s + Console.RESET)
  }
  def Errorf = Fatalf _
  def Logf(s: String): Unit = println(Console.GREEN + s + Console.RESET)
  def endT: Unit = {
    println(
      Console.BLUE + "Time: " +
      ((System.currentTimeMillis - startTime) / 1000.0) +
      " Seconds" + Console.RESET
    )
    if(hasError) System.exit(1) else System.exit(0)
    println("Wei's Test")
  }
}

object T {
  def main(args: Array[String]) {
    val t = new T()

    addressing_test.TestCircuit(t)
    // cntl1_test.TestLogisticModel(t)
    // cntl1_test.TestMultinomialModel(t)
    // ntm_test.TestRMSProp(t)
    
    t.endT
  }
}

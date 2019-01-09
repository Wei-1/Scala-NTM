package ntm

class T {
	var hasError = false
	def Fatalf(s: String): Unit = {
		hasError = true
		Console.err.println(Console.RED + s + Console.RESET)
	}
	def Logf(s: String): Unit = println(Console.GREEN + s + Console.RESET)
	def endT: Unit = if(hasError) System.exit(1) else System.exit(0)
}
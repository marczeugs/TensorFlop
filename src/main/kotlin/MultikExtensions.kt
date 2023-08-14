import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.get

fun <T> MultiArray<T, D2>.rows() = (0..<shape[0]).map { this[it] }

import io.kotest.core.spec.style.StringSpec
import org.jetbrains.kotlinx.multik.api.identity
import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.rand
import org.jetbrains.kotlinx.multik.ndarray.operations.minus
import org.jetbrains.kotlinx.multik.ndarray.operations.plus
import org.jetbrains.kotlinx.multik.ndarray.operations.stack
import org.jetbrains.kotlinx.multik.ndarray.operations.times
import kotlin.time.measureTime

class MultikSpeedTest : StringSpec({
    val a = mk.rand<Double>(1000, 1000)
    val b = mk.rand<Double>(1000, 1000)

    "a + b" {
        println(measureTime { repeat(100) { a + b } })
    }

    "a * b" {
        println(measureTime { repeat(100) { a * b } })
    }

    "a dot b" {
        println(measureTime { repeat(100) { a dot b } })
    }

    "softmax derivative" {
        val column = mk.rand<Double>(1000)

        println(measureTime { repeat(2048) {
            val stackedHorizontally = mk.stack(List(column.size) { column }, axis = 1)
            val identity = mk.identity<Double>(column.size)

            stackedHorizontally * (identity - stackedHorizontally.transpose())
        } })
    }
})
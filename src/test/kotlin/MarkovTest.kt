
import io.kotest.core.spec.style.BehaviorSpec
import kotlinx.serialization.ExperimentalSerializationApi
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.decodeFromStream
import org.jetbrains.kotlinx.multik.api.d1array
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.toNDArray
import org.jetbrains.kotlinx.multik.ndarray.operations.toList
import java.time.Duration
import java.time.Instant
import kotlin.random.Random

fun main() {
    // language=regexp
    val wordPartEndMarkerRegex = "[/()\\-_?!]"

    val wordPartFinderRegex = Regex(" *[A-Z0]+?.?(?=\\s+|$|$wordPartEndMarkerRegex|(?<=$wordPartEndMarkerRegex))| *.+?(?=[A-Z]|\\s+|$|$wordPartEndMarkerRegex|(?<=$wordPartEndMarkerRegex))")

    fun String.toWordParts() = listOf(null) + wordPartFinderRegex.findAll(this).map { it.value }.toList()


    val start = Instant.now()


    @OptIn(ExperimentalSerializationApi::class)
    val messages = Json.decodeFromStream<List<String>>(MarkovTest::class.java.getResource("data.json")!!.openStream())

    val mostUsedWordParts = messages
        .flatMap { it.toWordParts() }
        .groupingBy { it }
        .eachCount()
        .entries
        .sortedByDescending { it.value }
        .map { it.key }
        .take(1000)
        .sortedWith { o1, o2 -> o2?.let { o1?.compareTo(it) } ?: 0 }

    fun String?.wordPartToEncodedArray() =
        mk.d1array(mostUsedWordParts.size) {
            if (it == mostUsedWordParts.indexOf(this)) {
                1.0
            } else {
                0.0
            }
        }

    val (inputs, targets) = messages
        .map { it.toWordParts() + listOf(null) }
        .filter { wordParts -> wordParts.all { it in mostUsedWordParts } }
        .take(2000)
        .flatMap { wordParts ->
            wordParts.windowed(3)
                .map { windowedWordParts ->
                    windowedWordParts.map { it.wordPartToEncodedArray() }
                }
                .map { encodedWindowedWordParts ->
                    encodedWindowedWordParts.take(2).flatMap { it.toList() }.toNDArray() to encodedWindowedWordParts.drop(2).first()
                }
        }
        .unzip()

    val network = NeuralNetwork.withEmptyLayers(
        sizes = listOf(mostUsedWordParts.size * 2, 1000, 1000, mostUsedWordParts.size),
        weightInitializer = NeuralNetwork.WeightInitializers.Xavier,
        hiddenLayerActivationFunction = NeuralNetwork.ActivationFunctions.relu,
        outputLayerActivationFunction = NeuralNetwork.ActivationFunctions.softmax
    )

    println("Before training: ${Duration.between(start, Instant.now())}")

    network.train(
        inputs = inputs,
        targets = targets,
        epochs = 1,
        batchSize = 2048,
        validationPercentage = 0.1,
        lossFunction = NeuralNetwork.LossFunctions.crossEntropy,
        optimizer = NeuralNetwork.Optimizer.StochasticGradientDescent(learningRate = 0.2),
        regularizationFunction = NeuralNetwork.RegularizationFunctions.L1()
    )

    println("After training: ${Duration.between(start, Instant.now())}")

    fun generateAnswer() {
        val output = mutableListOf(null, mostUsedWordParts.random())

        while (output.last() != null && output.size < 30) {
            output.add(
                network.predict(
                    output.takeLast(2)
                        .map { it.wordPartToEncodedArray() }
                        .flatMap { it.toList() }
                        .toNDArray()
                )
                    .let { predictedOutput ->
                        val summedProbabilities = predictedOutput.toList().runningFold(0.0) { sum, next -> sum + next }
                        val random = Random.nextDouble()

                        mostUsedWordParts[summedProbabilities.indexOfFirst { it > random } - 1]
                    }
            )
        }

        println(output.drop(1).dropLast(1).joinToString(""))
    }

    repeat(100) {
        generateAnswer()
    }

    println("End: ${Duration.between(start, Instant.now())}")
}

class MarkovTest : BehaviorSpec({
    given("a neural network") {
        // language=regexp
        val wordPartEndMarkerRegex = "[/()\\-_?!]"

        val wordPartFinderRegex = Regex(" *[A-Z0]+?.?(?=\\s+|$|$wordPartEndMarkerRegex|(?<=$wordPartEndMarkerRegex))| *.+?(?=[A-Z]|\\s+|$|$wordPartEndMarkerRegex|(?<=$wordPartEndMarkerRegex))")

        fun String.toWordParts() = listOf(null) + wordPartFinderRegex.findAll(this).map { it.value }.toList()


        @OptIn(ExperimentalSerializationApi::class)
        val messages = Json.decodeFromStream<List<String>>(MarkovTest::class.java.getResource("data.json")!!.openStream())

        val mostUsedWordParts = messages
            .flatMap { it.toWordParts() }
            .groupingBy { it }
            .eachCount()
            .entries
            .sortedByDescending { it.value }
            .map { it.key }
            .take(1000)

        fun String?.wordPartToEncodedArray() =
            mk.d1array(mostUsedWordParts.size) {
                if (it == mostUsedWordParts.indexOf(this)) {
                    1.0
                } else {
                    0.0
                }
            }

        val (inputs, targets) = messages
            .map { it.toWordParts() + listOf(null) }
            .filter { wordParts -> wordParts.all { it in mostUsedWordParts } }
            .take(2000)
            .flatMap { wordParts ->
                wordParts.windowed(3)
                    .map { windowedWordParts ->
                        windowedWordParts.map { it.wordPartToEncodedArray() }
                    }
                    .map { encodedWindowedWordParts ->
                        encodedWindowedWordParts.take(2).flatMap { it.toList() }.toNDArray() to encodedWindowedWordParts.drop(2).first()
                    }
            }
            .unzip()

        val network = NeuralNetwork.withEmptyLayers(
            sizes = listOf(mostUsedWordParts.size * 2, 1000, 1000, mostUsedWordParts.size),
            weightInitializer = NeuralNetwork.WeightInitializers.Xavier,
            hiddenLayerActivationFunction = NeuralNetwork.ActivationFunctions.relu,
            outputLayerActivationFunction = NeuralNetwork.ActivationFunctions.softmax
        )

        `when`("we train it on r/forsen comments as input") {
            network.train(
                inputs = inputs,
                targets = targets,
                epochs = 5,
                batchSize = 32,
                validationPercentage = 0.1,
                lossFunction = NeuralNetwork.LossFunctions.crossEntropy,
                optimizer = NeuralNetwork.Optimizer.StochasticGradientDescent(learningRate = 0.2),
                regularizationFunction = NeuralNetwork.RegularizationFunctions.L1()
            )

            then("we get reasonable comments as output") {
                fun generateAnswer() {
                    val output = mutableListOf(null, mostUsedWordParts.random())

                    while (output.last() != null && output.size < 30) {
                        output.add(
                            network.predict(
                                output.takeLast(2)
                                    .map { it.wordPartToEncodedArray() }
                                    .flatMap { it.toList() }
                                    .toNDArray()
                            )
                                .let { predictedOutput ->
                                    val summedProbabilities = predictedOutput.toList().runningFold(0.0) { sum, next -> sum + next }
                                    val random = Random.nextDouble()

                                    mostUsedWordParts[summedProbabilities.indexOfFirst { it > random } - 1]
                                }
                        )
                    }

                    println(output.dropLast(1).joinToString(""))
                }

                repeat(100) {
                    generateAnswer()
                }
            }
        }
    }
})
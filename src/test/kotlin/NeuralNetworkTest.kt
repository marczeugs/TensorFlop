
import io.kotest.core.spec.style.BehaviorSpec
import io.kotest.matchers.doubles.shouldBeGreaterThan
import io.kotest.matchers.doubles.shouldBeLessThan
import io.kotest.matchers.shouldBe
import org.jetbrains.kotlinx.multik.api.math.argMax
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.api.toNDArray
import org.jetbrains.kotlinx.multik.ndarray.operations.first
import org.knowm.xchart.BitmapEncoder
import org.knowm.xchart.XYChartBuilder
import java.io.File
import kotlin.io.path.Path
import kotlin.io.path.createParentDirectories

class NeuralNetworkTest : BehaviorSpec() {
    private val failedTests = mutableListOf<Pair<String, NeuralNetwork.TrainingOutput>>()

    private lateinit var lastTrainingOutput: NeuralNetwork.TrainingOutput

    init {
        afterTest { (testCase, result) ->
            if (result.isErrorOrFailure) {
                failedTests += testCase.name.testName to lastTrainingOutput
            }
        }

        afterSpec {
            File("errorgraphs").deleteRecursively()

            failedTests.forEach { (testName, trainingOutputs) ->
                BitmapEncoder.saveBitmap(
                    XYChartBuilder().apply {
                        width(800)
                        height(600)
                        xAxisTitle("Epoch")
                        yAxisTitle("Value")
                        title(testName)
                    }.build().apply {
                        styler.setYAxisLogarithmic(true)
                        styler.markerSize = 0

                        addSeries("Validation Loss", trainingOutputs.validationLosses)
                        //addSeries("Validation Accuracy", trainingOutputs.third)
                        addSeries("Training Loss", trainingOutputs.trainingLosses)
                    },
                    Path("errorgraphs/$testName.png").createParentDirectories().toString(),
                    BitmapEncoder.BitmapFormat.PNG
                )
            }
        }

        given("an XOR predicting neural network") {
            val xorNetwork = NeuralNetwork.withEmptyLayers(
                sizes = listOf(2, 20, 20, 1),
                weightInitializer = NeuralNetwork.WeightInitializers.Xavier,
                hiddenLayerActivationFunction = NeuralNetwork.ActivationFunctions.relu,
                outputLayerActivationFunction = NeuralNetwork.ActivationFunctions.sigmoid
            )

            `when`("we train it on XOR data") {
                lastTrainingOutput = xorNetwork.train(
                    inputs = listOf(
                        mk.ndarray(mk[0.0, 0.0]),
                        mk.ndarray(mk[1.0, 0.0]),
                        mk.ndarray(mk[0.0, 1.0]),
                        mk.ndarray(mk[1.0, 1.0]),
                    ),
                    targets = listOf(
                        mk.ndarray(mk[0.0]),
                        mk.ndarray(mk[1.0]),
                        mk.ndarray(mk[1.0]),
                        mk.ndarray(mk[0.0]),
                    ),
                    validationPercentage = 0.0,
                    epochs = 100,
                    batchSize = 4,
                    lossFunction = NeuralNetwork.LossFunctions.meanSquaredError,
                    optimizer = NeuralNetwork.Optimizer.StochasticGradientDescent()
                )

                then("it should predict XOR data correctly") {
                    xorNetwork.predict(mk.ndarray(mk[0.0, 0.0])).first() shouldBeLessThan 0.1
                    xorNetwork.predict(mk.ndarray(mk[1.0, 0.0])).first() shouldBeGreaterThan 0.9
                    xorNetwork.predict(mk.ndarray(mk[0.0, 1.0])).first() shouldBeGreaterThan 0.9
                    xorNetwork.predict(mk.ndarray(mk[1.0, 1.0])).first() shouldBeLessThan 0.1
                }
            }
        }

        given("an Iris dataset neural network") {
            val (inputs, targets) = NeuralNetworkTest::class.java.getResource("iris.data")!!.readText()
                .lines()
                .map {
                    it.split(",").let { (i0, i1, i2, i3, o) ->
                        listOf(i0, i1, i2, i3).map(String::toDouble).toNDArray() to when (o) {
                            "Iris-setosa" -> mk.ndarray(mk[1.0, 0.0, 0.0])
                            "Iris-versicolor" -> mk.ndarray(mk[0.0, 1.0, 0.0])
                            "Iris-virginica" -> mk.ndarray(mk[0.0, 0.0, 1.0])
                            else -> error("your mom")
                        }
                    }
                }
                .unzip()

            val irisNetwork = NeuralNetwork.withEmptyLayers(
                sizes = listOf(4, 64, 64, 3),
                weightInitializer = NeuralNetwork.WeightInitializers.Xavier,
                hiddenLayerActivationFunction = NeuralNetwork.ActivationFunctions.relu,
                outputLayerActivationFunction = NeuralNetwork.ActivationFunctions.softmax
            )

            `when`("we train it on the Iris dataset") {
                lastTrainingOutput = irisNetwork.train(
                    inputs = inputs,
                    targets = targets,
                    epochs = 100,
                    batchSize = 32,
                    validationPercentage = 0.2,
                    lossFunction = NeuralNetwork.LossFunctions.meanSquaredError,
                    optimizer = NeuralNetwork.Optimizer.StochasticGradientDescent()
                )

                then("it should predict the flowers correctly") {
                    irisNetwork.predict(mk.ndarray(mk[6.9, 3.2, 5.7, 2.3])).argMax() shouldBe 2
                    irisNetwork.predict(mk.ndarray(mk[6.6, 3.0, 4.4, 1.4])).argMax() shouldBe 1
                    irisNetwork.predict(mk.ndarray(mk[5.1, 3.8, 1.9, 0.4])).argMax() shouldBe 0
                }
            }
        }
    }
}
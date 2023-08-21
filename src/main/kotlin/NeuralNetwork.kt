
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.double
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonPrimitive
import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.api.math.argMax
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.*
import java.util.random.RandomGenerator
import kotlin.math.*
import kotlin.random.Random

class NeuralNetwork private constructor(private val layers: List<Layer>) {
    companion object {
        fun withEmptyLayers(
            sizes: List<Int>,
            weightInitializer: (previousLayerSize: Int, currentLayerSize: Int) -> Double,
            hiddenLayerActivationFunction: ActivationFunction,
            outputLayerActivationFunction: ActivationFunction,
        ) =
            NeuralNetwork(
                sizes.windowed(2).mapIndexed { index, (previousLayerSize, currentLayerSize) ->
                    Layer(
                        biases = mk.d1array(currentLayerSize) { 0.0 }.expandDims(0),
                        weights = mk.d2array(previousLayerSize, currentLayerSize) {
                            weightInitializer(previousLayerSize, currentLayerSize)
                        },
                        activationFunction = if (index < sizes.size - 2) {
                            hiddenLayerActivationFunction
                        } else {
                            outputLayerActivationFunction
                        },
                    )
                }
            )

        fun fromJson(model: JsonElement, hiddenLayerActivationFunction: ActivationFunction, outputLayerActivationFunction: ActivationFunction) =
            NeuralNetwork(
                model.jsonArray.chunked(2).mapIndexed { index, (weights, biases) ->
                    Layer(
                        weights = weights.jsonArray
                            .map { row -> row.jsonArray.map { it.jsonPrimitive.double } }
                            .toNDArray()
                            .reshape(weights.jsonArray.size, weights.jsonArray[0].jsonArray.size),
                        biases = biases.jsonArray.map { it.jsonPrimitive.double }.toNDArray().expandDims(0),
                        activationFunction = if (index < model.jsonArray.size / 2 - 1) {
                            hiddenLayerActivationFunction
                        } else {
                            outputLayerActivationFunction
                        }
                    )
                }
            )
    }

    fun predict(input: MultiArray<Double, D1>) =
        forwardPass(input.expandDims(0)).last().second.reshape(layers.last().weights.shape[1])

    private fun forwardPass(inputLayer: MultiArray<Double, D2>): List<Pair<MultiArray<Double, D2>, MultiArray<Double, D2>>> =
        layers.runningFold(inputLayer to inputLayer) { (_, previousNodes), currentLayer ->
            val activations =
                (previousNodes dot currentLayer.weights) +
                // We have to account for cases because the mk.stack function doesn't work for an array size of 1
                when (inputLayer.shape[0]) {
                    1 -> currentLayer.biases
                    else -> mk.stack(List(inputLayer.shape[0]) { currentLayer.biases.flatten() }, axis = 0)
                }

            activations to when (val f = currentLayer.activationFunction.f) {
                is ActivationFunction.Operation.SingleValue -> activations.map(f::invoke)
                is ActivationFunction.Operation.MultipleValues ->
                    // We have to account for cases because the mk.stack function doesn't work for an array size of 1
                    when (activations.shape[0]) {
                        1 -> f(activations)
                        else -> mk.stack(
                            (0..<activations.shape[0])
                                .map { f(activations[it].expandDims(0)).flatten() }
                        )
                    }
            }
        }

    private fun backwardPass(
        targetOutputLayers: MultiArray<Double, D2>,
        activations: List<MultiArray<Double, D2>>,
        hiddenStates: List<MultiArray<Double, D2>>,
        lossFunctionDerivative: (MultiArray<Double, D1>, MultiArray<Double, D1>) -> MultiArray<Double, D1>,
        regularizationFunctionDerivative: (MultiArray<Double, D2>) -> MultiArray<Double, D2>,
    ): Pair<List<MultiArray<Double, D2>>, List<MultiArray<Double, D2>>> {
        val batchSize = targetOutputLayers.shape[0]

        // Calculate loss function derivative
        var delta = mk.stack(
            (0..<batchSize)
                .map {
                    lossFunctionDerivative(
                        targetOutputLayers[it],
                        hiddenStates.last()[it]
                    )
                }
        )

        val biasGradients = MutableList(layers.size) { layers[it].biases }
        val weightGradients = MutableList(layers.size) { layers[it].weights }

        // Calculate rest of the deltas
        for (k in layers.indices.reversed()) {
            // Calculate loss function derivative dot hidden/output function derivative
            delta = when (val df = layers[k].activationFunction.df) {
                is ActivationFunction.Operation.SingleValue -> delta * activations[k + 1].map(df::invoke)
                is ActivationFunction.Operation.MultipleValues -> mk.stack(
                    (0..<batchSize) // batchSize is also delta.shape[0]
                        .map {
                            (delta[it].expandDims(0) dot df(activations[k + 1][it].expandDims(0))).flatten()
                        }
                )
            }

            // Calculate the bias gradients as the sum of gradients across the batch
            biasGradients[k] = delta.rows()
                .fold(mk.zeros<Double>(delta.shape[1])) { sum, next -> sum + next }
                .expandDims(0)

            weightGradients[k] = (hiddenStates[k].transpose() dot delta) + regularizationFunctionDerivative(layers[k].weights)

            delta = delta dot layers[k].weights.transpose()
        }

        return weightGradients to biasGradients
    }

    fun train(
        inputs: List<MultiArray<Double, D1>>,
        targets: List<MultiArray<Double, D1>>,
        epochs: Int,
        batchSize: Int = 32,
        validationPercentage: Double = 0.2,
        lossFunction: LossFunction = LossFunctions.meanSquaredError,
        optimizer: Optimizer = Optimizer.StochasticGradientDescent(),
        regularizationFunction: RegularizationFunction = RegularizationFunctions.None
    ): TrainingOutput {
        val trainingLosses = mutableListOf<Double>()
        val validationLosses = mutableListOf<Double>()
        val validationAccuracies = mutableListOf<Double>()

        // TODO: adam optimizer
        // TODO: learning rate decay

        val trainingData = inputs.zip(targets).shuffled()

        val actualTrainingData = trainingData.take(ceil(trainingData.size * (1 - validationPercentage)).roundToInt())
        val validationData = trainingData.drop(ceil(trainingData.size * (1 - validationPercentage)).roundToInt())

        repeat(epochs) { epoch ->
            val loss = actualTrainingData.windowed(size = batchSize, step = batchSize, partialWindows = true)
                .map { it.unzip() }
                .sumOf { (inputBatch, targetBatch) ->
                    val (activations, hiddenStates) = forwardPass(mk.stack(inputBatch)).unzip()

                    val (weightGradients, biasGradients) = backwardPass(
                        targetOutputLayers = mk.stack(targetBatch),
                        activations = activations,
                        hiddenStates = hiddenStates,
                        lossFunctionDerivative = lossFunction.df,
                        regularizationFunctionDerivative = regularizationFunction.df
                    )

                    optimizer.applyGradients(layers, weightGradients, biasGradients, batchSize)

                    targetBatch.zip(hiddenStates.last().rows()).sumOf { (target, hiddenState) -> lossFunction.f(target, hiddenState) } +
                        layers.sumOf { regularizationFunction.f(it.weights) } * batchSize
                }

            trainingLosses += loss / inputs.size

            val (validationLoss, validationAccuracy) = if (validationData.isNotEmpty()) {
                validationData.unzip()
                    .let { (validationInputs, validationTargets) ->
                        val (_, hiddenStates) = forwardPass(mk.stack(validationInputs)).unzip()

                        val predictionIndices: MultiArray<Int, D1> = hiddenStates.last().rows().map { it.argMax() }.toNDArray()
                        val correctIndices: MultiArray<Int, D1> = mk.stack(validationTargets).rows().map { it.argMax() }.toNDArray()

                        Pair(
                            validationTargets.zip(hiddenStates.last().rows())
                                .sumOf { (target, hiddenState) -> lossFunction.f(target, hiddenState) } / validationTargets.size,
                            predictionIndices.toList()
                                .zip(correctIndices.toList())
                                .count { (prediction, correct) -> prediction == correct }.toDouble() / predictionIndices.size
                        )
                    }
            } else {
                Double.NaN to Double.NaN
            }

            validationLosses += validationLoss
            validationAccuracies += validationAccuracy

            println("Epoch ${epoch + 1} - Training loss: ${trainingLosses.last()} - Validation loss: $validationLoss - Validation accuracy: $validationAccuracy")

            if (trainingLosses.last().isNaN()) {
                error("Losses are NaN")
            }
        }

        return TrainingOutput(
            trainingLosses = trainingLosses,
            validationLosses = validationLosses,
            validationAccuracies = validationAccuracies
        )
    }


    data class TrainingOutput(
        val trainingLosses: List<Double>,
        val validationLosses: List<Double>,
        val validationAccuracies: List<Double>,
    )


    data class Layer(
        val weights: MultiArray<Double, D2>,
        val biases: MultiArray<Double, D2>,
        val activationFunction: ActivationFunction
    )


    sealed class WeightInitializers(private val initializer: (Int, Int) -> Double) : (Int, Int) -> Double by initializer {
        data object He : WeightInitializers({ previousLayerSize, _ ->
            RandomGenerator.getDefault().nextGaussian(0.0, sqrt(2.0 / previousLayerSize))
        })

        data object Xavier : WeightInitializers({ previousLayerSize, currentLayerSize ->
            (sqrt(6.0) / sqrt(previousLayerSize.toDouble() + currentLayerSize)).let { Random.nextDouble(-it, it) }
        })
    }


    data class ActivationFunction(
        val f: Operation<*>,
        val df: Operation<*>
    ) {
        sealed class Operation<T>(private val f: (T) -> T) : (T) -> T by f {
            class SingleValue(f: (Double) -> Double) : Operation<Double>(f)
            class MultipleValues(f: (MultiArray<Double, D2>) -> MultiArray<Double, D2>) : Operation<MultiArray<Double, D2>>(f)
        }
    }

    object ActivationFunctions {
        val simple = ActivationFunction(
            f = ActivationFunction.Operation.SingleValue { it },
            df = ActivationFunction.Operation.SingleValue { 1.0 }
        )


        val relu = ActivationFunction(
            f = ActivationFunction.Operation.SingleValue {
                if (it <= 0) {
                    0.0
                } else {
                    it
                }
            },
            df = ActivationFunction.Operation.SingleValue {
                if (it <= 0) {
                    0.0
                } else {
                    1.0
                }
            }
        )


        private fun softmax(values: MultiArray<Double, D2>): MultiArray<Double, D2> {
            val exponentiatedShiftedValues = (values - mk.math.max(values)).map { exp(it) }
            return exponentiatedShiftedValues / exponentiatedShiftedValues.map { it +  Double.MIN_VALUE }.sum()
        }

        val softmax: ActivationFunction = ActivationFunction(
            f = ActivationFunction.Operation.MultipleValues(::softmax),
            df = ActivationFunction.Operation.MultipleValues { values ->
                val column = softmax(values).flatten()
                val stackedHorizontally = mk.stack(List(values.size) { column }, axis = 1)
                val identity = mk.identity<Double>(values.size)

                stackedHorizontally * (identity - stackedHorizontally.transpose())
            }
        )


        private fun sigmoid(value: Double) = 1.0 / (1.0 + exp(-value))

        val sigmoid: ActivationFunction = ActivationFunction(
            f = ActivationFunction.Operation.SingleValue(::sigmoid),
            df = ActivationFunction.Operation.SingleValue {
                sigmoid(it) * (1.0 - sigmoid(it))
            }
        )
    }


    data class LossFunction(
        val f: (targetOutput: MultiArray<Double, D1>, computedOutput: MultiArray<Double, D1>) -> Double,
        val df: (targetOutput: MultiArray<Double, D1>, computedOutput: MultiArray<Double, D1>) -> MultiArray<Double, D1>
    )

    object LossFunctions {
        val meanSquaredError = LossFunction(
            f = { targetOutput, computedOutput ->
                (1.0 / computedOutput.size) * ((targetOutput - computedOutput) * (targetOutput - computedOutput)).sum()
            },
            df = { targetOutput, computedOutput ->
                (1.0 / computedOutput.size) * -2.0 * (targetOutput - computedOutput)
            }
        )

        val crossEntropy = LossFunction(
            f = { targetOutput, computedOutput ->
                (-targetOutput * computedOutput.map { ln(it) }).sum()
            },
            df = { targetOutput, computedOutput ->
                computedOutput - targetOutput // TODO: confirm this
            }
        )
    }


    interface Optimizer {
        fun applyGradients(layers: List<Layer>, weightGradients: List<MultiArray<Double, D2>>, biasGradients: List<MultiArray<Double, D2>>, batchSize: Int)

        class StochasticGradientDescent(
            private val learningRate: Double = 0.1,
            private val momentum: Double = 0.9
        ) : Optimizer {
            private var weightVelocities: MutableList<MultiArray<Double, D2>>? = null
            private var biasVelocities: MutableList<MultiArray<Double, D2>>? = null

            private var epoch = 0

            override fun applyGradients(layers: List<Layer>, weightGradients: List<MultiArray<Double, D2>>, biasGradients: List<MultiArray<Double, D2>>, batchSize: Int) {
                if (weightVelocities == null && biasVelocities == null) {
                    weightVelocities = MutableList(layers.size) { index -> layers[index].weights.shape.let { mk.zeros<Double>(it[0], it[1]) } }
                    biasVelocities = MutableList(layers.size) { index -> layers[index].biases.shape.let { mk.zeros<Double>(it[0], it[1]) } }
                }

                layers.forEachIndexed { index, layer ->
                    weightVelocities!![index] = weightVelocities!![index] * momentum - learningRate * weightGradients[index] * (1.0 / batchSize)
                    biasVelocities!![index] = biasVelocities!![index] * momentum - learningRate * biasGradients[index] * (1.0 / batchSize)

                    (layer.weights as D2Array<Double>) += weightVelocities!![index]
                    (layer.biases as D2Array<Double>) += biasVelocities!![index]
                }

                epoch++
            }
        }
    }


    abstract class RegularizationFunction(
        val f: (weights: MultiArray<Double, D2>) -> Double,
        val df: (weights: MultiArray<Double, D2>) -> MultiArray<Double, D2>
    )

    object RegularizationFunctions {
        object None : RegularizationFunction(
            f = { 0.0 },
            df = { mk.zeros<Double>(it.shape[0], it.shape[1]) }
        )

        class L1(private val regularizationPenalty: Double = 0.01) : RegularizationFunction(
            f = { weights -> weights.map { it.absoluteValue }.sum() * regularizationPenalty },
            df = { weights ->  weights.map { if (it < 0) -regularizationPenalty else regularizationPenalty } }
        )

        class L2(private val regularizationPenalty: Double = 0.01) : RegularizationFunction(
            f = { weights -> weights.map { it.pow(2) }.sum() * regularizationPenalty },
            df = { weights ->  weights.map { 2.0 * it * regularizationPenalty } }
        )
    }
}
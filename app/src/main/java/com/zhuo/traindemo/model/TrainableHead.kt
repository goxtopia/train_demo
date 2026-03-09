package com.zhuo.traindemo.model

import com.zhuo.traindemo.math.MathUtils
import java.util.Random
import kotlin.math.sqrt

class TrainableHead(val inputDim: Int, val numClasses: Int, val intermediateChannels: Int = 1280) {
    // Weights: [inputDim, numClasses] (flattened)
    // Wait, if inputDim is the dimension BEFORE intermediateChannels, we need to clarify.
    // For original mode, linear expects inputDim -> numClasses.
    // For semi-unfrozen, linear expects intermediateChannels -> numClasses.
    // We will initialize linear weights based on intermediateChannels if intermediateChannels > 0 and inputDim != intermediateChannels.
    // However, it's easier to just assume inputDim is the input to the Linear layer if not semiUnfrozen.
    // Actually, in semi-unfrozen YOLOv8n, input features are 256. intermediateChannels is 1280. Linear is 1280 -> numClasses.
    // So linear input size should be intermediateChannels.
    val linearInputSize = intermediateChannels

    // Linear Head Parameters
    var weights = FloatArray(linearInputSize * numClasses)
    var bias = FloatArray(numClasses)

    // Adam optimizer state (Linear)
    private var mWeights = FloatArray(weights.size)
    private var vWeights = FloatArray(weights.size)
    private var mBias = FloatArray(bias.size)
    private var vBias = FloatArray(bias.size)
    var t = 0

    // Semi-Unfrozen Parameters (1x1 Conv)
    // Weights: [intermediateChannels, inputDim] for 1x1 Conv
    // Bias: [intermediateChannels]
    var convWeights = FloatArray(intermediateChannels * inputDim)
    var convBias = FloatArray(intermediateChannels)

    // Adam optimizer state (Conv)
    private var mConvWeights = FloatArray(convWeights.size)
    private var vConvWeights = FloatArray(convWeights.size)
    private var mConvBias = FloatArray(convBias.size)
    private var vConvBias = FloatArray(convBias.size)

    // Hyperparameters
    private val beta1 = 0.9f
    private val beta2 = 0.999f
    private val epsilon = 1e-8f
    private val dropoutRate = 0.3f // 50% dropout
    private val weightDecay = 0.05f // L2 regularization for AdamW
    private val maxGradNorm = 2.0f // Gradient clipping threshold
    private val random = Random()

    init {
        // Xavier initialization for Linear Head
        val limit = sqrt(6.0f / (linearInputSize + numClasses))
        for (i in weights.indices) {
            weights[i] = (random.nextFloat() * 2 * limit - limit)
        }
        // Bias initialized to 0

        // Xavier initialization for 1x1 Conv (if used without pretrained weights)
        val convLimit = sqrt(6.0f / (inputDim + intermediateChannels))
        for (i in convWeights.indices) {
            convWeights[i] = (random.nextFloat() * 2 * convLimit - convLimit)
        }
    }

    fun loadConvWeights(w: FloatArray, b: FloatArray) {
        if (w.size == convWeights.size && b.size == convBias.size) {
            System.arraycopy(w, 0, convWeights, 0, w.size)
            System.arraycopy(b, 0, convBias, 0, b.size)
        }
    }

    private fun sigmoid(x: Float): Float {
        return 1.0f / (1.0f + kotlin.math.exp(-x))
    }

    /**
     * Forward pass with Dropout support.
     * Input: [Batch, Channels, H, W] flattened. We perform GAP first -> [Batch, Channels].
     * Then Linear -> [Batch, NumClasses].
     * Returns: Logits [Batch, NumClasses] (flattened)
     */
    fun forward(input: FloatArray, batchSize: Int, channels: Int, height: Int, width: Int, training: Boolean = false, semiUnfrozen: Boolean = false): FloatArray {
        var gapOutput: FloatArray
        var actualChannels = channels

        if (semiUnfrozen) {
            // 1. 1x1 Conv
            // Input: [Batch, channels, height, width]
            // Output: [Batch, intermediateChannels, height, width]
            val convOutput = FloatArray(batchSize * intermediateChannels * height * width)
            val spatialSize = height * width
            for (b in 0 until batchSize) {
                for (oc in 0 until intermediateChannels) {
                    val biasVal = convBias[oc]
                    for (s in 0 until spatialSize) {
                        var sum = biasVal
                        for (ic in 0 until channels) {
                            sum += input[b * channels * spatialSize + ic * spatialSize + s] * convWeights[oc * channels + ic]
                        }
                        // Apply SiLU: x * sigmoid(x)
                        val sig = sigmoid(sum)
                        convOutput[b * intermediateChannels * spatialSize + oc * spatialSize + s] = sum * sig
                    }
                }
            }

            // 2. GAP
            gapOutput = globalAveragePool(convOutput, batchSize, intermediateChannels, height, width)
            actualChannels = intermediateChannels
        } else {
            gapOutput = globalAveragePool(input, batchSize, channels, height, width)
        }

        // Apply Dropout if training
        if (training) {
            // We apply dropout to the features after GAP (before Linear)
            for (i in gapOutput.indices) {
                if (random.nextFloat() < dropoutRate) {
                    gapOutput[i] = 0f
                } else {
                    // Scale inverted dropout
                    gapOutput[i] = gapOutput[i] / (1.0f - dropoutRate)
                }
            }
        }

        val output = FloatArray(batchSize * numClasses)
        for (b in 0 until batchSize) {
            for (c in 0 until numClasses) {
                var sum = bias[c]
                for (i in 0 until actualChannels) {
                    sum += gapOutput[b * actualChannels + i] * weights[i * numClasses + c]
                }
                output[b * numClasses + c] = sum
            }
        }
        return output
    }

    // Helper to store dropout mask for backward pass?
    // Since we do re-forward inside trainStep, we can implement dropout inside trainStep logic specifically
    // to keep the mask consistent.

    /**
     * Backward pass and Update.
     * Input: Original input features [Batch, C, H, W]
     * Logits: Output of forward pass (before softmax) [Batch, NumClasses]
     * Targets: Class indices [Batch]
     */
    fun trainStep(input: FloatArray, batchSize: Int, channels: Int, height: Int, width: Int, targets: IntArray, learningRate: Float, labelSmoothing: Float = 0.1f, semiUnfrozen: Boolean = false): Float {
        var gapOutput: FloatArray
        var actualChannels = channels
        var convInputPreSiLU: FloatArray? = null
        var convOutputPostSiLU: FloatArray? = null

        val spatialSize = height * width

        if (semiUnfrozen) {
            // 1. 1x1 Conv Forward
            convInputPreSiLU = FloatArray(batchSize * intermediateChannels * spatialSize)
            convOutputPostSiLU = FloatArray(batchSize * intermediateChannels * spatialSize)

            for (b in 0 until batchSize) {
                for (oc in 0 until intermediateChannels) {
                    val biasVal = convBias[oc]
                    for (s in 0 until spatialSize) {
                        var sum = biasVal
                        for (ic in 0 until channels) {
                            sum += input[b * channels * spatialSize + ic * spatialSize + s] * convWeights[oc * channels + ic]
                        }
                        convInputPreSiLU[b * intermediateChannels * spatialSize + oc * spatialSize + s] = sum
                        // Apply SiLU
                        val sig = sigmoid(sum)
                        convOutputPostSiLU[b * intermediateChannels * spatialSize + oc * spatialSize + s] = sum * sig
                    }
                }
            }

            // 2. GAP
            gapOutput = globalAveragePool(convOutputPostSiLU, batchSize, intermediateChannels, height, width)
            actualChannels = intermediateChannels
        } else {
            // 1. GAP
            gapOutput = globalAveragePool(input, batchSize, channels, height, width)
        }

        // 2. Apply Dropout and store mask
        val dropoutMask = FloatArray(gapOutput.size)
        for (i in gapOutput.indices) {
            if (random.nextFloat() < dropoutRate) {
                dropoutMask[i] = 0f
                gapOutput[i] = 0f
            } else {
                dropoutMask[i] = 1.0f / (1.0f - dropoutRate)
                gapOutput[i] = gapOutput[i] * dropoutMask[i]
            }
        }

        // 3. Linear Forward
        val logits = FloatArray(batchSize * numClasses)
        for (b in 0 until batchSize) {
            for (c in 0 until numClasses) {
                var sum = bias[c]
                for (i in 0 until actualChannels) {
                    sum += gapOutput[b * actualChannels + i] * weights[i * numClasses + c]
                }
                logits[b * numClasses + c] = sum
            }
        }

        // 4. Softmax & Loss (with Label Smoothing)
        val probs = FloatArray(logits.size)
        var totalLoss = 0f

        for (b in 0 until batchSize) {
            val offset = b * numClasses
            // Softmax for this sample
            var maxVal = Float.NEGATIVE_INFINITY
            for (i in 0 until numClasses) maxVal = maxOf(maxVal, logits[offset + i])

            var sum = 0f
            for (i in 0 until numClasses) {
                probs[offset + i] = kotlin.math.exp(logits[offset + i] - maxVal)
                sum += probs[offset + i]
            }
            for (i in 0 until numClasses) probs[offset + i] /= sum

            // Cross Entropy Loss with Label Smoothing
            // Target distribution: (1 - epsilon) * one_hot + epsilon / numClasses
            val targetIdx = targets[b]
            for (i in 0 until numClasses) {
                var targetProb = labelSmoothing / numClasses
                if (i == targetIdx) {
                    targetProb += (1.0f - labelSmoothing)
                }
                totalLoss -= targetProb * kotlin.math.ln(maxOf(probs[offset + i], 1e-7f))
            }
        }

        // 5. Backward
        // dL/dLogits = probs - target_distribution
        val dLogits = FloatArray(logits.size)
        for (b in 0 until batchSize) {
            val targetIdx = targets[b]
            for (c in 0 until numClasses) {
                var targetProb = labelSmoothing / numClasses
                if (c == targetIdx) {
                    targetProb += (1.0f - labelSmoothing)
                }

                dLogits[b * numClasses + c] = probs[b * numClasses + c] - targetProb
                dLogits[b * numClasses + c] /= batchSize.toFloat() // Average gradients
            }
        }

        // dL/dWeights = GAP(input)^T * dL/dLogits
        val dWeights = FloatArray(weights.size)
        // dL/dGap = dL/dLogits * Weights^T
        val dGap = FloatArray(gapOutput.size)

        for (i in 0 until actualChannels) {
            for (j in 0 until numClasses) {
                var sum = 0f
                for (b in 0 until batchSize) {
                    sum += gapOutput[b * actualChannels + i] * dLogits[b * numClasses + j]
                }
                dWeights[i * numClasses + j] = sum
            }
        }

        for (b in 0 until batchSize) {
            for (i in 0 until actualChannels) {
                var sum = 0f
                for (j in 0 until numClasses) {
                    sum += dLogits[b * numClasses + j] * weights[i * numClasses + j]
                }
                dGap[b * actualChannels + i] = sum * dropoutMask[b * actualChannels + i] // backprop through dropout
            }
        }

        // dL/dBias = sum(dL/dLogits) across batch
        val dBias = FloatArray(numClasses)
        for (j in 0 until numClasses) {
            var sum = 0f
            for (b in 0 until batchSize) {
                sum += dLogits[b * numClasses + j]
            }
            dBias[j] = sum
        }

        val dConvWeights = FloatArray(convWeights.size)
        val dConvBias = FloatArray(convBias.size)

        if (semiUnfrozen && convInputPreSiLU != null) {
            // Backprop through GAP
            // dL/dConvOutput = dL/dGap / spatialSize
            val dConvOutput = FloatArray(batchSize * intermediateChannels * spatialSize)
            for (b in 0 until batchSize) {
                for (oc in 0 until intermediateChannels) {
                    val grad = dGap[b * intermediateChannels + oc] / spatialSize
                    for (s in 0 until spatialSize) {
                        dConvOutput[b * intermediateChannels * spatialSize + oc * spatialSize + s] = grad
                    }
                }
            }

            // Backprop through SiLU
            // dSiLU = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
            for (i in dConvOutput.indices) {
                val x = convInputPreSiLU[i]
                val sig = sigmoid(x)
                val dSilu = sig + x * sig * (1.0f - sig)
                dConvOutput[i] *= dSilu
            }

            // Backprop through 1x1 Conv
            // dL/dConvWeight = sum(dL/dConvOutput * input)
            // dL/dConvBias = sum(dL/dConvOutput)
            for (b in 0 until batchSize) {
                for (oc in 0 until intermediateChannels) {
                    for (s in 0 until spatialSize) {
                        val grad = dConvOutput[b * intermediateChannels * spatialSize + oc * spatialSize + s]
                        dConvBias[oc] += grad
                        for (ic in 0 until channels) {
                            dConvWeights[oc * channels + ic] += grad * input[b * channels * spatialSize + ic * spatialSize + s]
                        }
                    }
                }
            }
        }

        // Gradient Clipping (L2 Norm)
        var globalNormSq = 0f
        for(g in dWeights) globalNormSq += g*g
        for(g in dBias) globalNormSq += g*g
        if (semiUnfrozen) {
            for(g in dConvWeights) globalNormSq += g*g
            for(g in dConvBias) globalNormSq += g*g
        }
        val globalNorm = kotlin.math.sqrt(globalNormSq)

        // If global norm is greater than threshold, clip gradients
        if (globalNorm > maxGradNorm) {
            val scale = maxGradNorm / (globalNorm + 1e-6f) // Add epsilon to avoid div by zero
            for(i in dWeights.indices) dWeights[i] *= scale
            for(i in dBias.indices) dBias[i] *= scale
            if (semiUnfrozen) {
                for(i in dConvWeights.indices) dConvWeights[i] *= scale
                for(i in dConvBias.indices) dConvBias[i] *= scale
            }
        }

        // 6. Update (AdamW)
        t++
        adamWUpdate(weights, dWeights, mWeights, vWeights, learningRate, applyDecay = true)
        adamWUpdate(bias, dBias, mBias, vBias, learningRate, applyDecay = false)
        if (semiUnfrozen) {
            adamWUpdate(convWeights, dConvWeights, mConvWeights, vConvWeights, learningRate, applyDecay = true)
            adamWUpdate(convBias, dConvBias, mConvBias, vConvBias, learningRate, applyDecay = false)
        }

        return totalLoss / batchSize
    }

    private fun globalAveragePool(input: FloatArray, batchSize: Int, channels: Int, height: Int, width: Int): FloatArray {
        val output = FloatArray(batchSize * channels)
        val spatialSize = (height * width).toFloat()

        for (b in 0 until batchSize) {
            for (c in 0 until channels) {
                var sum = 0f
                for (h in 0 until height) {
                    for (w in 0 until width) {
                        // Input layout assumed: NCHW flattened -> [b][c][h][w]
                        // Index: b*(C*H*W) + c*(H*W) + h*W + w
                        val idx = b * (channels * height * width) + c * (height * width) + h * width + w
                        sum += input[idx]
                    }
                }
                output[b * channels + c] = sum / spatialSize
            }
        }
        return output
    }

    private fun adamWUpdate(params: FloatArray, grads: FloatArray, m: FloatArray, v: FloatArray, lr: Float, applyDecay: Boolean) {
        val beta1Pow = Math.pow(beta1.toDouble(), t.toDouble()).toFloat()
        val beta2Pow = Math.pow(beta2.toDouble(), t.toDouble()).toFloat()

        // Typical AdamW applies decay like: theta = theta - lr * (grad + decay * theta)
        // Decoupled: theta = theta - lr * decay * theta - lr * adam_step

        for (i in params.indices) {
            // Apply weight decay
            if (applyDecay) {
                params[i] = params[i] * (1.0f - lr * weightDecay)
            }

            // Adam Update
            m[i] = beta1 * m[i] + (1 - beta1) * grads[i]
            v[i] = beta2 * v[i] + (1 - beta2) * grads[i] * grads[i]

            val mHat = m[i] / (1 - beta1Pow)
            val vHat = v[i] / (1 - beta2Pow)

            params[i] -= lr * mHat / (kotlin.math.sqrt(vHat) + epsilon)
        }
    }
}

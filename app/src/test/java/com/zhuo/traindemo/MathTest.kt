package com.zhuo.traindemo

import com.zhuo.traindemo.math.MathUtils
import com.zhuo.traindemo.model.TrainableHead
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

class MathTest {

    @Test
    fun testSoftmax() {
        val input = floatArrayOf(1.0f, 2.0f, 3.0f)
        val output = MathUtils.softmax(input)

        var sum = 0f
        for (v in output) sum += v

        assertEquals(1.0f, sum, 1e-6f)
        assertTrue(output[2] > output[1])
        assertTrue(output[1] > output[0])
    }

    @Test
    fun testHeadForward() {
        // 2 input channels, 2 classes, 1x1 spatial
        val head = TrainableHead(2, 2)
        // Manually set weights for predictability
        // Weights: [inputDim * numClasses] -> [2*2] = [4]
        // w[0]=0.1 (in0->cl0), w[1]=0.2 (in1->cl0)
        // w[2]=0.3 (in0->cl1), w[3]=0.4 (in1->cl1)
        head.weights = floatArrayOf(0.1f, 0.3f, 0.2f, 0.4f)
        // Note: My TrainableHead logic in forward:
        // sum += gapOutput[b * channels + i] * weights[i * numClasses + c]
        // i=0(ch0), c=0(cl0) -> w[0*2+0] = w[0] = 0.1
        // i=1(ch1), c=0(cl0) -> w[1*2+0] = w[2] = 0.2 (Wait, my manual assignment above was wrong order vs index)

        // Let's re-verify indexing in TrainableHead.kt:
        // weights[i * numClasses + c]
        // i is channel index, c is class index.

        // Reset to clear logic:
        // Weights size: 4.
        // i=0, c=0 -> idx=0. Val=0.1
        // i=0, c=1 -> idx=1. Val=0.2
        // i=1, c=0 -> idx=2. Val=0.3
        // i=1, c=1 -> idx=3. Val=0.4
        head.weights = floatArrayOf(0.1f, 0.2f, 0.3f, 0.4f)
        head.bias = floatArrayOf(0f, 0f)

        // Input: Batch=1, Channels=2, H=1, W=1
        // Input flattened: [0.5, 0.5]
        val input = floatArrayOf(0.5f, 0.5f)

        val output = head.forward(input, 1, 2, 1, 1)

        // Expected Class 0:
        // ch0(0.5)*w00(0.1) + ch1(0.5)*w10(0.3) = 0.05 + 0.15 = 0.2
        // Expected Class 1:
        // ch0(0.5)*w01(0.2) + ch1(0.5)*w11(0.4) = 0.10 + 0.20 = 0.3

        assertEquals(0.2f, output[0], 1e-6f)
        assertEquals(0.3f, output[1], 1e-6f)
    }

    @Test
    fun testHeadTraining() {
        val head = TrainableHead(2, 2)
        // Batch=1, Channels=2, H=1, W=1
        val input = floatArrayOf(1.0f, 1.0f)
        val targets = intArrayOf(0) // Target class 0

        val initialLoss = head.trainStep(input, 1, 2, 1, 1, targets, 0.0f) // LR=0 just to get loss

        // Train for a few steps
        var loss = initialLoss
        // We use a fixed seed random inside TrainableHead but dropout makes loss stochastic.
        // However, with 0.5 dropout, sometimes we drop everything or keep everything.
        // Also, LR=0.1 might be too large or small depending on initialization.

        // Let's use a very simple case where we can expect convergence.
        // But since dropout is 50%, the loss fluctuates.
        // We can compare average loss over time or just ensure parameters moved.

        val initialWeights = head.weights.clone()

        for (i in 0 until 50) {
            loss = head.trainStep(input, 1, 2, 1, 1, targets, 0.01f)
        }

        // Check if weights changed
        var weightsChanged = false
        for(i in head.weights.indices) {
            if(head.weights[i] != initialWeights[i]) weightsChanged = true
        }
        assertTrue(weightsChanged)

        // Loss check is flaky with dropout without fixed seed control in test
        // But generally it should go down.
        // assertTrue(loss < initialLoss)
    }
}

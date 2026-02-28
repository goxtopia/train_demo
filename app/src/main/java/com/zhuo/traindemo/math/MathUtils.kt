package com.zhuo.traindemo.math

import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max

object MathUtils {

    fun matMul(a: FloatArray, b: FloatArray, aRows: Int, aCols: Int, bCols: Int): FloatArray {
        if (a.size != aRows * aCols || b.size != aCols * bCols) {
            throw IllegalArgumentException("Matrix dimensions mismatch")
        }
        val result = FloatArray(aRows * bCols)
        for (i in 0 until aRows) {
            for (j in 0 until bCols) {
                var sum = 0f
                for (k in 0 until aCols) {
                    sum += a[i * aCols + k] * b[k * bCols + j]
                }
                result[i * bCols + j] = sum
            }
        }
        return result
    }

    fun addVector(a: FloatArray, b: FloatArray): FloatArray {
        if (a.size != b.size) {
            throw IllegalArgumentException("Vector dimensions mismatch")
        }
        val result = FloatArray(a.size)
        for (i in a.indices) {
            result[i] = a[i] + b[i]
        }
        return result
    }

    fun softmax(input: FloatArray): FloatArray {
        val result = FloatArray(input.size)
        var maxVal = Float.NEGATIVE_INFINITY
        for (v in input) {
            maxVal = max(maxVal, v)
        }
        var sum = 0f
        for (i in input.indices) {
            result[i] = exp(input[i] - maxVal)
            sum += result[i]
        }
        for (i in result.indices) {
            result[i] /= sum
        }
        return result
    }

    // Softmax for a batch (2D array flattened)
    fun softmaxBatch(input: FloatArray, batchSize: Int, numClasses: Int): FloatArray {
        val result = FloatArray(input.size)
        for (b in 0 until batchSize) {
            val offset = b * numClasses
            var maxVal = Float.NEGATIVE_INFINITY
            for (i in 0 until numClasses) {
                maxVal = max(maxVal, input[offset + i])
            }
            var sum = 0f
            for (i in 0 until numClasses) {
                result[offset + i] = exp(input[offset + i] - maxVal)
                sum += result[offset + i]
            }
            for (i in 0 until numClasses) {
                result[offset + i] /= sum
            }
        }
        return result
    }

    fun crossEntropyLoss(predictions: FloatArray, targetIndex: Int): Float {
        // predictions should be output of softmax
        val p = max(predictions[targetIndex], 1e-7f)
        return -ln(p)
    }

    fun argmax(array: FloatArray): Int {
        if (array.isEmpty()) return -1
        var maxIdx = 0
        var maxVal = array[0]
        for (i in 1 until array.size) {
            if (array[i] > maxVal) {
                maxVal = array[i]
                maxIdx = i
            }
        }
        return maxIdx
    }
}

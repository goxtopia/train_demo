package com.zhuo.traindemo.model

import android.content.Context
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File

class ModelManager(private val context: Context) {
    private val modelFile = File(context.filesDir, "trained_head.bin")

    fun saveModel(head: TrainableHead, classLabels: List<String>) {
        DataOutputStream(modelFile.outputStream()).use { dos ->
            // Version Byte: 0x01 indicates new format with Conv weights
            dos.writeByte(0x01)

            // Save Class Labels
            dos.writeInt(classLabels.size)
            for (label in classLabels) {
                dos.writeUTF(label)
            }
            // Save Head Parameters
            dos.writeInt(head.inputDim)
            dos.writeInt(head.numClasses)
            dos.writeInt(head.intermediateChannels)

            // Weights
            dos.writeInt(head.weights.size)
            for (w in head.weights) {
                dos.writeFloat(w)
            }
            // Bias
            dos.writeInt(head.bias.size)
            for (b in head.bias) {
                dos.writeFloat(b)
            }

            // Conv Weights
            dos.writeInt(head.convWeights.size)
            for (w in head.convWeights) {
                dos.writeFloat(w)
            }
            // Conv Bias
            dos.writeInt(head.convBias.size)
            for (b in head.convBias) {
                dos.writeFloat(b)
            }

            // Optimizer Step
            dos.writeInt(head.t)
        }
    }

    fun loadModel(): Pair<TrainableHead, MutableList<String>>? {
        if (!modelFile.exists()) return null

        try {
            DataInputStream(modelFile.inputStream()).use { dis ->
                val version = dis.readByte().toInt()
                if (version != 1) {
                    // Unsupported or old format without version byte, discard.
                    return null
                }

                // Load Class Labels
                val numLabels = dis.readInt()
                val labels = mutableListOf<String>()
                for (i in 0 until numLabels) {
                    labels.add(dis.readUTF())
                }

                // Load Head Parameters
                val inputDim = dis.readInt()
                val numClasses = dis.readInt()
                val intermediateChannels = dis.readInt()

                if (numClasses != numLabels) {
                    // Mismatch, maybe corrupted or logic error
                    return null
                }

                val head = TrainableHead(inputDim, numClasses, intermediateChannels)

                // Weights
                val weightSize = dis.readInt()
                if (weightSize != head.weights.size) return null
                for (i in 0 until weightSize) {
                    head.weights[i] = dis.readFloat()
                }

                // Bias
                val biasSize = dis.readInt()
                if (biasSize != head.bias.size) return null
                for (i in 0 until biasSize) {
                    head.bias[i] = dis.readFloat()
                }

                // Conv Weights
                val convWeightSize = dis.readInt()
                if (convWeightSize != head.convWeights.size) return null
                for (i in 0 until convWeightSize) {
                    head.convWeights[i] = dis.readFloat()
                }

                // Conv Bias
                val convBiasSize = dis.readInt()
                if (convBiasSize != head.convBias.size) return null
                for (i in 0 until convBiasSize) {
                    head.convBias[i] = dis.readFloat()
                }

                // Optimizer Step
                try {
                    head.t = dis.readInt()
                } catch (e: Exception) {
                    // Ignore if missing, assume 0
                }

                return Pair(head, labels)
            }
        } catch (e: Exception) {
            e.printStackTrace()
            return null
        }
    }
}

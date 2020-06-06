import 'regenerator-runtime/runtime'
import * as tf from "@tensorflow/tfjs"
import * as tfvis from "@tensorflow/tfjs-vis"
import { MnistData } from './data.js';

const IMAGE_WIDTH = 28
const IMAGE_HEIGHT = 28
const IMAGE_CHANNELS = 1

async function showExamples(data) {
  const surface = tfvis.visor().surface({ name: "Input data examples", tab: "Input data" })

  const examples = data.nextTestBatch(20)
  const numExamples = examples.xs.shape[0]

  for (let i = 0; i < numExamples; i++) {
    const imageTensor = tf.tidy(() => examples.xs
      .slice([i, 0], [1, examples.xs.shape[1]])
      .reshape([28, 28, 1])
    )

    const canvas = document.createElement("canvas")
    canvas.width = 28
    canvas.height = 28
    canvas.style = "margin: 4px;";

    await tf.browser.toPixels(imageTensor, canvas)
    surface.drawArea.appendChild(canvas)

    imageTensor.dispose()
  }
}

function getModel() {
  const model = tf.sequential()

  // 1st convolution
  model.add(tf.layers.conv2d({
    inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
    kernelSize: 5,
    filters: 8,
    strides: 1,
    activation: "relu",
    kernelInitializer: "varianceScaling"
  }))

  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }))

  // 2nd convolution
  model.add(tf.layers.conv2d({
    kernelSize: 5,
    filters: 16,
    stride: 1,
    activation: "relu",
    kernelInitializer: "varianceScaling"
  }))
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }))


  model.add(tf.layers.flatten())

  const NUM_OUTPUT_CLASSES = 10
  model.add(tf.layers.dense({
    units: NUM_OUTPUT_CLASSES,
    kernelInitializer: "varianceScaling",
    activation: "softmax"
  }))

  const optimiser = tf.train.adam()
  model.compile({
    // bloody Americans
    optimizer: optimiser,
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"]
  })

  return model
}

async function train(model, data) {
  const metrics = ["loss", "val_loss", "acc", "val_acc"]
  const container = { name: "Model Training", styles: { height: "1000px" } }
  const fitCallbacks = tfvis.show.fitCallbacks(container, metrics)

  const TRAIN_DATA_SIZE = 5500

  const [trainingXs, trainingYs] = tf.tidy(() => {
    const d = data.nextTrainBatch(TRAIN_DATA_SIZE)
    return [
      d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]),
      d.labels
    ]
  })

  const TEST_DATA_SIZE = 1000
  const [testXs, testYs] = tf.tidy(() => {
    const d = data.nextTestBatch(TEST_DATA_SIZE)
    return [
      d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]),
      d.labels
    ]
  })

  const BATCH_SIZE = 512
  return model.fit(trainingXs, trainingYs, {
    batchSize: BATCH_SIZE,
    validationData: [testXs, testYs],
    epochs: 10,
    shuffle: true,
    callbacks: fitCallbacks
  })
}

async function run() {
  const data = new MnistData()
  await data.load()

  const model = getModel()

  await train(model, data)
  await showAccuracy(model, data)
  await showConfusion(model, data)
}

const classNames = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]

function predict(model, data, testDataSize = 500) {
  const testData = data.nextTestBatch(testDataSize)
  const testXs = testData.xs.reshape([testDataSize, IMAGE_WIDTH, IMAGE_HEIGHT, 1])
  const labels = testData.labels.argMax(-1)
  const predictions = model.predict(testXs).argMax(-1)

  testXs.dispose()
  return [predictions, labels]
}

async function showAccuracy(model, data) {
  const [predictions, labels] = predict(model, data)
  const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, predictions)
  const container = { name: "Accuracy", tab: "Evaluation" }
  tfvis.show.perClassAccuracy(container, classAccuracy, classNames)

  labels.dispose()
}

async function showConfusion(model, data) {
  const [predictions, labels] = predict(model, data)
  const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, predictions)
  const container = { name: "Confusion Matrix", tab: "Evaluation" }

  tfvis.render.confusionMatrix(container, { values: confusionMatrix }, classNames)
  labels.dispose()
}

document.addEventListener("DOMContentLoaded", run)

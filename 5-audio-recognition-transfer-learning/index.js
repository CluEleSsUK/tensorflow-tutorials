let recogniser


const app = async () => {
  recogniser = speechCommands.create("BROWSER_FFT")
  await recogniser.ensureModelLoaded()
  buildModel()
}

app()

// each frame of audio is 23ms
const NUM_FRAMES = 3
const examples = []

function collect(label) {
  if (recogniser.isListening()) {
    return recogniser.stopListening()
  }

  if (label == null) {
    return
  }

  recogniser.listen(async ({ spectrogram: { frameSize, data } }) => {
    // only use the last part of the audio
    const numOfAudioFramesToCapture = 3
    const vals = data.subarray(-frameSize * numOfAudioFramesToCapture)
      .map(normalise)

    examples.push({ vals, label })
    document.getElementById("console").textContent = `${examples.length} examples collected`
  }, {
    overlapFactor: 0.999,
    includeSpectrogram: true,
    invokeCallbackOnNoiseUnknown: true
  })
}

function normalise(x) {
  const mean = -100
  const std = 10

  return (x - mean) / std
}

const FREQUENCY_SAMPLES = 232
const INPUT_SHAPE = [NUM_FRAMES, FREQUENCY_SAMPLES, 1]

const model = tf.sequential()

async function train() {
  toggleButtons(false)
  const ys = tf.oneHot(examples.map(it => it.label), 3)
  const xsShape = [examples.length, ...INPUT_SHAPE]
  const xs = tf.tensor(flatten(examples.map(it => it.vals)), xsShape)

  await model.fit(xs, ys, {
    batchSize: 16,
    epochs: 10,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        document.querySelector("#console").textContent = `Accuracy: ${(logs.acc * 100).toFixed(1)}% Epoch: ${epoch + 1}`
      }
    }
  })

  tf.dispose([xs, ys])
  toggleButtons(true)
}

function buildModel() {
  model.add(tf.layers.depthwiseConv2d({
    depthMultiplier: 8,
    kernelSize: [NUM_FRAMES, 3],
    activation: "relu",
    inputShape: INPUT_SHAPE
  }))

  model.add(tf.layers.maxPooling2d({ poolSize: [1, 2], strides: [2, 2] }))
  model.add(tf.layers.flatten())
  model.add(tf.layers.dense({ units: 3, activation: "softmax" }))
  const optimizer = tf.train.adam(0.01)

  model.compile({
    optimizer,
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"]
  })
}

function toggleButtons(enable) {
  document.querySelectorAll("button").forEach(b => b.disabled = !enable)
}

function flatten(tensors) {
  const size = tensors[0].length
  const result = new Float32Array(tensors.length * size)

  tensors.forEach((arr, i) => result.set(arr, i * size))
  return result
}

async function moveSlider(labelTensor) {
  const label = (await labelTensor.data())[0]

  document.getElementById("console").textContent = label

  if (label === 2) {
    return
  }

  let delta = 0.1
  const prevValue = +document.getElementById("output").value
  document.getElementById("output").value = prevValue + (label === 0 ? -delta : delta)
}

function listen() {
  if (recogniser.isListening()) {
    recogniser.stopListening()
    toggleButtons(true)
    document.getElementById("listen").textContent = "Listen"
    return
  }

  toggleButtons(false)
  document.getElementById("listen").textContent = "Stop"
  document.getElementById("listen").disabled = false

  recogniser.listen(async ({ spectrogram: { frameSize, data } }) => {
    const vals = data.subarray(-frameSize * NUM_FRAMES)
      .map(normalise)
    const input = tf.tensor(vals, [1, ...INPUT_SHAPE])
    const probs = model.predict(input)
    const predLabel = probs.argMax(1)
    await moveSlider(predLabel)

    tf.dispose([input, probs, predLabel])
  }, {
    overlapFactor: 0.999,
    includeSpectrogram: true,
    invokeCallbackOnNoiseAndUnknown: true
  })
}
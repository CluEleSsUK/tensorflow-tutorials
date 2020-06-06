async function getData() {
  const response = await fetch("https://storage.googleapis.com/tfjs-tutorials/carsData.json");
  const json = await response.json()
  const mapBody = car => ({
    mpg: car.Miles_per_Gallon,
    horsepower: car.Horsepower
  })

  return json.map(mapBody)
    .filter(car => car.mpg != null && car.horsepower != null)
}

function createModel() {
  const model = tf.sequential()
  model.add(tf.layers.dense({ inputShape: [1], units: 1, useBias: true }))
  // model.add(tf.layers.dense({ units: 50, activation: "sigmoid" }))
  model.add(tf.layers.dense({ units: 1, useBias: true }))
  return model
}

function convertToTensor(data) {
  return tf.tidy(() => {
    tf.util.shuffle(data)

    const inputs = data.map(it => it.horsepower)
    const labels = data.map(it => it.mpg)

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 1])
    const labelTensor = tf.tensor2d(labels, [inputs.length, 1])

    const inputMax = inputTensor.max()
    const inputMin = inputTensor.min()
    const labelMax = labelTensor.max()
    const labelMin = labelTensor.min()

    const normalisedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin))
    const normalisedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin))

    return {
      inputs: normalisedInputs,
      labels: normalisedLabels,
      inputMax,
      inputMin,
      labelMax,
      labelMin
    }
  })
}

async function trainModel(model, inputs, labels) {
  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.meanSquaredError,
    metrics: ["mse"]
  })

  return await model.fit(inputs, labels, {
    batchSize: 32,
    epochs: 50,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks(
      { name: "Training Performance" },
      ["loss", "mse"],
      { height: 200, callbacks: ["onEpochEnd"] }
    )
  })

}

async function run() {
  const data = await getData()
  const model = createModel()
  const tensorData = convertToTensor(data)
  const { inputs, labels } = tensorData

  await trainModel(model, inputs, labels)
  console.log("Done training")

  testModel(model, data, tensorData)
}

document.addEventListener("DOMContentLoaded", run)

function testModel(model, inputData, normalisationData) {
  const { inputMin, inputMax, labelMin, labelMax } = normalisationData

  const [xs, preds] = tf.tidy(() => {
    const xs = tf.linspace(0, 1, 100)

    const preds = model.predict(xs.reshape([100, 1]))
    const denormalisedXs = xs.mul(inputMax.sub(inputMin))
      .add(inputMin)

    const denormalisedPreds = preds.mul(labelMax.sub(labelMin))
      .add(labelMin)

    return [denormalisedXs.dataSync(), denormalisedPreds.dataSync()]
  })

  const predictedPoints = Array.from(xs).map((val, i) => {
    return { x: val, y: preds[i] }
  })

  const originalPoints = inputData.map(it => ({
    x: it.horsepower,
    y: it.mpg
  }))

  tfvis.render.scatterplot(
    { name: "Model predictions vs Original Data" },
    { values: [originalPoints, predictedPoints], series: ["original", "predicted"] },
    {
      xLabel: "Horsepower",
      yLabel: "mpg",
      height: 300
    }
  )
}

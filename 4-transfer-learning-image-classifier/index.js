let net
const classifier = knnClassifier.create()

async function app() {
  console.log("Loading mobilenet...")

  net = await mobilenet.load()
  console.log("Successfully loaded mobilenet")

  const webcamElement = document.getElementById("webcam")
  const webcam = await tf.data.webcam(webcamElement)

  const addExample = async classId => {
    const img = await webcam.capture()
    const activation = net.infer(img, true)
    classifier.addExample(activation, classId)
    img.dispose()
  }

  document.getElementById("class-a").addEventListener("click", () => addExample(0))
  document.getElementById("class-b").addEventListener("click", () => addExample(1))
  document.getElementById("class-c").addEventListener("click", () => addExample(2))
  document.getElementById("class-none").addEventListener("click", () => addExample(3))

  while (true) {
    if (classifier.getNumClasses() <= 0) {
      await tf.nextFrame()
      continue
    }

    const img = await webcam.capture()
    const activation = net.infer(img, "conv_preds")
    const result = await classifier.predictClass(activation)

    const classes = ["A", "B", "C", "None"]

    document.getElementById("console").innerText = `
      prediction: ${classes[result.label]}\n
      probability: ${result.confidences[result.label]}
    `
    img.dispose()
    await tf.nextFrame()
  }
}

app()

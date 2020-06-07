require("@tensorflow/tfjs-node")

const http = require("http")
const socketio = require("socket.io")
const pitchType = require("./pitch-type")

const TIMEOUT_BETWEEN_EPOCHS = 500
const PORT = 8001

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms))
}

async function run() {
  const port = process.env.port || PORT
  const server = http.createServer()
  const io = socketio(server)

  server.listen(port, () => console.log(`server running on ${port}`))

  io.on("connection", (socket) => {
    socket.on("predictSample", async sample => {
      io.emit("predictResult", await pitchType.predictSample(sample))
    })
  })

  const numOfTrainingIterations = 10
  for (let i = 0; i < numOfTrainingIterations; i++) {
    console.log(`Training iteration ${i + 1} of ${numOfTrainingIterations}`)
    await pitchType.model.fitDataset(pitchType.trainingData, { epochs: 1 })
    console.log("accuracyPerClass", await pitchType.evaluate(true))
    await sleep(TIMEOUT_BETWEEN_EPOCHS)
  }

  io.emit("trainingComplete", true)
}

run()
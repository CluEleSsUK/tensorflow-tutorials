import io from "socket.io-client"

const predictContainer = document.getElementById("predictContainer")
const predictButton = document.getElementById("predict-button")

const socket = io("localhost:8001", { reconnectionDelay: 300, reconnectionDelayMax: 300 })
const testSample = [4.982, -118.493, -2.973, -7.128, 19.161, -30.677, 80.9, 0]

predictButton.onclick = () => {
  predictButton.disabled = true
  socket.emit("predictSample", testSample)
}

socket.on("connect", () => {
  document.getElementById("waiting-msg").style.display = "none"
  document.getElementById("trainingStatus").innerHTML = "Training in progress"
})

socket.on("trainingComplete", () => {
  document.getElementById("trainingStatus").innerHTML = "Training complete"
  document.getElementById("predictSample").innerHTML = `[${testSample.join(",")}]`
  predictContainer.style.display = "block"
})

socket.on("predictResult", result => {
  plotPredictResult(result)
})

function plotPredictResult(result) {
  predictButton.disabled = false
  document.getElementById("predictResult").innerHTML = result
  console.log(result)
}

socket.on("disconnect", () => {
  document.getElementById("trainingStatus").innerHTML = ""
  predictContainer.style.display = "none"
  document.getElementById("waiting-msg").style.display = "block"
})

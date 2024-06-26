// Copyright 2023 The MediaPipe Authors.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//      http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
"use strict"

import {
  HandLandmarker,
  PoseLandmarker,
  FilesetResolver,
  DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";


const demosSection = document.getElementById("demos");

let handLandmarker = undefined;
let poseLandmarker = undefined;
let runningMode = "IMAGE";
let enableWebcamButton = document.createElement("button");
let webcamRunning = false;

const serviceUuid = "23769b79-5549-4e20-9d0d-37d0a0a8603b";
const characteristicsUUID = {
  msg: "739a7ad4-ab67-4292-818f-cdcc130e65e5"
};

let myCharacteristic;
let myBLE = new p5ble();
// let myBLE;
let msgCharacteristic;
let input, submitButton;

let connectButton = document.getElementById('bleButton')
if (connectButton) {
  console.log("connect button found")
}

// connectButton.onclick = connectAndSend;
connectButton.addEventListener("click", connectAndSend);


function connectAndSend() {
  console.log("connect and send called");
  myBLE.connect(serviceUuid, gotCharacteristics);
}

function gotCharacteristics(error, characteristics) {
  let charSuccess = document.getElementById('bleSuccess');
  if (error) {
    console.log('error: ', error);
    charSuccess.innerHTML = 'Sorry unsuccessful connection, please try again.';
  }
  console.log('characteristics: ', characteristics);
  // Set the first characteristic as myCharacteristic
  myCharacteristic = characteristics[0];
  // sendMessage()
  charSuccess.innerHTML = 'BLE Connected!';
  setTimeout(() => {
    charSuccess.innerHTML = "";
  }, 1000);
}

let prevSent = '';
function sendMessage(resultText, isEndSentence) {  
  // let rawSentence = document.getElementById("predictedText").innerHTML;
  let sentence = '@' + resultText;
  // let sentence = "Hhihi"
  // console.log("sentence", sentence);
  if (resultText != prevSent && sentence && isEndSentence) {
    prevSent = resultText;
    console.log("begin sending", isEndSentence)
    let index = 0;
    let intervalId = setInterval(() => {
      if (index < sentence.length) {
        let letter = sentence.charCodeAt(index);
        console.log(letter);
        myBLE.write(myCharacteristic, letter);
        index++;
      } else {
        clearInterval(intervalId); // Stop the interval when all letters are sent
      }
    }, 1000);
  }
  // console.log("current index: " + index);

  // while (index < sentence.length) {
  //   let letter = sentence.charCodeAt(index);
  //   console.log(letter);
  //   myBLE.write(myCharacteristic, letter);
  //   index++;
  // }
  // return;
}

// sendMessage();
// connectButton.onclick = connectAndSend;

// Before we can use HandLandmarker and PoseLandmarker class we must wait for it to finish
// loading. Machine Learning models can be large and take a moment to
// get everything needed to run.
const createLandmarker = async () => {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
  );
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
      delegate: "GPU"
    },
    runningMode: runningMode,
    numHands: 2
  });
  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`,
      delegate: "GPU"
    },
    runningMode: runningMode,
    numPoses: 2
  });
  demosSection.classList.remove("invisible");
};
createLandmarker();

/********************************************************************
// Continuously grab image from webcam stream and detect it.
********************************************************************/

const video = document.getElementById("webcam");
video.type = HTMLVideoElement;
const canvasElement = document.getElementById(
  "output_canvas"
)
canvasElement.type = HTMLCanvasElement;
const canvasCtx = canvasElement.getContext("2d");
const drawingUtils = new DrawingUtils(canvasCtx);

// Check if webcam access is supported.
const hasGetUserMedia = () => !!navigator.mediaDevices?.getUserMedia;

// If webcam supported, add event listener to button for when user
// wants to activate it.
if (hasGetUserMedia()) {
  enableWebcamButton = document.getElementById("webcamButton");
  enableWebcamButton.addEventListener("click", enableCam);
} else {
  console.warn("getUserMedia() is not supported by your browser");
}

// Enable the live webcam view and start detection.
function enableCam(event) {
  if (!handLandmarker || !poseLandmarker) {
    console.log("Wait! objectDetector not loaded yet.");
    return;
  }

  if (webcamRunning === true) {
    webcamRunning = false;
    enableWebcamButton.innerText = "ENABLE PREDICTIONS";
  } else {
    webcamRunning = true;
    enableWebcamButton.innerText = "DISABLE PREDICTIONS";
    const countdownText = document.getElementById("countdown");
    countdownText.innerText = "Wait about 5 seconds until translation";
  //   setTimeout(() => {
  //     countdownText.innerText = "translation now available";
  // }, 1000); 
  setTimeout(() => {
    countdownText.innerText = "Translation is now available!";

    // Wait for another 3 seconds before making the text blank
    setTimeout(() => {
        countdownText.innerText = "";
    }, 3000); // 3000 milliseconds = 3 seconds delay

}, 1000); // 1000 milliseconds = 1 second delay
  }

  // getUsermedia parameters.
  const constraints = {
    video: true
  };

  // Activate the webcam stream.
  navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
    video.srcObject = stream;
    video.addEventListener("loadeddata", predictWebcam);
  });
}

let lastVideoTime = -1;
let results = undefined;
let poseResults = undefined;
let rawLandmarks = [];


console.log(video);
async function predictWebcam() {
  canvasElement.style.width = video.videoWidth;;
  canvasElement.style.height = video.videoHeight;
  canvasElement.width = video.videoWidth;
  canvasElement.height = video.videoHeight;
  let poseLandmarksRaw = Array.from({ length: 33 }, () => Array.from({ length: 3 }, () => 0)); 
  let handLandmarksRaw = [];
  
  // Now let's start detecting the stream.
  if (runningMode === "IMAGE") {
    runningMode = "VIDEO";
    await handLandmarker.setOptions({ runningMode: "VIDEO" });
    await poseLandmarker.setOptions({ runningMode: "VIDEO" })
  }
  let startTimeMs = performance.now();
  if (lastVideoTime !== video.currentTime) {
    lastVideoTime = video.currentTime;
    results = handLandmarker.detectForVideo(video, startTimeMs);
    poseResults = poseLandmarker.detectForVideo(video, startTimeMs);
  }
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  if (results.landmarks) {
    for (const landmarks of results.landmarks) {
      // myResults.push(landmarks);
      // console.log("hands", landmarks);
      // console.log("keys", Object.keys(landmarks));
      drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {
        color: "#00FF00",
        lineWidth: 4
      });
      drawLandmarks(canvasCtx, landmarks, { color: "#FF0000", lineWidth: 1 });
      for (let j = 0; j < 21; j ++) {
        if (landmarks.hasOwnProperty(j)) {
          handLandmarksRaw.push([landmarks[j].x, landmarks[j].y, landmarks[j].z]);
        }
      }
    }
  } else {
    handLandmarksRaw = Array.from({ length: 21 }, () => Array.from({ length: 3 }, () => 0)); 
  }
  if (poseResults.landmarks) {
    for (const landmark of poseResults.landmarks) {
      // console.log("pose", landmark);
      drawingUtils.drawLandmarks(landmark, {
        radius: (data) => DrawingUtils.lerp(data.from.z, -0.15, 0.1, 5, 1)
      });
      drawingUtils.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS);
      for (let j = 0; j < 33; j ++) {
        if (landmark.hasOwnProperty(j)) {
          poseLandmarksRaw[j]=[landmark[j].x, landmark[j].y, landmark[j].z];
          // console.log("j",  poseLandmarksRaw[j])
        }
      }
    }
  }
  if (poseLandmarksRaw.length != 33) {
    poseLandmarksRaw = Array.from({ length: 33 }, () => Array.from({ length: 3 }, () => 0)); 
  }
  if (handLandmarksRaw.length == 0) {
    handLandmarksRaw = Array.from({ length: 42 }, () => Array.from({ length: 3 }, () => 0)); 
  } else if (handLandmarksRaw.length == 21) {
    let oneHandLandmarksZero = Array.from({ length: 21 }, () => Array.from({ length: 3 }, () => 0));
    handLandmarksRaw = handLandmarksRaw.concat(oneHandLandmarksZero);
  }
  let allLandmarksConcat = poseLandmarksRaw.concat(handLandmarksRaw);
  // console.log(allLandmarksConcat);
  rawLandmarks.push(allLandmarksConcat);
  if (rawLandmarks.length >= 30) {
    let rawLandmarksCopy = JSON.parse(JSON.stringify(rawLandmarks));
    // console.log(rawLandmarksCopy.length);
    rawLandmarks = [];
    // rawLandmarksCopy.slice(-30);
    // console.log("raw", rawLandmarks)
    sendData(rawLandmarksCopy);
  }
  // console.log(myResults)
  canvasCtx.restore();
  // processLandmarks(myResults);
  // Call this function again to keep predicting when the browser is ready.
  if (webcamRunning === true) {
    window.requestAnimationFrame(predictWebcam);
  }
}

// let csrfTokenIndex = document.cookie.indexOf('csrftoken')
// var csrfToken = document.cookie.slice(csrfTokenIndex + 'csrftoken='.length)

function sendData(data) {
  // console.log(data)
  fetch('/translate/', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        // 'X-CSRFToken': csrfToken,
    },
    body: JSON.stringify({ "landmarks": data }),
  })
  .then(response => response.json())
  .then(data => {
    // Handle response from backend
    console.log(data);
    const predictionElement = document.getElementById('predictedText');
    if (data.predicted_sentence != "") {
      predictionElement.innerHTML = `Prediction: ${data.predicted_sentence}`;
    }
    if (data.end_of_sentence && webcamRunning) {
      sendMessage(data.prev_predicted_sentence, data.end_of_sentence);
    }
  })

  .catch(error => {
    console.error('Error:', error);
  });
}

 
  // let sentence = "hello";
  // console.log("length: " + sentence.length);
  // for (let i = 0; i < sentence.length; i++) {
  //   sendMessage(sentence.charCodeAt(i));
  //   console.log(sentence.charCodeAt(i));
  // }

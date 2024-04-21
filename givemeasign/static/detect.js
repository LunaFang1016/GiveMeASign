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
var myResults = [];
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
        lineWidth: 5
      });
      drawLandmarks(canvasCtx, landmarks, { color: "#FF0000", lineWidth: 2 });
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
      myResults.push(landmark);
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

function sendData(data) {
  // Assuming `landmarks` is your array of numbers
  // console.log(data)
  fetch('/translate/', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',

    },
    body: JSON.stringify({ "landmarks": data }),
  })
  .then(response => response.json())
  .then(data => {
    // Handle response from backend
    console.log(data);
  })
  .catch(error => {
    console.error('Error:', error);
  });
}
// let rawLandmarks = [];
// function truncateLandmarks() {
//   let rawResults = [];
//   frameCount += 1;
//   if (frameCount > 30) {
//     rawResults = myResults;
//     myResults = [];
//     frameCount = 0;
//     // console.log(rawResults)
//     rawLandmarks = processLandmarks(rawResults);
//   }
// }


// // Sends a new request to update the to-do list
// function getTranslation() {
//   let xhr = new XMLHttpRequest()
//   xhr.onreadystatechange = function() {
//       if (this.readyState !== 4) return
//       updatePage(xhr)
//   }

//   xhr.open("GET", "/ajax_todolist/get-list", true)
//   xhr.send()
// }


// function updatePage(xhr) {
//   if (xhr.status === 200) {
//       let response = JSON.parse(xhr.responseText)
//       updateList(response)
//       return
//   }

//   if (xhr.status === 0) {
//       displayError("Cannot connect to server")
//       return
//   }


//   if (!xhr.getResponseHeader('content-type') === 'application/json') {
//       displayError(`Received status = ${xhr.status}`)
//       return
//   }

//   let response = JSON.parse(xhr.responseText)
//   if (response.hasOwnProperty('error')) {
//       displayError(response.error)
//       return
//   }

//   displayError(response)
// }

// function displayError(message) {
//   let errorElement = document.getElementById("error")
//   errorElement.innerHTML = message
// }
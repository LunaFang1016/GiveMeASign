const serviceUuid = "1164f202-3472-44a3-9561-ac2d6b05356e";
const characteristicsUUID = {
  msg: "0e515ccd-5290-440c-add7-a93d60f022da"
};

let myBLE;
let msgCharacteristic;
let input, submitButton;

function setup() {
  createCanvas(400, 400);
  myBLE = new p5ble();

  input = createInput();
  input.position(20, 30);

  submitButton = createButton('Send');
  submitButton.position(input.x + input.width + 10, 30);
  submitButton.mousePressed(sendMessage);

  const connectButton = createButton('Connect');
  connectButton.position(20, 60);
  connectButton.mousePressed(connectAndSend);
}

function connectAndSend() {
  myBLE.connect(serviceUuid, gotCharacteristics);
}

function gotCharacteristics(error, characteristics) {
  if (error) console.log('error: ', error);
  for (let i = 0; i < characteristics.length; i++) {
    if (characteristics[i].uuid == characteristicsUUID.msg) {
      msgCharacteristic = characteristics[i];
    }
  }
}

function sendMessage() {
  let userInput = input.value();
  let index = 0; // the index of the character!
  let ASCII = [];
  for (let i = index; i < userInput.length; i++) {
    ASCII.push(userInput.charCodeAt(i)); // pushing each ASCII unicode number to the array!
  }
  console.log(ASCII);
  if (userInput !== "") {
    myBLE.write(msgCharacteristic, ASCII);
    input.value('');
  }
}
const tf = require("@tensorflow/tfjs-node");
const { Storage } = require("@google-cloud/storage");
const path = require("path");

const bucketName = process.env.CLOUD_STORAGE_BUCKET; // Set in .env
const modelPath = "model/model.json";

async function loadModel() {
  const storage = new Storage();
  const localPath = path.join(__dirname, "local_model");

  await storage.bucket(bucketName).file(modelPath).download({ destination: localPath });
  const model = await tf.loadLayersModel(`file://${localPath}`);
  return model;
}

async function predictImage(imageBuffer, model) {
  const tensor = tf.node.decodeImage(imageBuffer, 3)
    .resizeNearestNeighbor([224, 224])
    .toFloat()
    .div(255)
    .expandDims();

  const prediction = model.predict(tensor);
  return prediction.dataSync()[0];
}

module.exports = { loadModel, predictImage };

const tf = require("@tensorflow/tfjs-node");

/**
 * Fungsi untuk melakukan prediksi berdasarkan gambar yang diberikan.
 * @param {Buffer} imageBuffer - Buffer dari file gambar.
 * @param {tf.LayersModel} model - Model TensorFlow yang telah dimuat.
 * @returns {Promise<number>} - Hasil prediksi dalam rentang [0, 1].
 */
async function predictImage(imageBuffer, model) {
  try {
    // Konversi buffer gambar menjadi tensor
    const tensor = tf.node
      .decodeImage(imageBuffer, 3) // Decode RGB
      .resizeNearestNeighbor([224, 224]) // Resize ke [224, 224]
      .toFloat()
      .div(255) // Normalisasi pixel antara [0, 1]
      .expandDims(); // Tambahkan dimensi batch

    // Prediksi menggunakan model
    const prediction = model.predict(tensor);

    // Ambil hasil prediksi (array pertama)
    return prediction.dataSync()[0];
  } catch (err) {
    console.error("Error in predictImage:", err);
    throw new Error("Failed to process image for prediction");
  }
}

module.exports = { predictImage };

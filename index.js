const express = require("express");
const multer = require("multer");
const { loadModel } = require("./src/modelLoader");
const { predictImage } = require("./src/predictHandler");
const { Firestore } = require("@google-cloud/firestore");
const { v4: uuidv4 } = require("uuid");

const app = express();
const firestore = new Firestore();
const upload = multer({ limits: { fileSize: 1000000 } });
const PORT = process.env.PORT || 8080;

// Load the TensorFlow model
let model;
loadModel()
  .then((loadedModel) => {
    model = loadedModel;
    console.log("Model loaded successfully.");
  })
  .catch((err) => console.error("Error loading model:", err));

// Endpoint predict
app.post("/predict", upload.single("image"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ status: "fail", message: "No image provided" });
    }

    const prediction = await predictImage(req.file.buffer, model);
    const result = prediction > 0.5 ? "Cancer" : "Non-cancer";
    const suggestion =
      result === "Cancer"
        ? "Segera periksa ke dokter!"
        : "Penyakit kanker tidak terdeteksi.";

    const response = {
      status: "success",
      message: "Model is predicted successfully",
      data: {
        id: uuidv4(),
        result,
        suggestion,
        createdAt: new Date().toISOString(),
      },
    };

    // Simpan hasil ke Firestore
    await firestore
      .collection("predictions")
      .doc(response.data.id)
      .set(response.data);

    res.json(response);
  } catch (err) {
    console.error(err);
    res.status(400).json({ status: "fail", message: "Terjadi kesalahan dalam melakukan prediksi" });
  }
});

// Endpoint untuk mendapatkan riwayat prediksi
app.get("/predict/histories", async (req, res) => {
    try {
      const predictionsRef = firestore.collection("predictions");
      const snapshot = await predictionsRef.get();
  
      // Jika tidak ada data di koleksi
      if (snapshot.empty) {
        return res.status(200).json({
          status: "success",
          data: [],
        });
      }
  
      // Format data sesuai dengan permintaan
      const histories = snapshot.docs.map((doc) => ({
        id: doc.id,
        history: doc.data(),
      }));
  
      res.status(200).json({
        status: "success",
        data: histories,
      });
    } catch (err) {
      console.error("Error fetching histories:", err);
      res.status(500).json({
        status: "fail",
        message: "Terjadi kesalahan dalam mengambil data riwayat prediksi",
      });
    }
  });
  

// Error handling untuk ukuran file
app.use((err, req, res, next) => {
  if (err.code === "LIMIT_FILE_SIZE") {
    res.status(413).json({
      status: "fail",
      message: "Payload content length greater than maximum allowed: 1000000",
    });
  } else {
    next(err);
  }
});

app.listen(PORT, () => console.log(`Server running on port ${PORT}`));

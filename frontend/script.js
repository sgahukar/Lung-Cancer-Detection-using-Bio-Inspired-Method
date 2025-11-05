document.getElementById("uploadForm").addEventListener("submit", async function (e) {
  e.preventDefault();

  const modelSelect = document.getElementById("modelSelect");
  const imageUpload = document.getElementById("imageUpload");

  if (!modelSelect.value || !imageUpload.files.length) {
    alert("Please select a model and upload an image!");
    return;
  }

  const formData = new FormData();
  formData.append("model", modelSelect.value);
  formData.append("file", imageUpload.files[0]);

  const preview = document.getElementById("preview");
  preview.src = URL.createObjectURL(imageUpload.files[0]);

  try {
    const response = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error("Server error while predicting!");
    }

    const result = await response.json();

    document.getElementById("modelName").textContent = result.model;
    document.getElementById("predictedClass").textContent = result.predicted_class;
    document.getElementById("confidence").textContent = (result.confidence * 100).toFixed(2) + "%";

    document.getElementById("resultBox").classList.remove("hidden");
  } catch (err) {
    alert("Error: " + err.message);
    console.error(err);
  }
});

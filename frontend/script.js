// ------------------------------------------------------------
// RAGify - Frontend
// Plain JavaScript, FastAPI backend
// ------------------------------------------------------------

const uploadForm = document.getElementById("upload-form");
const fileInput = document.getElementById("pdf-file");
const fileNameDisplay = document.getElementById("file-name-display");
const uploadBtn = document.getElementById("upload-btn");
const uploadStatus = document.getElementById("upload-status");
const indexedFiles = document.getElementById("indexed-files");
const dropZone = document.getElementById("drop-zone");

const askForm = document.getElementById("ask-form");
const questionInput = document.getElementById("question-input");
const askBtn = document.getElementById("ask-btn");
const askStatus = document.getElementById("ask-status");

const answerBlock = document.getElementById("answer-block");
const answerText = document.getElementById("answer-text");
const sourcesList = document.getElementById("sources-list");
const sourceCount = document.getElementById("source-count");


// ------------------------------------------------------------
// Helpers
// ------------------------------------------------------------

function setStatus(element, message, kind) {
  element.textContent = message;
  element.className = `status ${kind}`;
}


function clearStatus(element) {
  element.textContent = "";
  element.className = "status";
}


function setButtonLoading(button, loadingText) {
  button.disabled = true;

  const text = button.querySelector(".btn-text");

  if (text) {
    text.textContent = loadingText;
  }
}


function resetButton(button, defaultText) {
  button.disabled = false;

  const text = button.querySelector(".btn-text");

  if (text) {
    text.textContent = defaultText;
  }
}


// ------------------------------------------------------------
// File selection
// ------------------------------------------------------------

fileInput.addEventListener("change", () => {

  const file = fileInput.files[0];

  if (!file) {
    fileNameDisplay.textContent = "Drop your PDF here";
    return;
  }

  fileNameDisplay.textContent = file.name;

});


// ------------------------------------------------------------
// Drag & Drop
// ------------------------------------------------------------

["dragenter", "dragover"].forEach((eventName) => {

  dropZone.addEventListener(eventName, (event) => {

    event.preventDefault();

    dropZone.classList.add("dragover");

  });

});


["dragleave", "drop"].forEach((eventName) => {

  dropZone.addEventListener(eventName, (event) => {

    event.preventDefault();

    dropZone.classList.remove("dragover");

  });

});


dropZone.addEventListener("drop", (event) => {

  const files = event.dataTransfer.files;

  if (!files.length) {
    return;
  }

  const file = files[0];

  if (file.type !== "application/pdf") {

    setStatus(
      uploadStatus,
      "Please select a PDF file.",
      "error"
    );

    return;
  }

  fileInput.files = files;

  fileNameDisplay.textContent = file.name;

});


// ------------------------------------------------------------
// Upload & Index
// ------------------------------------------------------------

uploadForm.addEventListener("submit", async (event) => {

  event.preventDefault();

  const file = fileInput.files[0];

  if (!file) {

    setStatus(
      uploadStatus,
      "Choose a PDF file first.",
      "error"
    );

    return;
  }


  if (file.type !== "application/pdf") {

    setStatus(
      uploadStatus,
      "Only PDF files are supported.",
      "error"
    );

    return;
  }


  const formData = new FormData();

  formData.append("file", file);


  setButtonLoading(
    uploadBtn,
    "Indexing..."
  );


  setStatus(
    uploadStatus,
    "Uploading document and building vector index...",
    "info"
  );


  try {

    const response = await fetch(
      "/documents/upload",
      {
        method: "POST",
        body: formData
      }
    );


    const data = await response.json();


    if (!response.ok) {
      throw new Error(
        data.detail || "Upload failed."
      );
    }


    setStatus(
      uploadStatus,
      `✓ Indexed ${data.filename} — ${data.pages_extracted} page(s), ${data.chunks_indexed} chunk(s).`,
      "success"
    );


    const item = document.createElement("li");

    item.textContent =
      `${data.filename} • ${data.pages_extracted} pages • ${data.chunks_indexed} chunks`;


    indexedFiles.prepend(item);


    uploadForm.reset();

    fileNameDisplay.textContent =
      "Drop your PDF here";


  } catch (error) {

    setStatus(
      uploadStatus,
      error.message,
      "error"
    );

  } finally {

    resetButton(
      uploadBtn,
      "Upload & Index"
    );

  }

});


// ------------------------------------------------------------
// Ask Question
// ------------------------------------------------------------

askForm.addEventListener("submit", async (event) => {

  event.preventDefault();


  const question =
    questionInput.value.trim();


  if (!question) {

    setStatus(
      askStatus,
      "Type a question first.",
      "error"
    );

    return;
  }


  setButtonLoading(
    askBtn,
    "Thinking..."
  );


  answerBlock.classList.add("hidden");


  setStatus(
    askStatus,
    "Searching your document and generating an answer...",
    "info"
  );


  try {

    const response = await fetch(
      "/ask",
      {
        method: "POST",

        headers: {
          "Content-Type": "application/json"
        },

        body: JSON.stringify({
          question
        })
      }
    );


    const data = await response.json();


    if (!response.ok) {

      throw new Error(
        data.detail ||
        "Could not get an answer."
      );

    }


    clearStatus(askStatus);


    answerText.textContent =
      data.answer;


    sourcesList.innerHTML = "";


    const sources =
      Array.isArray(data.sources)
        ? data.sources
        : [];


    sourceCount.textContent =
      sources.length;


    if (sources.length === 0) {

      const item =
        document.createElement("li");

      item.textContent =
        "No specific source chunks were used.";

      sourcesList.appendChild(item);

    } else {

      sources.forEach((source) => {

        const item =
          document.createElement("li");


        item.textContent =
          source.page != null
            ? `${source.source}, page ${source.page}`
            : source.source;


        sourcesList.appendChild(item);

      });

    }


    answerBlock.classList.remove(
      "hidden"
    );


    // Smoothly scroll answer into view
    setTimeout(() => {

      answerBlock.scrollIntoView({
        behavior: "smooth",
        block: "nearest"
      });

    }, 100);


  } catch (error) {

    setStatus(
      askStatus,
      error.message,
      "error"
    );

  } finally {

    resetButton(
      askBtn,
      "Ask AI"
    );

  }

});


// ------------------------------------------------------------
// Keyboard shortcut
// ------------------------------------------------------------

questionInput.addEventListener(
  "keydown",
  (event) => {

    if (
      event.key === "Enter" &&
      (event.ctrlKey || event.metaKey)
    ) {

      event.preventDefault();

      askForm.requestSubmit();

    }

  }
);
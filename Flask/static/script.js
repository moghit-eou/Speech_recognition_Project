/*let mediaRecorder;
let audioChunks = [];
let audioBlob;

document.getElementById('recordBtn').onclick = async function() {
  // Ask for permission and get audio stream
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  mediaRecorder = new MediaRecorder(stream);

  audioChunks = [];

  mediaRecorder.ondataavailable = event => {
    if (event.data.size > 0) {
      audioChunks.push(event.data);
    }
  };

  mediaRecorder.onstop = () => {
    audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
    // Optionally, playback:
    const audioURL = URL.createObjectURL(audioBlob);
    const audioPlayback = document.getElementById('audioPlayback');
    audioPlayback.src = audioURL;
    audioPlayback.style.display = 'block';
    document.getElementById('submitBtn').disabled = false;
  };

  mediaRecorder.start();
  setTimeout(() => mediaRecorder.stop(), 1000); // Stop after 1 second
};

document.getElementById('submitBtn').onclick = function() {
  // Send audioBlob to backend using fetch (POST)
  const formData = new FormData();
  formData.append('audio', audioBlob, 'recording.wav');

  fetch('/upload', { // your Flask route here
    method: 'POST',
    body: formData
  }).then(response => {
    if (response.ok) alert('Audio uploaded!');
    else alert('Upload failed.');
  });
};*/

class AudioRecorder {
  constructor() {
    this.mediaRecorder = null
    this.audioChunks = []
    this.recordedBlob = null
    this.isRecording = false

    this.recordBtn = document.getElementById("recordBtn")
    this.recordBtnText = document.getElementById("recordBtnText")
    this.progressRing = document.getElementById("progressRing")
    this.progressPath = document.getElementById("progressPath")
    this.statusMessage = document.getElementById("statusMessage")
    this.audioPlayback = document.getElementById("audioPlayback")
    this.submitBtn = document.getElementById("submitBtn")

    // Prediction display elements
    this.predictionContainer = document.getElementById("predictionContainer")
    this.wordDisplay = document.getElementById("wordDisplay")
    this.digitDisplay = document.getElementById("digitDisplay")
    this.fullDisplay = document.getElementById("fullDisplay")

    // Probabilities elements
    this.probabilitiesContainer = document.getElementById("probabilitiesContainer")
    this.probabilitiesList = document.getElementById("probabilitiesList")

    this.initializeEventListeners()
  }

  initializeEventListeners() {
    this.recordBtn.addEventListener("click", () => this.handleRecordClick())
    this.submitBtn.addEventListener("click", () => this.submitAudio())
  }

  async handleRecordClick() {
    if (this.isRecording) {
      this.stopRecording()
    } else {
      // Clear previous results when starting new recording
      this.clearPreviousResults()
      await this.startRecording()
    }
  }

  clearPreviousResults() {
    this.predictionContainer.classList.add("hidden")
    this.probabilitiesContainer.classList.add("hidden")
    this.wordDisplay.textContent = "Word"
    this.digitDisplay.textContent = "0"
    this.fullDisplay.textContent = "Full: Word (0)"
    this.probabilitiesList.innerHTML = ""
  }

  async startRecording() {
    try {
      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 44100,
          channelCount: 1,
          volume: 1.0,
        },
      })

      // Create MediaRecorder instance
      this.mediaRecorder = new MediaRecorder(stream, {
        mimeType: "audio/webm;codecs=opus",
      })

      this.audioChunks = []
      this.isRecording = true

      // Set up event handlers
      this.mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          this.audioChunks.push(event.data)
        }
      }

      this.mediaRecorder.onstop = () => {
        this.processRecording()
      }

      // Start recording
      this.mediaRecorder.start()

      // Update UI
      this.updateUIForRecording()

      // Auto-stop after 1 second
      setTimeout(() => {
        if (this.isRecording) {
          this.stopRecording()
        }
      }, 1000)
    } catch (error) {
      console.error("Error accessing microphone:", error)
      this.statusMessage.textContent = "Error: Could not access microphone"
      this.statusMessage.style.color = "#ff6b6b"
    }
  }

  stopRecording() {
    if (this.mediaRecorder && this.isRecording) {
      this.mediaRecorder.stop()
      this.isRecording = false

      // Stop all tracks to release microphone
      this.mediaRecorder.stream.getTracks().forEach((track) => track.stop())
    }
  }

  updateUIForRecording() {
    this.recordBtn.disabled = true
    this.recordBtn.classList.add("recording")
    this.recordBtnText.textContent = "Recording..."
    this.progressRing.classList.remove("hidden")
    this.statusMessage.textContent = "Recording digit... Speak clearly!"
    this.statusMessage.style.color = "#00ffff"

    // Animate progress ring
    let progress = 0
    const interval = setInterval(() => {
      progress += 100 / 100 // 1 second = 100 steps
      this.progressPath.setAttribute("stroke-dashoffset", 100 - progress)

      if (progress >= 100 || !this.isRecording) {
        clearInterval(interval)
        this.progressRing.classList.add("hidden")
        this.progressPath.setAttribute("stroke-dashoffset", 100) // Reset
      }
    }, 10)
  }

  processRecording() {
    // Create blob from recorded chunks
    this.recordedBlob = new Blob(this.audioChunks, { type: "audio/webm" })

    // Create URL for playback
    const audioUrl = URL.createObjectURL(this.recordedBlob)
    this.audioPlayback.src = audioUrl
    this.audioPlayback.style.display = "block"

    // Update UI
    this.recordBtn.disabled = false
    this.recordBtn.classList.remove("recording")
    this.recordBtnText.textContent = "Record New Digit"
    this.submitBtn.disabled = false
    this.submitBtn.classList.remove("cursor-not-allowed")
    this.submitBtn.classList.add("cursor-pointer")

    this.statusMessage.textContent = "Recording complete! Click 'Predict Digit' to analyze."
    this.statusMessage.style.color = "#00ff88"
  }

  async submitAudio() {
    if (!this.recordedBlob) {
      this.statusMessage.textContent = "No audio to submit. Please record first."
      this.statusMessage.style.color = "#ff6b6b"
      return
    }

    try {
      // Disable submit button during upload
      this.submitBtn.disabled = true
      this.submitBtn.textContent = "Analyzing..."
      this.statusMessage.textContent = "AI is processing your audio..."
      this.statusMessage.style.color = "#00ffff"

      // Create FormData
      const formData = new FormData()
      formData.append("audio", this.recordedBlob, "recording.wav")

      // Send to Flask backend
      const response = await fetch("/upload", {
        method: "POST",
        body: formData,
      })

      const result = await response.json()

      if (response.ok && result.success) {
        // Display the prediction results
        this.displayPrediction(result)

        // Display all probabilities
        if (result.probabilities) {
          this.displayProbabilities(result.probabilities)
          this.probabilitiesContainer.classList.remove("hidden")
        }

        this.statusMessage.textContent = `AI predicted: "${result.display}"`
        this.statusMessage.style.color = "#00ff88"
      } else {
        throw new Error(result.error || "Prediction failed")
      }

      // Re-enable submit button
      this.submitBtn.disabled = false
      this.submitBtn.textContent = "Predict Digit"
    } catch (error) {
      console.error("Upload error:", error)
      this.statusMessage.textContent = `Analysis failed: ${error.message}`
      this.statusMessage.style.color = "#ff6b6b"
      this.submitBtn.disabled = false
      this.submitBtn.textContent = "Predict Digit"
    }
  }

  displayPrediction(result) {
    // Main prediction display
    this.wordDisplay.textContent = result.prediction.charAt(0).toUpperCase() + result.prediction.slice(1)
    this.digitDisplay.textContent = result.digit
    this.fullDisplay.textContent = `Full: ${result.display}`
    this.predictionContainer.classList.remove("hidden")
  }

  displayProbabilities(probabilities) {
    this.probabilitiesList.innerHTML = ""

    probabilities.forEach((item, index) => {
      const probabilityItem = document.createElement("div")
      probabilityItem.className = `probability-item ${index === 0 ? "top-prediction" : ""}`

      const word = item.word.charAt(0).toUpperCase() + item.word.slice(1)
      const digit = this.getDigitFromWord(item.word)
      const percentage = item.probability.toFixed(1)

      probabilityItem.innerHTML = `
        <div class="flex justify-between items-center w-full">
          <span class="text-white font-medium">${word} ${digit}</span>
          <span class="text-cyan-200 text-sm">${percentage}%</span>
        </div>
        <div class="probability-bar" style="width: ${item.probability}%"></div>
      `

      this.probabilitiesList.appendChild(probabilityItem)
    })
  }

  getDigitFromWord(word) {
    const wordToDigit = {
      zero: "0",
      one: "1",
      two: "2",
      three: "3",
      four: "4",
      five: "5",
      six: "6",
      seven: "7",
      eight: "8",
      nine: "9",
    }
    return wordToDigit[word.toLowerCase()] || "?"
  }

  resetForNewRecording() {
    this.recordedBlob = null
    this.audioChunks = []
    this.audioPlayback.style.display = "none"
    this.audioPlayback.src = ""
    this.submitBtn.disabled = true
    this.submitBtn.classList.add("cursor-not-allowed")
    this.submitBtn.classList.remove("cursor-pointer")
    this.recordBtnText.textContent = "Record Digit (1s)"
    this.statusMessage.textContent = "Click Record to start"
    this.statusMessage.style.color = "rgba(255, 255, 255, 0.8)"
  }
}

// Initialize the audio recorder when the page loads
document.addEventListener("DOMContentLoaded", () => {
  new AudioRecorder()
})



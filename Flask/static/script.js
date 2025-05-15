let mediaRecorder;
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
};

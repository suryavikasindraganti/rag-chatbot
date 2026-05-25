const chatBox = document.getElementById("chat-box");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");
const statusBar = document.getElementById("status-bar");

let sessionId = localStorage.getItem("sessionId");
if (!sessionId) {
  sessionId = Math.random().toString(36).substring(7);
  localStorage.setItem("sessionId", sessionId);
}

function appendMessage(role, text) {
  const div = document.createElement("div");
  div.className = `message ${role}`;
  div.textContent = text;
  chatBox.appendChild(div);
  chatBox.scrollTop = chatBox.scrollHeight;
  return div;
}

async function sendMessage() {
  const message = userInput.value.trim();
  if (!message) return;

  userInput.value = "";
  sendBtn.disabled = true;
  statusBar.textContent = "";

  appendMessage("user", message);
  const thinkingEl = appendMessage("thinking", "Thinking...");

  try {
    const response = await fetch("http://127.0.0.1:8000/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sessionId, message })
    });

    const data = await response.json();
    thinkingEl.remove();

    if (data.reply) {
      appendMessage("bot", data.reply);
      if (data.retrievedChunks !== undefined) {
        statusBar.textContent = `Retrieved ${data.retrievedChunks} chunk(s) from knowledge base.`;
      }
    } else {
      appendMessage("bot", data.error || "No response received.");
    }

  } catch (error) {
    thinkingEl.remove();
    appendMessage("bot", "Error connecting to backend. Is the server running?");
  }

  sendBtn.disabled = false;
  userInput.focus();
}

// Send on button click
sendBtn.addEventListener("click", sendMessage);

// Send on Enter key
userInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendMessage();
});
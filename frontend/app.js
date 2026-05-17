const API_BASE = "http://127.0.0.1:8000";
let sessionId = localStorage.getItem("dataset_agent_session_id");

const chat = document.getElementById("chat");
const form = document.getElementById("chatForm");
const questionInput = document.getElementById("question");
const loading = document.getElementById("loading");
const errorBox = document.getElementById("error");
const answerBox = document.getElementById("answer");
const tableContainer = document.getElementById("tableContainer");
const codeBox = document.getElementById("codeBox");
const visualization = document.getElementById("visualization");
const sendButton = form.querySelector("button");

function appendMessage(role, text) {
  const div = document.createElement("div");
  div.className = `message ${role}`;
  div.textContent = `${role === "user" ? "You" : "Agent"}: ${text}`;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

function renderTable(rows) {
  if (!rows || rows.length === 0) {
    tableContainer.textContent = "No table returned.";
    return;
  }

  const columns = Object.keys(rows[0]);
  let html = "<table><thead><tr>";
  html += columns.map(col => `<th>${escapeHtml(col)}</th>`).join("");
  html += "</tr></thead><tbody>";

  rows.forEach(row => {
    html += "<tr>";
    html += columns.map(col => `<td>${escapeHtml(String(row[col]))}</td>`).join("");
    html += "</tr>";
  });

  html += "</tbody></table>";
  tableContainer.innerHTML = html;
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const question = questionInput.value.trim();
  if (!question) return;

  appendMessage("user", question);
  questionInput.value = "";

  loading.hidden = false;
  errorBox.hidden = true;
  errorBox.textContent = "";
  sendButton.disabled = true;

  try {
    const response = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, question })
    });

    const data = await response.json();

    if (data.session_id) {
      sessionId = data.session_id;
      localStorage.setItem("dataset_agent_session_id", sessionId);
    }

    if (data.error) {
      errorBox.hidden = false;
      errorBox.textContent = data.error;
      appendMessage("agent", `Error: ${data.error}`);
      return;
    }

    answerBox.textContent = data.answer || "No answer returned.";
    codeBox.textContent = data.generated_code || "No code returned.";
    renderTable(data.result_table);

    if (data.visualization) {
      visualization.innerHTML = data.visualization;
      const scripts = visualization.querySelectorAll("script");
      scripts.forEach(oldScript => {
        const newScript = document.createElement("script");
        [...oldScript.attributes].forEach(attr => newScript.setAttribute(attr.name, attr.value));
        newScript.text = oldScript.text;
        oldScript.replaceWith(newScript);
      });
    } else {
      visualization.textContent = "No visualization generated for this result.";
    }

    appendMessage("agent", data.answer || "Done.");

  } catch (error) {
    errorBox.hidden = false;
    errorBox.textContent = "Could not connect to backend. Make sure FastAPI is running on port 8000.";
    appendMessage("agent", "Frontend/backend connection failed.");
  } finally {
    loading.hidden = true;
    sendButton.disabled = false;
  }
});

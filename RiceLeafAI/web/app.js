const form = document.querySelector("#upload")
const photo = document.querySelector("#photo")
const button = document.querySelector("#analyze")
const preview = document.querySelector("#preview")
const statusText = document.querySelector("#status")
const message = document.querySelector("#message")
const scores = document.querySelector("#scores")
let previewUrl
let ready = false

async function checkModel() {
  try {
    const response = await fetch("/api/status")
    if (!response.ok) throw new Error()
    ready = (await response.json()).ready
    statusText.textContent = ready ? "Model ready" : "Training required"
    if (!ready) message.textContent = "Train the ANN using the README instructions, then refresh this page."
  } catch {
    statusText.textContent = "Backend unavailable"
  }
  button.disabled = !ready || !photo.files.length
}

photo.addEventListener("change", () => {
  if (previewUrl) URL.revokeObjectURL(previewUrl)
  const file = photo.files[0]
  preview.hidden = !file
  if (file) {
    previewUrl = URL.createObjectURL(file)
    preview.src = previewUrl
  } else {
    preview.removeAttribute("src")
  }
  scores.replaceChildren()
  if (ready) message.textContent = "Your result will appear here."
  button.disabled = !ready || !file
})

form.addEventListener("submit", async event => {
  event.preventDefault()
  const file = photo.files[0]
  if (!ready || !file) return
  scores.replaceChildren()
  if (file.size > 8 * 1024 * 1024) {
    message.textContent = "Choose an image smaller than 8 MB."
    return
  }
  photo.disabled = button.disabled = true
  message.textContent = "Analyzing image…"
  try {
    const response = await fetch("/api/predict", { method: "POST", body: file })
    if (!response.ok) {
      const error = await response.json().catch(() => ({}))
      throw new Error(error.detail || "Analysis failed. Try again.")
    }
    const result = await response.json()
    message.textContent = result.low_score
      ? "Uncertain match. Top candidate: " + result.prediction + ". Try a clearer photo."
      : "Top match: " + result.prediction + "."
    for (const { label, score } of result.scores) {
      const row = document.createElement("div")
      row.className = "score"
      const name = document.createElement("span")
      name.textContent = label
      const percentage = document.createElement("span")
      percentage.textContent = (score * 100).toFixed(1) + "%"
      const bar = document.createElement("progress")
      bar.max = 1
      bar.value = score
      bar.setAttribute("aria-label", label + " model score")
      row.append(name, percentage, bar)
      scores.append(row)
    }
    const note = document.createElement("small")
    note.textContent = "Percentages are relative model scores. The model always scores the four supported classes."
    scores.append(note)
  } catch (error) {
    message.textContent = error.message
  } finally {
    photo.disabled = button.disabled = false
  }
})

checkModel()

import gradio as gr
import subprocess
import threading
import queue
import sys
import time


# ----------------------------
# Subprocess session
# ----------------------------
class CLISession:
    def __init__(self, cmd):
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.q = queue.Queue()
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        for line in self.proc.stdout:
            self.q.put(line)

    def is_alive(self):
        return self.proc.poll() is None

    def send(self, text):
        if self.proc.stdin and self.is_alive():
            self.proc.stdin.write(text + "\n")
            self.proc.stdin.flush()

    def drain(self):
        out = []
        while True:
            try:
                out.append(self.q.get_nowait())
            except queue.Empty:
                break
        return "".join(out)

    def wait_for_banner(self):
        start = time.time()
        collected = []

        while time.time() - start < 20:
            try:
                line = self.q.get(timeout=0.25)
                collected.append(line)
                if "Type your question" in line:
                    break
            except queue.Empty:
                continue

        collected.append(self.drain())
        return "".join(collected)

    def wait_for_answer(self):
        collected = []
        last = None
        start = time.time()

        while True:
            if time.time() - start > 120:
                break
            if last and time.time() - last > 0.8:
                break

            try:
                line = self.q.get(timeout=0.25)
                collected.append(line)
                last = time.time()
            except queue.Empty:
                continue

        collected.append(self.drain())
        return "".join(collected)


# ----------------------------
# Bubble formatter
# ----------------------------
def format_user(msg):
    return f"""
<div class="user-bubble">
{msg}
</div>
"""


def format_assistant(msg):
    return f"""
<div class="assistant-bubble">
{msg}
</div>
"""


def parse_output(raw):
    raw = raw.strip()
    answer_lines = []
    sources = ""

    for line in raw.splitlines():
        if line.startswith("Sources:"):
            sources = line
        else:
            answer_lines.append(line)

    answer = "\n".join(answer_lines).strip()
    if sources:
        answer += f"<div class='sources'>{sources}</div>"

    return answer


# ----------------------------
# Handlers
# ----------------------------
def start_session(mode):
    if mode == "Explain":
        cmd = [sys.executable, "-m", "app.rag.qa"]
    elif mode == "Practice":
        cmd = [sys.executable, "-m", "app.rag.practice_app"]
    else:  # Analytics
        cmd = [sys.executable, "-m", "app.rag.analytics"]  # new tiny file

    s = CLISession(cmd)

    if mode == "Analytics":
        # Analytics prints and exits, so read the whole output right now
        raw = s.wait_for_answer()
        text = raw[raw.find("Progress"):] if "Progress" in raw else raw
        transcript_a = format_assistant(text.strip())
        return s, transcript_a, gr.update(interactive=False), gr.update(interactive=False)

    if mode == "Practice":
        # Practice prints the first question; read it like an answer chunk
        first = s.wait_for_answer().strip()
        transcript = format_assistant(first)
        return s, transcript, gr.update(interactive=True), gr.update(interactive=True)

    banner = s.wait_for_banner().strip()
    transcript = format_assistant(banner)
    return s, transcript, gr.update(interactive=True), gr.update(interactive=True)


def send_message(session, text, transcript):
    if not session or not session.is_alive():
        return session, transcript, gr.update()

    text = (text or "").strip()
    if not text:
        return session, transcript, gr.update()

    transcript = (transcript or "") + format_user(text)

    session.send(text)
    raw = session.wait_for_answer()
    answer = parse_output(raw)

    transcript += format_assistant(answer)

    return session, transcript, gr.update(value="")

def toggle_inputs(mode):
    if mode == "Analytics":
        return gr.update(visible=False), gr.update(visible=False)
    else:
        return gr.update(visible=True), gr.update(visible=True)


# ----------------------------
# UI
# ----------------------------
with gr.Blocks(css="""
#container {max-width: 900px; margin:auto;}

.user-bubble {
    background: #4A5568;
    padding: 10px 14px;
    border-radius: 14px;
    margin: 8px 0;
    margin-left: 20%;
    text-align: left;
}

.assistant-bubble {
    background: #1A202C;
    padding: 10px 14px;
    border-radius: 14px;
    margin: 8px 0;
    margin-right: 20%;
    white-space: pre-wrap;
}

.sources {
    margin-top: 8px;
}
""") as demo:

    with gr.Column(elem_id="container"):
        gr.Markdown("## 🎓 AI Teaching Assistant")

        mode_select = gr.Radio(
            ["Explain", "Practice", "Analytics"],
            value="Explain",
            label="Mode"
        )

        start_btn = gr.Button("Start")
        transcript = gr.Markdown()

        user_input = gr.Textbox(
            placeholder="Ask a question...",
            interactive=False,
            visible=True,
        )

        send_btn = gr.Button("Send", interactive=False, visible=True)

        session_state = gr.State(None)

        # hide/show inputs when mode changes
        mode_select.change(
            toggle_inputs,
            inputs=[mode_select],
            outputs=[user_input, send_btn],
        )

        start_btn.click(
            start_session,
            inputs=[mode_select],
            outputs=[session_state, transcript, user_input, send_btn],
        )

        send_btn.click(
            send_message,
            inputs=[session_state, user_input, transcript],
            outputs=[session_state, transcript, user_input],
        )

        user_input.submit(
            send_message,
            inputs=[session_state, user_input, transcript],
            outputs=[session_state, transcript, user_input],
        )

demo.launch()

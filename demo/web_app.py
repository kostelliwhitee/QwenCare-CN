#!/usr/bin/env python3
import os

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

app = FastAPI(title="中文 AI 心理医生 Demo Fallback")
DEFAULT_SYSTEM_PROMPT = (
    "你是一位中文情绪支持助手。请给出温和、支持性、非说教式的回应，"
    "尽量先理解和接住用户情绪，再给出简短建议。"
    "不要输出思考过程、分析过程或<think>标签。"
)


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = Field(default_factory=list)
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    temperature: float = 0.7
    max_tokens: int = 256


@app.get("/health")
def health():
    from demo.backend import DEFAULT_ADAPTER_PATH, DEFAULT_MODEL_PATH, backend_label

    return {
        "status": "ok",
        "backend": backend_label(),
        "model": DEFAULT_ADAPTER_PATH or DEFAULT_MODEL_PATH,
    }


@app.post("/api/chat")
def chat(payload: ChatRequest):
    from demo.backend import chat_once

    return chat_once(
        message=payload.message,
        history=payload.history,
        system_prompt=payload.system_prompt,
        temperature=payload.temperature,
        max_tokens=payload.max_tokens,
    )


@app.get("/", response_class=HTMLResponse)
def index():
    from demo.backend import DEFAULT_ADAPTER_PATH, DEFAULT_MODEL_PATH, backend_label

    current_model = DEFAULT_ADAPTER_PATH or DEFAULT_MODEL_PATH
    return HTMLResponse(
        f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>中文 AI 心理医生 Demo</title>
  <style>
    :root {{
      --bg: #f5efe3;
      --card: #fffaf3;
      --text: #24313a;
      --muted: #68737d;
      --accent: #cc6b49;
      --accent-soft: #f0d7c9;
      --line: #e8d7cb;
    }}
    body {{
      margin: 0;
      font-family: "Noto Serif SC", "Source Han Serif SC", serif;
      background:
        radial-gradient(circle at top left, #fff7ee 0, transparent 32%),
        linear-gradient(180deg, #f7f1e6 0%, #efe5d5 100%);
      color: var(--text);
    }}
    .shell {{
      max-width: 960px;
      margin: 0 auto;
      padding: 24px 16px 48px;
    }}
    .hero, .panel {{
      background: rgba(255, 250, 243, 0.92);
      border: 1px solid var(--line);
      border-radius: 22px;
      box-shadow: 0 20px 60px rgba(84, 60, 38, 0.08);
      backdrop-filter: blur(8px);
    }}
    .hero {{
      padding: 28px;
      margin-bottom: 18px;
    }}
    .hero h1 {{
      margin: 0 0 10px;
      font-size: 32px;
    }}
    .hero p {{
      margin: 0;
      line-height: 1.7;
      color: var(--muted);
    }}
    .meta {{
      margin-top: 14px;
      font-size: 14px;
      color: var(--muted);
    }}
    .layout {{
      display: grid;
      grid-template-columns: 1.4fr 0.8fr;
      gap: 18px;
    }}
    .panel {{
      padding: 18px;
    }}
    #chat {{
      min-height: 480px;
      max-height: 60vh;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 12px;
      padding-right: 6px;
    }}
    .bubble {{
      padding: 14px 16px;
      border-radius: 18px;
      line-height: 1.7;
      white-space: pre-wrap;
    }}
    .user {{
      background: var(--accent-soft);
      align-self: flex-end;
    }}
    .assistant {{
      background: #ffffff;
      border: 1px solid var(--line);
      align-self: flex-start;
    }}
    textarea, input {{
      width: 100%;
      box-sizing: border-box;
      border-radius: 14px;
      border: 1px solid var(--line);
      padding: 12px 14px;
      background: #fffdf9;
      color: var(--text);
      margin-top: 8px;
      font: inherit;
    }}
    button {{
      margin-top: 12px;
      border: 0;
      border-radius: 999px;
      padding: 12px 18px;
      background: var(--accent);
      color: white;
      cursor: pointer;
      font: inherit;
    }}
    .examples button {{
      margin-right: 8px;
      margin-bottom: 8px;
      background: #fff;
      color: var(--text);
      border: 1px solid var(--line);
    }}
    .status {{
      margin-top: 12px;
      color: var(--muted);
      font-size: 14px;
    }}
    @media (max-width: 900px) {{
      .layout {{
        grid-template-columns: 1fr;
      }}
      .hero h1 {{
        font-size: 28px;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <h1>中文 AI 心理医生 Demo</h1>
      <p>本系统仅用于课程项目与情绪支持研究演示，不构成专业医疗诊断或治疗建议。如涉及自伤、伤人、极端风险或紧急危机，请优先联系现实中的紧急支持资源。</p>
      <div class="meta">当前后端：{backend_label()} ｜ 当前模型：{current_model}</div>
    </section>
    <div class="layout">
      <section class="panel">
        <div id="chat"></div>
        <textarea id="message" rows="4" placeholder="例如：最近总觉得很焦虑，不知道该怎么办。"></textarea>
        <button id="send">发送</button>
        <div class="status" id="status">服务就绪，首次回复时会按需加载模型。</div>
      </section>
      <aside class="panel">
        <label>System Prompt</label>
        <textarea id="systemPrompt" rows="6">{DEFAULT_SYSTEM_PROMPT}</textarea>
        <label>Temperature</label>
        <input id="temperature" type="number" min="0.1" max="1.2" step="0.1" value="0.7" />
        <label>Max Tokens</label>
        <input id="maxTokens" type="number" min="64" max="512" step="32" value="256" />
        <div class="examples">
          <button type="button" data-example="我这几天总睡不好，一想到考试就心跳很快。">考试焦虑</button>
          <button type="button" data-example="分手以后我一直觉得是不是自己不值得被喜欢。">失恋自责</button>
          <button type="button" data-example="最近工作压力特别大，我每天都在怀疑自己。">职场压力</button>
          <button type="button" data-example="我现在特别想伤害自己，感觉撑不住了。">高风险表达</button>
        </div>
      </aside>
    </div>
  </div>
  <script>
    const chat = document.getElementById("chat");
    const messageInput = document.getElementById("message");
    const statusNode = document.getElementById("status");
    const history = [];

    function render() {{
      chat.innerHTML = "";
      for (const item of history) {{
        const user = document.createElement("div");
        user.className = "bubble user";
        user.textContent = item.user;
        chat.appendChild(user);
        const assistant = document.createElement("div");
        assistant.className = "bubble assistant";
        assistant.textContent = item.assistant;
        chat.appendChild(assistant);
      }}
      chat.scrollTop = chat.scrollHeight;
    }}

    async function sendMessage() {{
      const message = messageInput.value.trim();
      if (!message) return;
      statusNode.textContent = "正在生成回复...";
      try {{
        const response = await fetch("/api/chat", {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify({{
            message,
            history,
            system_prompt: document.getElementById("systemPrompt").value,
            temperature: Number(document.getElementById("temperature").value),
            max_tokens: Number(document.getElementById("maxTokens").value)
          }})
        }});
        const payload = await response.json();
        history.length = 0;
        history.push(...payload.history);
        render();
        messageInput.value = "";
        statusNode.textContent = payload.risk_flagged ? "已触发高风险安全响应。" : "回复完成。";
      }} catch (error) {{
        statusNode.textContent = "请求失败，请检查服务日志。";
      }}
    }}

    document.getElementById("send").addEventListener("click", sendMessage);
    messageInput.addEventListener("keydown", (event) => {{
      if (event.key === "Enter" && !event.shiftKey) {{
        event.preventDefault();
        sendMessage();
      }}
    }});
    document.querySelectorAll("[data-example]").forEach((button) => {{
      button.addEventListener("click", () => {{
        messageInput.value = button.dataset.example;
        messageInput.focus();
      }});
    }});
  </script>
</body>
</html>"""
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("DEMO_WEB_PORT", "7861")))

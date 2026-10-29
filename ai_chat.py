import os
import uuid
import torch
import time
import requests
import hashlib
import json
import sqlite3  # 新增：SQLite 数据库
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

# ================= 配置区域 =================
MODEL_PATH = "/mnt/workspace/Test/QwenModel"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"正在加载模型: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, 
    dtype=torch.float16 if DEVICE == "cuda" else torch.float32, 
    device_map="auto", 
    trust_remote_code=True
)
print("基础模型加载完成！")

# ================= 数据库初始化 =================
DB_PATH = "chat.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # 用户表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT,
            nickname TEXT,
            avatar TEXT,
            api_key TEXT
        )
    ''')
    # 会话表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            username TEXT,
            title TEXT,
            model TEXT,
            system_prompt TEXT
        )
    ''')
    # 消息表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT,
            role TEXT,
            content TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ================= 内存数据 (仅用于 Token 临时验证) =================
tokens_db = {}

class AuthReq(BaseModel):
    username: str
    password: str
    nickname: Optional[str] = ""

class ChatReq(BaseModel):
    conversation_id: str
    message: str
    model: Optional[str] = "local"
    max_new_tokens: Optional[int] = 512

def hash_pwd(pwd: str):
    return hashlib.sha256(pwd.encode()).hexdigest()

# ================= 接口实现 =================
@app.post("/api/auth/register")
async def register(req: AuthReq):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (req.username,))
    if cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=400, detail="用户名已存在")
    
    cursor.execute("INSERT INTO users (username, password, nickname, avatar) VALUES (?, ?, ?, ?)",
                   (req.username, hash_pwd(req.password), req.nickname or req.username, "🙂"))
    conn.commit()
    conn.close()
    return {"message": "注册成功"}

@app.post("/api/auth/login")
async def login(req: AuthReq):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT password, nickname, avatar FROM users WHERE username = ?", (req.username,))
    row = cursor.fetchone()
    conn.close()
    
    if not row or row[0] != hash_pwd(req.password):
        raise HTTPException(status_code=401, detail="账号或密码错误")
    
    token = str(uuid.uuid4())
    tokens_db[token] = req.username
    return {"token": token, "user": {"username": req.username, "nickname": row[1], "avatar": row[2]}}

@app.get("/api/auth/me")
async def get_me(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="未登录")
    token = authorization.split(" ")[1]
    if token not in tokens_db: raise HTTPException(status_code=401, detail="登录已过期")
    username = tokens_db[token]
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT nickname, avatar FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()
    return {"user": {"username": username, "nickname": row[0], "avatar": row[1]}}

@app.put("/api/auth/password")
async def change_pwd(req: dict, authorization: str = Header(None)):
    token = authorization.split(" ")[1]
    username = tokens_db.get(token)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT password FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    if not row or row[0] != hash_pwd(req.get("old_password")):
        conn.close()
        raise HTTPException(status_code=400, detail="原密码不正确")
    
    cursor.execute("UPDATE users SET password = ? WHERE username = ?", (hash_pwd(req.get("new_password")), username))
    conn.commit()
    conn.close()
    return {"message": "密码修改成功"}

@app.put("/api/profile")
async def update_profile(req: dict, authorization: str = Header(None)):
    token = authorization.split(" ")[1]
    username = tokens_db.get(token)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET nickname = ?, avatar = ? WHERE username = ?", 
                   (req.get("nickname"), req.get("avatar"), username))
    if "api_key" in req:
        cursor.execute("UPDATE users SET api_key = ? WHERE username = ?", (req["api_key"], username))
    conn.commit()
    
    cursor.execute("SELECT nickname, avatar FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()
    return {"user": {"username": username, "nickname": row[0], "avatar": row[1]}}

@app.get("/api/models")
async def get_models():
    return {"models": [
        {"id": "local", "name": "本地 SFT 模型"},
        {"id": "lora", "name": "本地 LoRA 模型"},
        {"id": "cloud", "name": "云端大模型 (智谱)"}
    ], "default": "local"}

# 会话管理
@app.get("/api/conversations")
async def get_convs(authorization: str = Header(None)):
    token = authorization.split(" ")[1]
    username = tokens_db[token]
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, title, model, system_prompt FROM conversations WHERE username = ?", (username,))
    rows = cursor.fetchall()
    conn.close()
    return {"conversations": [{"id": r[0], "title": r[1], "model": r[2], "system_prompt": r[3]} for r in rows]}

@app.post("/api/conversations")
async def create_conv(req: dict, authorization: str = Header(None)):
    token = authorization.split(" ")[1]
    username = tokens_db[token]
    new_id = str(uuid.uuid4())
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO conversations (id, username, title, model, system_prompt) VALUES (?, ?, ?, ?, ?)",
                   (new_id, username, "新对话", req.get("model", "local"), ""))
    conn.commit()
    conn.close()
    return {"conversation": {"id": new_id, "title": "新对话", "model": req.get("model", "local"), "system_prompt": ""}}

@app.put("/api/conversations/{conv_id}")
async def update_conv(conv_id: str, req: dict, authorization: str = Header(None)):
    token = authorization.split(" ")[1]
    username = tokens_db[token]
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    if "title" in req:
        cursor.execute("UPDATE conversations SET title = ? WHERE id = ? AND username = ?", (req["title"], conv_id, username))
    if "system_prompt" in req:
        cursor.execute("UPDATE conversations SET system_prompt = ? WHERE id = ? AND username = ?", (req["system_prompt"], conv_id, username))
    if "model" in req:
        cursor.execute("UPDATE conversations SET model = ? WHERE id = ? AND username = ?", (req["model"], conv_id, username))
    conn.commit()
    
    cursor.execute("SELECT id, title, model, system_prompt FROM conversations WHERE id = ?", (conv_id,))
    c = cursor.fetchone()
    conn.close()
    return {"conversation": {"id": c[0], "title": c[1], "model": c[2], "system_prompt": c[3]}}

@app.delete("/api/conversations/{conv_id}")
async def delete_conv(conv_id: str, authorization: str = Header(None)):
    token = authorization.split(" ")[1]
    username = tokens_db[token]
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM conversations WHERE id = ? AND username = ?", (conv_id, username))
    cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
    conn.commit()
    conn.close()
    return {"message": "删除成功"}

@app.get("/api/conversations/{conv_id}/messages")
async def get_messages(conv_id: str, authorization: str = Header(None)):
    token = authorization.split(" ")[1]
    username = tokens_db[token]
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id ASC", (conv_id,))
    rows = cursor.fetchall()
    conn.close()
    return {"messages": [{"role": r[0], "content": r[1]} for r in rows]}

# ================= 核心聊天接口 =================
@app.post("/api/chat")
async def chat(req: ChatReq, authorization: str = Header(None)):
    token = authorization.split(" ")[1]
    username = tokens_db[token]
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT model, system_prompt FROM conversations WHERE id = ? AND username = ?", (req.conversation_id, username))
    target_conv = cursor.fetchone()
    if not target_conv:
        conn.close()
        raise HTTPException(status_code=404, detail="会话不存在")

    current_model = req.model
    sys_prompt = target_conv[1] or "你是一个乐于助人的AI助手。"

    # 记录用户消息
    cursor.execute("INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)", (req.conversation_id, "user", req.message))
    conn.commit()
    
    # 获取历史记录
    cursor.execute("SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id DESC LIMIT 10", (req.conversation_id,))
    history = cursor.fetchall()
    history.reverse()
    conn.close()

    history_text = ""
    for m in history:
        role = "User" if m[0] == "user" else "Assistant"
        history_text += f"{role}: {m[1]}\n"

    try:
        if current_model == "local":
            final_prompt = f"<|im_start|>system\n{sys_prompt}<|im_end|>\n{history_text}<|im_start|>assistant\n"
            inputs = tokenizer(final_prompt, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=256, pad_token_id=tokenizer.eos_token_id,
                    temperature=0.7, do_sample=True, repetition_penalty=1.1
                )
            reply = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            
        elif current_model == "lora":
            time.sleep(1)
            reply = None
            try:
                with open("sunwukong_dialogues.json", "r", encoding="utf-8") as f:
                    wukong_data = json.load(f)
                for item in wukong_data:
                    if item.get("instruction") and (item["instruction"] in req.message or req.message in item["instruction"]):
                        reply = f"【俺老孙来也】{item['output']}"
                        break
                if not reply:
                    reply = f"【俺老孙来也】你问俺“{req.message}”？俺老孙可听不懂凡人的弯弯绕绕，吃俺老孙一棒！"
            except FileNotFoundError:
                reply = "【俺老孙来也】哎呀，俺老孙找不到那本《西游记》经书（sunwukong_dialogues.json），请确认文件是否跟后端代码放在了同一目录下！"
            except Exception as e:
                reply = f"【俺老孙来也】俺老孙查阅经书时出了点岔子：{str(e)}"
            
        elif current_model == "cloud":
            # 从数据库获取 API Key
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT api_key FROM users WHERE username = ?", (username,))
            row = cursor.fetchone()
            api_key = row[0] if row else None
            conn.close()

            if not api_key:
                reply = "⚠️ 请先在右上角『个人设置』中填入你的智谱 API Key 才能使用云端模型哦！"
            else:
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
                payload = {
                    "model": "glm-4.7-flash",
                    "messages": [{"role": "system", "content": sys_prompt}] + [{"role": m[0], "content": m[1]} for m in history],
                    "max_tokens": 512
                }
                resp = requests.post(url, headers=headers, json=payload, timeout=30)
                resp.raise_for_status()
                reply = f"【智谱GLM】{resp.json()['choices'][0]['message']['content']}"

        if not reply.strip(): reply = "我没有听清，请再说一遍。"
    except Exception as e:
        reply = f"模型推理失败: {str(e)}"

    # 记录 AI 回复
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)", (req.conversation_id, "assistant", reply))
    conn.commit()
    conn.close()

    return {"reply": reply}

app.mount("/", StaticFiles(directory=".", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
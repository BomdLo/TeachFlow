import streamlit as st
import fitz
import httpx
import json
import re
import asyncio
import sqlite3
import hashlib
from opencc import OpenCC
from docx import Document
import os
import jieba
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import io
import sys


# --- 1. 資料庫與安全性設定 ---
def init_db():
    conn = sqlite3.connect('teachflow.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)')
    c.execute('''CREATE TABLE IF NOT EXISTS history 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, 
                  task_type TEXT, result TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()


def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()


def check_hashes(password, hashed_text):
    return make_hashes(password) == hashed_text


init_db()
cc = OpenCC('s2twp')



# --- 2. 初始化 Session State ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "quiz_results" not in st.session_state:
    st.session_state.quiz_results = None
if "display_task" not in st.session_state:
    st.session_state.display_task = None


# --- 3. AI 核心邏輯 (優化 Prompt 以確保 4 個選項) ---
async def run_ai(content, task_type, model_name):
    api_url = "https://api.groq.com/openai/v1/chat/completions"
    # 在 run_ai 函式中
    api_key = st.secrets["GROQ_API_KEY"]
    groq_model = "llama-3.3-70b-versatile"

    if task_type == "生成考題":
        prompt = (
            "你是一位台灣資深教師。請根據內容出20題『單選題』。\n"
            "要求：1.繁體中文 2.嚴格輸出 JSON 陣列 3.每題必須有 4 個選項(A,B,C,D)。\n"
            "格式：[{\"question\": \"..\", \"options\": [\"..\",\"..\",\"..\",\"..\"], \"answer\": 0}]\n\n"
            f"內容：{content[:4000]}"
        )
    else:
        prompt = f"請用繁體中文針對內容進行{task_type}：\n\n{content[:4000]}"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json; charset=utf-8"  # 明確指定 UTF-8
    }

    payload = {
        "model": groq_model,
        "messages": [
            {"role": "system", "content": "你是一個專業的教學助手，請用繁體中文回答。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            # 使用 json 參數會自動幫我們做 UTF-8 編碼
            response = await client.post(api_url, headers=headers, json=payload)
            result = response.json()

            if "error" in result:
                return f"⚠️ Groq API 報錯：{result['error'].get('message', '未知錯誤')}"

            # 取得結果並確保是繁體中文
            content = result['choices'][0]['message']['content']
            return content

        except Exception as e:
            # 如果是編碼錯誤，這裡會抓到並顯示出來
            return f"❌ 執行錯誤：{str(e)}"


# --- 4. Word 匯出邏輯 ---
def create_docx(quiz_data):
    doc = Document()
    doc.add_heading('TeachFlow 自動生成考卷', 0)
    for i, q in enumerate(quiz_data):
        doc.add_paragraph(f"第 {i + 1} 題：{q['question']}", style='List Number')
        for j, opt in enumerate(q['options']):
            doc.add_paragraph(f"({chr(65 + j)}) {opt}")
    doc.add_page_break()
    doc.add_heading('標準答案', level=1)
    for i, q in enumerate(quiz_data):
        doc.add_paragraph(f"第 {i + 1} 題：({chr(65 + q.get('answer', 0))})")
    bio = BytesIO()
    doc.save(bio)
    return bio.getvalue()


# --- 5. 介面邏輯 ---
st.set_page_config(page_title="TeachFlow AI", layout="wide")


def login_ui():
    st.title("🔐 TeachFlow 登入系統")
    choice = st.sidebar.selectbox("選單", ["登入", "註冊"])
    user = st.text_input("帳號")
    passwd = st.text_input("密碼", type='password')

    if choice == "註冊":
        if st.button("創建帳號"):
            conn = sqlite3.connect('teachflow.db')
            c = conn.cursor()
            try:
                c.execute('INSERT INTO users VALUES (?,?)', (user, make_hashes(passwd)))
                conn.commit()
                st.success("註冊成功，請切換至登入")
            except:
                st.error("帳號已存在")
            conn.close()
    else:
        if st.button("登入"):
            conn = sqlite3.connect('teachflow.db')
            c = conn.cursor()
            c.execute('SELECT password FROM users WHERE username=?', (user,))
            data = c.fetchone()
            if data and check_hashes(passwd, data[0]):
                st.session_state.logged_in = True
                st.session_state.username = user
                st.rerun()
            else:
                st.error("密碼錯誤或帳號不存在")


# --- 關鍵字雲生成邏輯 ---
def generate_wordcloud(text):
    # 1. 斷詞處理
    # 建議加入一些自定義的停止詞 (Stopwords)，過濾掉「的」、「是」、「在」等無意義字
    words = jieba.cut(text)
    clean_text = " ".join([word for word in words if len(word) > 1])

    # 2. 設定字型路徑 (請根據你的作業系統修改路徑)
    # Mac 範例路徑
    font_path = "NotoSansTC-VariableFont_wght.ttf"  # 確保檔案跟 app.py 放一起
    # Windows 範例路徑: "C:/Windows/Fonts/msjh.ttc"

    # 3. 建立文字雲物件
    wc = WordCloud(
        font_path=font_path,
        background_color="white",
        width=800,
        height=400,
        max_words=100,
        colormap="viridis"  # 顏色主題
    )

    # 4. 產生圖片
    wc.generate(clean_text)

    # 5. 將圖片轉為 Streamlit 可讀取的格式
    img_buffer = BytesIO()
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(img_buffer, format='png')
    return img_buffer

def main_app():
    st.title(f"🍎 TeachFlow: {st.session_state.username} 老師的助手")

    with st.sidebar:
        st.header("⚙️ 系統設定")
        model_name = st.selectbox("選擇模型", ["deepseek-r1:7b", "deepseek-r1:1.5b"])
        if st.button("登出"):
            st.session_state.logged_in = False
            st.rerun()

        st.divider()
        st.header("📜 歷史紀錄")
        conn = sqlite3.connect('teachflow.db')
        c = conn.cursor()
        c.execute(
            'SELECT id, timestamp, task_type, result FROM history WHERE username=? ORDER BY timestamp DESC LIMIT 5',
            (st.session_state.username,))
        records = c.fetchall()
        for r in records:
            if st.button(f"📅 {r[1][5:16]} | {r[2]}", key=f"hist_{r[0]}"):
                st.session_state.quiz_results = r[3]
                st.session_state.display_task = r[2]
                st.rerun()

    uploaded_file = st.file_uploader("上傳教材 PDF", type="pdf")

    if uploaded_file:
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        full_text = "".join([page.get_text() for page in doc])
        st.write(f"📄 字數：{len(full_text)}")
        task = st.radio("任務：", ["重點摘要", "生成考題", "教學策略建議"])
        # --- 在 Streamlit 介面中的呼叫方式 ---
        if st.button("📊 生成教材關鍵字雲"):
            with st.spinner("正在分析關鍵字並繪圖中..."):
                # 假設 full_text 是你解析 PDF 得到的全文
                cloud_img = generate_wordcloud(full_text)
                st.image(cloud_img, caption="教材核心關鍵字視覺化")

        if st.button("🚀 執行 AI 分析"):
            with st.spinner("思考中..."):
                raw = asyncio.run(run_ai(full_text, task, model_name))
                # 清洗與轉換
                processed = cc.convert(raw).replace("後-end", "後端")
                processed = re.sub(r'<think>.*?</think>', '', processed, flags=re.DOTALL)
                processed = re.sub(r'```json|```', '', processed)

                # 存入資料庫
                conn = sqlite3.connect('teachflow.db')
                c = conn.cursor()
                c.execute('INSERT INTO history (username, task_type, result) VALUES (?,?,?)',
                          (st.session_state.username, task, processed))
                conn.commit()
                conn.close()

                st.session_state.quiz_results = processed
                st.session_state.display_task = task
                st.rerun()

    if st.session_state.quiz_results:
        res = st.session_state.quiz_results
        if st.session_state.display_task == "生成考題":
            json_match = re.search(r'\[.*\]', res, re.DOTALL)
            if json_match:
                quiz_data = json.loads(json_match.group())
                for i, q in enumerate(quiz_data):
                    with st.container(border=True):
                        st.write(f"**Q{i + 1}: {q['question']}**")
                        ans = st.radio(f"選項", q['options'], key=f"q_{i}_{hash(res)}")
                        st.success(f"正確答案：{q['options'][q.get('answer', 0)]}")
                st.download_button("📥 下載 Word", create_docx(quiz_data), "exam.docx")
            else:
                st.write(res)
        else:
            st.info(res)
    st.divider()
    st.write("### 📢 您的回饋對我們非常重要")
    st.write("為了讓 TeachFlow 更貼近老師的需求，誠摯邀請您填寫 1 分鐘回饋問卷：")
    st.link_button("👉 填寫使用回饋", "https://forms.gle/p9iJdyMYaZBg9NxMA")


if not st.session_state.logged_in:
    login_ui()
else:
    main_app()

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
from streamlit_gsheets import GSheetsConnection
import pandas as pd
from datetime import datetime
# 建立連線
conn = st.connection("gsheets", type=GSheetsConnection)




def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()


def check_hashes(password, hashed_text):
    return make_hashes(password) == hashed_text



cc = OpenCC('s2twp')
# --- 1. 設計禁止清單：視覺注入 (CSS Injection) ---
def inject_custom_design():
    st.markdown("""
    <style>
        /* 1. 噪點與非對稱漸變背景 (拒絕純平與紫色) */
        .stApp {
            background-color: #f8fafc;
            background-image: 
                radial-gradient(at 0% 0%, rgba(226, 232, 240, 0.5) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(203, 213, 225, 0.3) 0px, transparent 50%),
                url("https://www.transparenttextures.com/patterns/p6.png");
            background-attachment: fixed;
        }

        /* 2. 專業文字風格 (拒絕 Emoji 功能圖標) */
        h1, h2, h3 {
            font-family: 'Inter', -apple-system, sans-serif;
            color: #1e293b;
            letter-spacing: -0.02em;
        }
        
        /* 3. 按鈕動畫 (拒絕 ease-in-out，改用 Spring 回彈) */
        .stButton>button {
            border-radius: 8px;
            border: 1px solid #e2e8f0;
            background-color: white;
            color: #475569;
            transition: transform 0.2s cubic-bezier(0.34, 1.56, 0.64, 1);
        }
        .stButton>button:hover {
            border-color: #94a3b8;
            color: #1e293b;
            transform: translateY(-1px);
        }
        .stButton>button:active {
            transform: scale(0.98);
        }

        /* 4. 移除容器陰影，改用簡潔邊框 */
        [data-testid="stVerticalBlock"] > div:has(div.stExpander) {
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.7);
        }
    </style>
    """, unsafe_allow_html=True)


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
async def run_ai(content, task_type):
    api_url = "https://api.groq.com/openai/v1/chat/completions"
    api_key = st.secrets["GROQ_API_KEY"]
    groq_model = "llama-3.3-70b-versatile"

    if task_type == "生成考題":
        # 這裡加入了 'explanation' 欄位，回擊「NotebookLM 也能做」的質疑
        prompt = (
            "你是一位台灣資深教師。請根據內容出20題『單選題』。\n"
            "要求：1.繁體中文 2.嚴格輸出 JSON 陣列 3.每題 4 個選項。\n"
            "4. 必須包含 'explanation' 欄位，詳述答案理由或課本出處。\n"
            "格式：[{\"question\": \"..\", \"options\": [\"..\",\"..\",\"..\",\"..\"], \"answer\": 0, \"explanation\": \"..\"}]\n\n"
            f"內容：{content[:4000]}"
        )
    else:
        prompt = f"請用繁體中文針對內容進行{task_type}：\n\n{content[:4000]}"

    payload = {
        "model": groq_model,
        "messages": [
            {"role": "system", "content": "TECHNICAL_ASSISTANT_V1: 專業教學助理，語氣精簡。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(api_url, headers={"Authorization": f"Bearer {api_key}"}, json=payload)
        return response.json()['choices'][0]['message']['content']

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
    
    st.title("🍎 TeachFlow AI 教學助手")
    st.info("這是專為台灣教師設計的 AI 助手，支援 PDF 轉考題、重點摘要與關鍵字雲分析。")
    
    st.title("🔐 TeachFlow 登入系統")
    
    # 建立連線
    conn_gs = st.connection("gsheets", type=GSheetsConnection)
    
    tab1, tab2 = st.tabs(["帳號登入", "快速註冊"])

    with tab2:
        st.subheader("尚未擁有帳號？")
        st.write("請先填寫註冊表單，完成後即可回來登入。")
        # 這裡請換成你的 Google 表單「長網址」
        st.link_button("👉 前往註冊表單", "https://docs.google.com/forms/d/e/1FAIpQLSdVXraSEhAp_rAuXyx5_PjtJTyBt9iut013SeSF_ndmgW0ALQ/viewform")

    with tab1:
        user_input = st.text_input("帳號", placeholder="請輸入註冊時的帳號")
        pass_input = st.text_input("密碼", type='password', placeholder="請輸入密碼")
        
        if st.button("確認登入"):
            if user_input and pass_input:
                try:
                    # 讀取試算表，ttl=0 代表不使用暫存，即時抓取最新資料
                    df = conn_gs.read(ttl=0)
                    
                    # 清理資料：移除欄位名稱前後可能存在的空格
                    df.columns = [c.strip() for c in df.columns]
                    
                    # 搜尋帳號 (轉換為字串並移除前後空格再比對)
                    # 假設你的欄位名稱叫 '帳號'
                    user_data = df[df['帳號'].astype(str).str.strip() == str(user_input).strip()]
                    
                    if not user_data.empty:
                        # 取得該帳號最後一次填寫的密碼 (iloc[-1])
                        # 假設你的欄位名稱叫 '密碼'
                        correct_password = user_data.iloc[-1]['密碼']
                        
                        if str(correct_password).strip() == str(pass_input).strip():
                            st.session_state.logged_in = True
                            st.session_state.username = user_input
                            st.success("驗證成功，進入系統中...")
                            st.rerun()
                        else:
                            st.error("密碼不正確，請再試一次")
                    else:
                        st.error("找不到此帳號，請確認是否已完成註冊表單")
                        
                except Exception as e:
                    st.error("登入系統暫時無法連線")
                    # 如果一直登入失敗，可以暫時取消下面這行的註解來除錯
                    # st.write("請檢查欄位名稱是否正確：", df.columns.tolist())
            else:
                st.warning("請完整填寫帳號與密碼")

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
    # 注入視覺設計
    inject_custom_design()
    
    # --- 資料庫初始化與自動修復 ---
    conn = sqlite3.connect('teachflow.db')
    c = conn.cursor()
    # 自動建立表格，避免 OperationalError
    c.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            task_type TEXT,
            result TEXT
        )
    ''')
    conn.commit()
    # -----------------------------

    st.title("TEACHFLOW_WORKSPACE_V2")
    st.caption(f"ACTIVE_USER: {st.session_state.username} | STATUS: ONLINE")

    # 2. 側邊欄
    with st.sidebar:
        st.markdown("### SYSTEM_CONTROL")
        model_name = st.selectbox("MODEL_SELECT", ["deepseek-r1:7b", "deepseek-r1:1.5b"])
        if st.button("LOGOUT_SESSION"):
            st.session_state.logged_in = False
            st.rerun()

        st.divider()
        st.markdown("### DATA_HISTORY")
        # 讀取最近 5 筆紀錄
        c.execute('SELECT id, timestamp, task_type, result FROM history WHERE username=? ORDER BY timestamp DESC LIMIT 5', (st.session_state.username,))
        records = c.fetchall()
        for r in records:
            if st.button(f"REC_{r[1][5:16]}", key=f"hist_{r[0]}", use_container_width=True):
                st.session_state.quiz_results = r[3]
                st.session_state.display_task = r[2]
                st.rerun()

    # 3. 主畫面佈局
    col_meta, col_workspace = st.columns([1, 2.5], gap="large")

    with col_meta:
        st.markdown("### 01_SOURCE_UPLOAD")
        uploaded_file = st.file_uploader("UPLOAD_PDF_DOCUMENT", type="pdf", label_visibility="collapsed")
        
        if uploaded_file:
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            full_text = "".join([page.get_text() for page in doc])
            st.code(f"METRICS: {len(full_text)} CHARS", language="bash")
            
            st.markdown("### 02_TASK_CONFIGURATION")
            task = st.radio("SELECT_OPERATION", ["重點摘要", "生成考題", "教學策略建議"], label_visibility="collapsed")
            
            if st.button("EXECUTE_AI_ANALYSIS", use_container_width=True):
                with st.spinner("AI_THINKING..."):
                    raw = asyncio.run(run_ai(full_text, task)) # 呼叫非同步 AI
                    processed = cc.convert(raw).replace("後-end", "後端")
                    processed = re.sub(r'<think>.*?</think>', '', processed, flags=re.DOTALL)
                    processed = re.sub(r'```json|```', '', processed)
                    
                    # 存入紀錄
                    c.execute('INSERT INTO history (username, task_type, result) VALUES (?,?,?)', (st.session_state.username, task, processed))
                    conn.commit()

                    st.session_state.quiz_results = processed
                    st.session_state.display_task = task
                    st.rerun()

            if st.button("GENERATE_WORD_CLOUD", use_container_width=True):
                with st.spinner("ANALYZING_KEYWORDS..."):
                    cloud_img = generate_wordcloud(full_text)
                    st.session_state.current_cloud = cloud_img

    with col_workspace:
        if "current_cloud" in st.session_state:
            st.image(st.session_state.current_cloud, use_container_width=True)

        if st.session_state.quiz_results:
            st.markdown(f"### 03_OUTPUT: {st.session_state.display_task}")
            res = st.session_state.quiz_results
            
            if st.session_state.display_task == "生成考題":
                json_match = re.search(r'\[.*\]', res, re.DOTALL)
                if json_match:
                    quiz_data = json.loads(json_match.group())
                    for i, q in enumerate(quiz_data):
                        with st.container(border=True):
                            st.markdown(f"**Q{i + 1}: {q['question']}**")
                            st.radio("OPTIONS", q['options'], key=f"q_{i}_{hash(res)}", label_visibility="collapsed")
                            with st.expander("VIEW_ANSWER_AND_LOGIC"):
                                st.markdown(f"**CORRECT_ANSWER:** {q['options'][q.get('answer', 0)]}")
                                if 'explanation' in q:
                                    st.caption(f"EXPLANATION: {q['explanation']}")
                    
                    st.download_button("EXPORT_AS_WORD", create_docx(quiz_data), "exam.docx", use_container_width=True)
                else:
                    st.text_area("RAW_OUTPUT", res, height=400)
            else:
                st.markdown(res)
        else:
            st.info("AWAITING_INPUT: 請在左側上傳檔案並選擇任務以開始分析。")
    
    # 關閉連線
    conn.close()
    st.divider()
    st.write("### 📢 您的回饋對我們非常重要")
    st.write("為了讓 TeachFlow 更貼近老師的需求，誠摯邀請您填寫 1 分鐘回饋問卷：")
    st.link_button("👉 填寫使用回饋", "https://forms.gle/p9iJdyMYaZBg9NxMA")


if not st.session_state.logged_in:
    login_ui()
else:
    main_app()

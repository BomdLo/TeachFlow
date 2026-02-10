#  TeachFlow AI：老師的智慧教學助手

TeachFlow 是一款專為台灣教師設計的 AI 工具，旨在大幅減少行政負擔，提升出題與教材分析效率。

##  核心功能
- **AI 自動出題**：上傳 PDF 教材，秒速生成繁體中文單選題。
- **教材視覺化**：產出「關鍵字雲」，一眼掌握教材核心重點。
- **重點摘要**：針對長篇教材提供精確的重點整理。
- **Word 導出**：生成的題目可直接導出為 docx 格式，方便排版。

##  技術架構
- **Frontend**: Streamlit
- **AI Engine**: Groq API (Llama-3.3-70b / DeepSeek-R1)
- **Data Processing**: PyMuPDF, Jieba, OpenCC
- **Database**: SQLite3



##  安全性與隱私
TeachFlow 尊重教育資料隱私，所有傳輸皆經過加密，且未來計畫支援完全離線的本地推論模式。

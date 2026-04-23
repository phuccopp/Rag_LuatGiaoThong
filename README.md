# RAG Luật Giao Thông (Vietnam Traffic Law Chatbot)

##  Giới thiệu

Dự án xây dựng chatbot hỏi đáp về **luật giao thông Việt Nam** sử dụng kiến trúc **RAG (Retrieval-Augmented Generation)**, kết hợp giữa truy xuất văn bản luật và mô hình ngôn ngữ lớn.

Hệ thống sử dụng:

*  Embedding: **BKAI / BGE Vietnamese Embedding**
*  LLM: **Google Gemini**

---

##  Kiến trúc hệ thống

```text
User Query
    ↓
Embedding (BKAI/BGE)
    ↓
Vector Database (FAISS/Chroma)
    ↓
Retriever (Top-k)
    ↓
LLM (Gemini)
    ↓
Final Answer + Legal Evidence
```

---

##  Công nghệ sử dụng

*  Python
*  HuggingFace Transformers
*  FAISS / ChromaDB
*  LangChain (RAG pipeline)
* 🇻🇳 **BKAI Embedding (Vietnamese optimized)**
*  Google **Gemini API**

---

##  Thành phần chính

###  1. Embedding Model

* Sử dụng **BKAI / BGE embedding** tối ưu cho tiếng Việt
* Chuyển câu hỏi và văn bản luật thành vector
* Giúp tăng độ chính xác khi truy xuất ngữ nghĩa

 Ưu điểm:

* Hiểu tốt tiếng Việt (so với embedding general)
* Giảm lệch semantic khi query tự nhiên

---

###  2. Retriever (Vector Search)

* Dùng FAISS hoặc ChromaDB
* Truy xuất **Top-k đoạn luật liên quan**
* Có thể tune tham số `k` để cân bằng recall/precision

---

###  3. LLM - Gemini

* Sử dụng Gemini để:

  * Tổng hợp thông tin từ context
  * Sinh câu trả lời tự nhiên
  * Giải thích rõ căn cứ pháp lý

 Ưu điểm:

* Hiểu tốt ngữ cảnh dài
* Trả lời mượt, dễ đọc
* Hỗ trợ reasoning tốt hơn nhiều model open-source nhẹ

---

##  Pipeline hoạt động

1. Người dùng nhập câu hỏi
2. Câu hỏi được encode bằng BKAI embedding
3. Truy vấn vector DB → lấy top-k đoạn luật
4. Ghép context vào prompt
5. Gửi vào Gemini để sinh câu trả lời
6. Trả về kết quả + căn cứ pháp lý

---

##  Ví dụ

**Input:**

```
Không đội mũ bảo hiểm bị phạt bao nhiêu?
```

**Output:**

```
Theo Nghị định ..., người điều khiển xe máy không đội mũ bảo hiểm sẽ bị phạt từ ...
```

---

##  Điểm nổi bật của project

* 🇻🇳 Tối ưu cho tiếng Việt (BKAI embedding)
*  Kết hợp LLM mạnh (Gemini) → trả lời tự nhiên
*  Có căn cứ pháp lý (không hallucinate bừa)
*  Pipeline RAG rõ ràng, dễ mở rộng

---

##  Hạn chế

* Phụ thuộc vào chất lượng chunking dữ liệu luật
* Gemini cần API key (có thể tốn chi phí)
* Retrieval chưa tối ưu → có thể miss context

---

##  Hướng phát triển

* Fine-tune embedding riêng cho luật Việt Nam
* Reranking (Cross-encoder) để tăng độ chính xác
* Multi-turn conversation (chat nhớ ngữ cảnh)
* Deploy web app (Streamlit / FastAPI)

---

##  Cấu hình API

Tạo file `.env`:

```
GEMINI_API_KEY=your_api_key_here
```

---

##  Tác giả

* Phuccopp

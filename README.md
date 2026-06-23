---
sdk: docker
---

# Hybrid RAG Luật Giao Thông Việt Nam

Chatbot hỏi đáp về Luật Giao thông đường bộ Việt Nam sử dụng kiến trúc Retrieval-Augmented Generation (RAG), kết hợp giữa truy xuất dữ liệu pháp lý và mô hình ngôn ngữ lớn để cung cấp câu trả lời có căn cứ pháp luật.

---

## Giới thiệu

Dự án được xây dựng nhằm hỗ trợ tra cứu nhanh các lỗi vi phạm giao thông, mức xử phạt và căn cứ pháp lý theo quy định hiện hành.

Hệ thống sử dụng:

- Embedding: BKAI Vietnamese Bi-Encoder
- Vector Database: FAISS
- Hybrid Retrieval: FAISS + BM25
- LLM: Google Gemini 2.5 Flash Lite
- Giao diện: Gradio

---

## Kiến trúc hệ thống

```text
User Query
     │
     ▼
Hybrid Retrieval
(FAISS + BM25)
     │
     ▼
Top-K Legal Context
     │
     ▼
Prompt Engineering
     │
     ▼
Gemini 2.5 Flash Lite
     │
     ▼
Final Answer
```

---

## Công nghệ sử dụng

- Python
- LangChain
- FAISS
- Rank-BM25
- Hugging Face Embeddings
- Google Gemini API
- Gradio

---

## Thành phần chính

### 1. Embedding Model

Mô hình:

```text
bkai-foundation-models/vietnamese-bi-encoder
```

Chức năng:

- Chuyển câu hỏi và văn bản luật thành vector
- Hỗ trợ truy xuất ngữ nghĩa tiếng Việt
- Tăng khả năng tìm kiếm các hành vi được diễn đạt theo ngôn ngữ tự nhiên

Ví dụ:

```text
"vượt đèn đỏ"
```

có thể truy xuất được:

```text
"không chấp hành hiệu lệnh của đèn tín hiệu giao thông"
```

---

### 2. Hybrid Retrieval

Hệ thống kết hợp hai phương pháp tìm kiếm:

#### Dense Search (FAISS)

- Tìm kiếm theo ngữ nghĩa
- Hiểu được các cách diễn đạt khác nhau của cùng một hành vi

Ví dụ:

```text
vượt đèn đỏ
```

và

```text
không chấp hành hiệu lệnh đèn tín hiệu giao thông
```

được xem là tương đồng.

#### Sparse Search (BM25)

- Tìm kiếm theo từ khóa
- Hữu ích khi câu hỏi chứa thuật ngữ pháp lý cụ thể

Ví dụ:

```text
nồng độ cồn
```

---

### 3. Prompt-based Legal Reasoning

Prompt được thiết kế riêng cho bài toán luật giao thông.

Mô hình sẽ:

- Xác định hành vi vi phạm
- Xác định loại phương tiện
- Chuẩn hóa ngôn ngữ đời thường thành thuật ngữ pháp lý
- Tổng hợp mức xử phạt từ ngữ cảnh được truy xuất

Ví dụ:

```text
vượt đèn đỏ
```

↓

```text
không chấp hành hiệu lệnh của đèn tín hiệu giao thông
```

---

### 4. Gemini 2.5 Flash Lite

Mô hình ngôn ngữ được sử dụng để:

- Phân tích ngữ cảnh pháp lý
- Tổng hợp thông tin từ nhiều điều khoản
- Sinh câu trả lời dễ hiểu
- Hạn chế việc tự suy diễn ngoài dữ liệu được cung cấp

---

## Pipeline hoạt động

1. Người dùng nhập câu hỏi
2. Hybrid Retrieval (FAISS + BM25) tìm các đoạn luật liên quan
3. Context được ghép vào prompt
4. Gemini phân tích ngữ cảnh pháp lý
5. Sinh câu trả lời
6. Trả kết quả cho người dùng

---

## Ví dụ

### Input

```text
Không đội mũ bảo hiểm bị phạt bao nhiêu?
```

### Output

```text
Hành vi vi phạm:
Không đội mũ bảo hiểm khi điều khiển xe mô tô, xe gắn máy.

Mức phạt:
...

Căn cứ pháp lý:
...
```

---

## Điểm nổi bật

- Tối ưu cho tiếng Việt
- Kết hợp Hybrid Search giúp tăng độ chính xác truy xuất
- Trả lời dựa trên dữ liệu pháp lý
- Có khả năng chuẩn hóa ngôn ngữ tự nhiên sang thuật ngữ pháp lý
- Dễ dàng mở rộng thêm dữ liệu luật mới

---

## Hạn chế

- Chất lượng phụ thuộc vào dữ liệu đầu vào và cách chunking
- Có thể bỏ sót một số điều khoản nếu retrieval chưa lấy đủ context
- Chưa sử dụng reranking nâng cao
- Chưa hỗ trợ hội thoại nhiều lượt

---

## Tác giả

Phuccopp

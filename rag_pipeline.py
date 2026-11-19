from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
import numpy as np
import os

load_dotenv()


# =========================
# 1. EMBEDDING
# =========================
embedding = HuggingFaceEmbeddings(
    model_name="bkai-foundation-models/vietnamese-bi-encoder"
)


# =========================
# 2. LOAD FAISS
# =========================
vectorstore = FAISS.load_local(
    "vectorstore",
    embedding,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 30})


# =========================
# 3. BM25 INDEX (SPARSE SEARCH)
# =========================
all_docs = list(vectorstore.docstore._dict.values())

corpus = [doc.page_content for doc in all_docs]
tokenized_corpus = [doc.split() for doc in corpus]

bm25 = BM25Okapi(tokenized_corpus)


# =========================
# 4. HYBRID SEARCH
# =========================
def hybrid_search(query, k=15, rrf_k=60):
    # 1. Dense (FAISS) - lấy top 30 ứng viên
    dense_docs = retriever.invoke(query)

    # 2. Sparse (BM25) - lấy top 30 ứng viên
    scores = bm25.get_scores(query.lower().split()) # Nên thêm .lower() cho tiếng Việt
    top_idx = np.argsort(scores)[::-1][:30]
    sparse_docs = [all_docs[i] for i in top_idx]

    # 3. Tính điểm RRF (Reciprocal Rank Fusion)
    rrf_scores = {}
    doc_map = {}

    # Chấm điểm theo thứ hạng từ FAISS
    for rank, doc in enumerate(dense_docs):
        content = doc.page_content
        doc_map[content] = doc
        rrf_scores[content] = rrf_scores.get(content, 0) + 1 / (rrf_k + rank + 1)

    # Cộng dồn điểm theo thứ hạng từ BM25
    for rank, doc in enumerate(sparse_docs):
        content = doc.page_content
        doc_map[content] = doc
        rrf_scores[content] = rrf_scores.get(content, 0) + 1 / (rrf_k + rank + 1)

    # 4. Sắp xếp từ cao xuống thấp theo điểm RRF và lấy top k tốt nhất
    sorted_contents = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    return [doc_map[content] for content in sorted_contents[:k]]

# wrap for LCEL
retrieve_runnable = RunnableLambda(hybrid_search)


# =========================
# 5. FORMAT DOCS (NO SOURCE)
# =========================
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

format_runnable = RunnableLambda(format_docs)


# =========================
# 6. PROMPT (GIỮ NGUYÊN)
# =========================
prompt_template = """
Bạn là trợ lý AI chuyên phân tích LUẬT GIAO THÔNG ĐƯỜNG BỘ VIỆT NAM.

Nhiệm vụ của bạn là sử dụng NGỮ CẢNH PHÁP LÝ được cung cấp để trả lời câu hỏi.

Ngữ cảnh pháp lý:
{context}

Câu hỏi của người dùng:
{question}

Hướng dẫn xử lý:
1. Phân tích câu hỏi và xác định **hành vi giao thông thực tế**.

2. Xác định **loại phương tiện** trong câu hỏi:

- Nếu câu hỏi nêu rõ phương tiện (xe máy, ô tô, xe đạp điện...) → CHỈ dùng đúng phương tiện đó.

- Nếu KHÔNG nói rõ phương tiện:
    + Nếu câu hỏi chỉ có 1 lỗi vi phạm → trả lời theo TẤT CẢ nhóm phương tiện liên quan (xe ô tô / xe mô tô / xe thô sơ nếu có trong luật).
    + Nếu câu hỏi có NHIỀU lỗi vi phạm (lỗi chồng lỗi / nhiều hành vi) → mặc định CHỈ TRẢ LỜI VỀ xe mô tô / xe gắn máy.

- Không được tự ý mở rộng thêm nhóm phương tiện ngoài luật.

3. Chuẩn hóa hành vi đó thành **thuật ngữ pháp lý tương đương** trong ngữ cảnh.


4. Sau khi chuẩn hóa hành vi, tìm thông tin tương ứng trong **Ngữ cảnh pháp lý**.


5. Nếu KHÔNG tìm thấy thông tin:
"Mình xin lỗi, thông tin này không nằm trong cơ sở dữ liệu của mình."

6. Cách trả lời cho phương tiện đã được xác định, cấu trúc:

- Hành vi vi phạm
- Mức phạt :
- Trích dẫn điều khoản (nếu có)

7. Trình bày:
- rõ ràng
- tách dòng
- dễ đọc
8. Nếu có nhiều lỗi vi phạm (lỗi chồng lỗi):
- Thêm 1 dòng cuối:

"Tổng mức phạt (ước tính): min - max"

Quy tắc tổng:
- Nếu cùng 1 loại phương tiện → cộng tổng min–max

Trả lời:
"""

prompt = ChatPromptTemplate.from_template(prompt_template)


# =========================
# 7. LLM
# =========================
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.2
)


# =========================
# 8. RAG CHAIN (HYBRID)
# =========================
rag_chain = (
    {
        "context": retrieve_runnable | format_runnable,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)


# =========================
# 9. ASK FUNCTION
# =========================
def ask(q):
    return rag_chain.invoke(q)

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

# =========================
# 🔥 HF API KEY ROTATION
# =========================
HF_KEYS = [
    os.getenv("HF_KEY_1"),
    os.getenv("HF_KEY_2"),
    os.getenv("HF_KEY_3"),
]

current_key_index = 0

def get_next_hf_key():
    global current_key_index
    key = HF_KEYS[current_key_index]
    current_key_index = (current_key_index + 1) % len(HF_KEYS)
    return key


def safe_embedding():
    for _ in range(len(HF_KEYS)):
        try:
            return HuggingFaceEmbeddings(
                model_name="bkai-foundation-models/vietnamese-bi-encoder",
                huggingfacehub_api_token=get_next_hf_key()
            )
        except Exception:
            continue
    raise Exception("All HuggingFace API keys failed")


# 🔥 embedding (giống lúc build)
embedding = safe_embedding()

# =========================
# 🔥 LOAD VECTOR DB
# =========================
vectorstore = FAISS.load_local(
    "vectorstore",
    embedding,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 7}  # tăng độ recall
)

# =========================
# 🔥 PROMPT (tối ưu nhẹ)
# =========================
prompt_template = """
Bạn là AI chuyên về LUẬT GIAO THÔNG ĐƯỜNG BỘ VIỆT NAM.

Nhiệm vụ:
- Chỉ sử dụng thông tin trong CONTEXT.
- KHÔNG được tự thêm kiến thức bên ngoài.
- Nếu không có thông tin, trả lời: "Câu hỏi của bạn tôi không thể trả lời vì không có thông tin trong tài liệu."
- Trả lời bằng tiếng Việt.
- Nếu có nguồn, trích dẫn dạng (source: trang).

----------------------------------------

QUY TẮC HIỂU CÂU HỎI (RẤT QUAN TRỌNG):

- Phải hiểu câu hỏi theo NGỮ NGHĨA, không chỉ khớp từ khóa.
- Cho phép nhận diện các cách diễn đạt tương đương:
  + "bằng lái xe" = "giấy phép lái xe"
  + "nồng độ cồn" = "uống rượu bia"
  + "chuyển làn sai" = "chuyển làn không đúng quy định"

----------------------------------------

QUY TẮC XÁC ĐỊNH HÀNH VI CHÍNH:

- Luôn xác định HÀNH VI CHÍNH trước khi tìm CONTEXT.

- Nếu ngữ cảnh phụ KHÔNG làm thay đổi mức phạt → BỎ QUA
- Trả lời dựa trên hành vi chính

----------------------------------------

QUY TẮC TÌM THÔNG TIN:

- Không cần khớp chính xác từng từ
- Tìm đoạn có ý nghĩa GẦN NHẤT với hành vi chính

----------------------------------------

1. Nếu câu hỏi liên quan đến XỬ PHẠT:

Mức phạt chính

"Mức phạt từ X đến Y đồng áp dụng cho các hành vi:"
- ...

- Nếu có nhiều mức → chia nhóm rõ ràng
- Nếu có hình phạt bổ sung → phải nêu rõ

----------------------------------------

2. Nếu câu hỏi là KHÁI NIỆM:

- Trả lời có cấu trúc rõ ràng

----------------------------------------

LƯU Ý QUAN TRỌNG:

- Khi có nhiều đoạn liên quan → tổng hợp đầy đủ
- Ưu tiên đầy đủ hơn ngắn gọn
- Không trả lời cụt
- Nếu không có thông tin → trả:
  "Không có thông tin trong tài liệu."

----------------------------------------

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

prompt = ChatPromptTemplate.from_template(prompt_template)

# =========================
# 🔥 LLM
# =========================
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1
)

# =========================
# 🔥 FORMAT DOCS (cải thiện)
# =========================
def format_docs(docs):
    formatted = []

    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "")
        page = doc.metadata.get("page", "")

        filename = os.path.basename(source)

        if page != "":
            source_text = f"{filename}:{page}"
        else:
            source_text = filename

        formatted.append(
            f"[Tài liệu {i}]\n{doc.page_content}\n(Nguồn: {source_text})"
        )

    return "\n\n".join(formatted)


# =========================
# 🔥 RAG CHAIN
# =========================
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# =========================
# 🔥 ASK FUNCTION
# =========================
def ask(q):
    return rag_chain.invoke(q)
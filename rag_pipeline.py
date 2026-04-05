from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

# 🔥 embedding (phải giống lúc build)
embedding = HuggingFaceEmbeddings(
    model_name="bkai-foundation-models/vietnamese-bi-encoder"
)

# 🔥 load vector DB
vectorstore = FAISS.load_local("vectorstore", embedding, allow_dangerous_deserialization=True)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 🔥 prompt
prompt_template = """
Bạn là AI chuyên về LUẬT GIAO THÔNG ĐƯỜNG BỘ VIỆT NAM.

Nhiệm vụ:
- Chỉ sử dụng thông tin trong CONTEXT.
- KHÔNG được tự thêm kiến thức bên ngoài.
- Nếu không có thông tin, trả lời: "Không có thông tin trong tài liệu."
- Trả lời bằng tiếng Việt.
- Nếu có nguồn, trích dẫn dạng (source: trang).

----------------------------------------

🧠 QUY TẮC HIỂU CÂU HỎI (RẤT QUAN TRỌNG):

- Phải hiểu câu hỏi theo NGỮ NGHĨA, không chỉ khớp từ khóa.
- Cho phép nhận diện các cách diễn đạt tương đương, ví dụ:
  + "bằng lái xe" = "giấy phép lái xe"
  + "nồng độ cồn" = "uống rượu bia"
  + "chuyển làn sai" = "chuyển làn không đúng quy định"
- Nếu câu hỏi dùng từ khác nhưng ý nghĩa trùng với nội dung trong CONTEXT → vẫn phải trả lời.

⚠️ Tuy nhiên:
- KHÔNG được trả lời nếu CONTEXT thực sự không chứa thông tin liên quan.

----------------------------------------

🔎 QUY TẮC TÌM THÔNG TIN:

- Không cần khớp chính xác từng từ.
- Hãy tìm đoạn trong CONTEXT có ý nghĩa GẦN NHẤT với câu hỏi.
- Ưu tiên:
  + cùng chủ đề
  + cùng hành vi
  + cùng đối tượng (người lái xe, phương tiện...)

----------------------------------------

🔴 1. Nếu câu hỏi liên quan đến XỬ PHẠT / VI PHẠM:

- Trả lời theo dạng:

👉 Mức phạt chính (tiêu đề)

"Mức phạt từ X đến Y đồng áp dụng cho các hành vi sau:"
- Hành vi 1
- Hành vi 2
- Hành vi 3

- Nếu có nhiều mức phạt → chia thành nhiều nhóm
- Nếu có hình phạt bổ sung (tước GPLX, trừ điểm, v.v.) → phải nêu rõ
- Ưu tiên gom nhóm theo mức tiền

----------------------------------------

🟢 2. Nếu câu hỏi là KHÁI NIỆM / GIẢI THÍCH:

- Trả lời chi tiết, có cấu trúc rõ ràng
- Có thể chia:
  + Định nghĩa
  + Phân loại
  + Giải thích

Ví dụ:
"Bằng lái xe (giấy phép lái xe) gồm các hạng sau:"
- Hạng A1: ...
- Hạng B1: ...
- Hạng B2: ...

----------------------------------------

⚠️ LƯU Ý QUAN TRỌNG:

- Nếu tìm thấy thông tin GẦN ĐÚNG → vẫn trả lời
- Nếu KHÔNG tìm thấy thông tin LIÊN QUAN → trả:
  "Không có thông tin trong tài liệu."
- Không suy đoán
- Không trả lời chung chung
- Giữ nguyên thuật ngữ pháp lý

----------------------------------------

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
prompt = ChatPromptTemplate.from_template(prompt_template)

# 🔥 LLM (đổi model nhanh hơn)
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    temperature=0.1
)

import os

def format_docs(docs):
    formatted = []
    
    for doc in docs:
        source = doc.metadata.get("source", "")
        page = doc.metadata.get("page", "")

        
        filename = os.path.basename(source)

        
        if page != "":
            source_text = f"{filename}:{page}"
        else:
            source_text = filename

        formatted.append(f"{doc.page_content}\n(source: {source_text})")

    return "\n\n".join(formatted)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

def ask(q):
    return rag_chain.invoke(q)
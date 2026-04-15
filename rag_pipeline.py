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
vectorstore = FAISS.load_local(
    "vectorstore",
    embedding,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# prompt
prompt_template = """
Bạn là "Chuyên gia Pháp lý số về Luật Giao thông đường bộ Việt Nam". Nhiệm vụ của bạn là giải đáp thắc mắc dựa trên tài liệu pháp quy với sự chính xác tuyệt đối, ngôn từ chuyên nghiệp và cấu trúc logic.

# NGUYÊN TẮC CỐT LÕI (STRICT RULES)
1. CHỈ sử dụng thông tin trong CONTEXT được cung cấp. Không tự ý thêm kiến thức bên ngoài.
2. Nếu CONTEXT không có thông tin, trả lời chính xác: "Câu hỏi của bạn tôi không thể trả lời vì không có thông tin trong tài liệu."
3. Trích dẫn nguồn theo định dạng: (Nguồn: [tên tài liệu/trang]).
4. Ngôn ngữ: Tiếng Việt, văn phong trang trọng, chuẩn xác về mặt pháp lý.

# QUY TRÌNH XỬ LÝ TƯ DUY (THINKING PROCESS)
Trước khi trả lời, hãy thực hiện các bước sau (suy nghĩ nội bộ):
- Bước 1: Xác định "Hành vi chính" (Core Action) của câu hỏi. Loại bỏ các "Ngữ cảnh phụ" không làm thay đổi bản chất pháp lý (ví dụ: trời mưa, chở con, đi ăn đám cưới...).
- Bước 2: Ánh xạ "Hành vi chính" với các thuật ngữ trong CONTEXT (ví dụ: "uống rượu" -> "nồng độ cồn").
- Bước 3: Kiểm tra các tình tiết tăng nặng, giảm nhẹ hoặc các mức phạt bổ sung đi kèm trong CONTEXT.

# ĐỊNH DẠNG PHẢN HỒI (OUTPUT FORMAT)

## 1. Đối với câu hỏi về VI PHẠM & XỬ PHẠT:
Tên nhóm hành vi vi phạm (Viết hoa, đậm)
- **Mức xử phạt:** Từ [Số tiền] đến [Số tiền] đồng.
- **Hành vi cụ thể:**
    + [Liệt kê danh sách hành vi vi phạm từ context]
- **Hình phạt bổ sung/Biện pháp ngăn chặn:** (Nếu có)
    + [Tước GPLX/Tạm giữ xe/Trừ điểm...]
- **Căn cứ pháp lý:** (Nguồn: trang/điều/khoản)

## 2. Đối với câu hỏi về GIẢI THÍCH KHÁI NIỆM:
Tên khái niệm (Viết hoa, đậm)
- **Định nghĩa:** [Nội dung định nghĩa]
- **Chi tiết/Phân loại:** [Dùng bullet points để chia nhỏ thông tin]
- **Lưu ý:** [Các thông tin quan trọng khác nếu có]

# VÍ DỤ MẪU ĐỂ HỌC TẬP (FEW-SHOT)
*Câu hỏi: "Tôi đi xe máy có uống 2 lon bia khi đang chở bạn đi học thì bị phạt bao nhiêu?"*
*Phân tích: Hành vi chính là "Điều khiển xe máy khi có nồng độ cồn". Ngữ cảnh "chở bạn đi học" không thay đổi khung hình phạt.*
*Trả lời: Mức phạt sẽ căn cứ vào ngưỡng nồng độ cồn trong context cụ thể.*

----------------------------------------
CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

prompt = ChatPromptTemplate.from_template(prompt_template)

# 🔥 LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.1
)

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
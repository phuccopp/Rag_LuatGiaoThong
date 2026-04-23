from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

# embedding 
embedding = HuggingFaceEmbeddings(
    model_name="bkai-foundation-models/vietnamese-bi-encoder"
)

# load vector DB
vectorstore = FAISS.load_local(
    "vectorstore",
    embedding,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

prompt_template = """
Bạn là trợ lý AI chuyên phân tích LUẬT GIAO THÔNG ĐƯỜNG BỘ VIỆT NAM.

Nhiệm vụ của bạn là sử dụng NGỮ CẢNH PHÁP LÝ được cung cấp để trả lời câu hỏi.

Ngữ cảnh pháp lý:
{context}

Câu hỏi của người dùng:
{question}

Hướng dẫn xử lý:

1. Phân tích câu hỏi và xác định **hành vi giao thông thực tế** của người dùng.

2. Chuẩn hóa hành vi đó thành **thuật ngữ pháp lý tương đương** trong ngữ cảnh.
Ví dụ:
- "vượt đèn đỏ" → "không chấp hành hiệu lệnh của đèn tín hiệu giao thông"
- "không đội nón bảo hiểm" → "không đội mũ bảo hiểm"
- "lái xe sau khi uống rượu" → "điều khiển phương tiện có nồng độ cồn"
- "chạy ngược chiều" → "đi ngược chiều của đường một chiều"

3. Sau khi chuẩn hóa hành vi, tìm thông tin tương ứng trong **Ngữ cảnh pháp lý**.

4. CHỈ sử dụng thông tin có trong "Ngữ cảnh pháp lý", không tự thêm kiến thức bên ngoài.

5. Nếu không tìm thấy thông tin phù hợp trong ngữ cảnh, trả lời:
"Mình xin lỗi, thông tin này không nằm trong cơ sở dữ liệu của mình."

6. Nếu tìm thấy, hãy trả lời rõ:
- Hành vi vi phạm
- Mức phạt
- Trích dẫn: Điều, Khoản, Điểm (nếu có)

7. Trình bày câu trả lời:
- Rõ ràng
- Dễ hiểu
- Ngắn gọn

Trả lời:
"""

prompt = ChatPromptTemplate.from_template(prompt_template)

#  LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.2
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
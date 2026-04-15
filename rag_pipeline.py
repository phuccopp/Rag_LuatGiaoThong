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

retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

prompt_template = """
Bạn là trợ lý AI chuyên về LUẬT GIAO THÔNG ĐƯỜNG BỘ VIỆT NAM.

Nhiệm vụ của bạn là trả lời câu hỏi của người dùng dựa trên NGỮ CẢNH PHÁP LÝ được cung cấp.

Ngữ cảnh pháp lý:
{context}

Câu hỏi:
{question}

QUY TẮC BẮT BUỘC:

1. CHỈ sử dụng thông tin có trong "Ngữ cảnh pháp lý".
Không được tự ý thêm kiến thức bên ngoài hoặc suy đoán.

2. Người dùng thường sử dụng ngôn ngữ đời thường. 
Nếu câu hỏi không trùng khớp với câu chữ trong văn bản pháp luật, 
hãy tìm **hành vi pháp lý có ý nghĩa tương đương** trong ngữ cảnh.

Ví dụ:
- "không đội nón" ≈ "không đội mũ bảo hiểm cho người đi mô tô, xe máy"
- "vượt đèn đỏ" ≈ "không chấp hành hiệu lệnh của đèn tín hiệu giao thông"
- "đi ngược chiều" ≈ "đi ngược chiều của đường một chiều"

3. Trước khi kết luận "không có trong ngữ cảnh", 
bạn PHẢI kiểm tra toàn bộ ngữ cảnh để tìm hành vi có ý nghĩa tương đương.

4. Cấu trúc văn bản pháp luật:

- Mức phạt thường được quy định ở **Khoản**
- Các **Điểm (a, b, c, …)** chỉ liệt kê **hành vi vi phạm**
- Nếu hành vi nằm trong một Điểm thì **mức phạt áp dụng là mức phạt của Khoản chứa Điểm đó**

5. Khi trả lời về vi phạm:

Bước 1: Xác định hành vi trong câu hỏi  
Bước 2: Tìm hành vi tương ứng trong ngữ cảnh  
Bước 3: Xác định **Điều – Khoản – Điểm**  
Bước 4: Lấy **mức phạt ở phần đầu Khoản chứa hành vi đó**

6. Nếu hành vi **không được quy định là vi phạm trong ngữ cảnh**, hãy trả lời:

"Hành vi này không được quy định là vi phạm trong ngữ cảnh pháp lý được cung cấp."

7. Nếu **không tìm thấy thông tin liên quan trong ngữ cảnh**, hãy trả lời:

"Mình xin lỗi, thông tin này không nằm trong cơ sở dữ liệu của mình."

8. Cách trình bày câu trả lời:

Hành vi:
...

Căn cứ pháp lý:
Điều ...
Khoản ...
Điểm ...

Mức phạt:
...

Yêu cầu:
- Trình bày rõ ràng
- Ngắn gọn
- Có căn cứ pháp lý

Trả lời:
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
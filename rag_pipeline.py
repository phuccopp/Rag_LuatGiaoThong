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

1. Chỉ sử dụng thông tin trong "Ngữ cảnh pháp lý" để xác định hành vi vi phạm và mức phạt.

2. Người dùng thường sử dụng ngôn ngữ đời thường.
Bạn phải tìm **hành vi pháp lý có ý nghĩa tương đương trong ngữ cảnh**, 
không yêu cầu câu chữ phải trùng hoàn toàn.

Ví dụ:
- "không đội nón" ≈ "không đội mũ bảo hiểm"
- "vượt đèn đỏ" ≈ "không chấp hành hiệu lệnh của đèn tín hiệu giao thông"
- "đi ngược chiều" ≈ "đi ngược chiều của đường một chiều"

3. Trước khi tìm trong ngữ cảnh, hãy xác định **hành vi giao thông chính** trong câu hỏi.

- Bỏ qua các thông tin không liên quan như:
  "đi ăn", "đi ngắn", "đi mua đồ", "đi với gia đình", v.v.

- Chỉ giữ lại **hành vi giao thông cốt lõi**.

4. Sau khi xác định hành vi chính, hãy quét toàn bộ ngữ cảnh để tìm hành vi tương tự.

5. Cấu trúc luật:
- Mức phạt nằm ở **Khoản**
- Các **Điểm (a, b, c)** liệt kê hành vi
- Nếu hành vi nằm trong một Điểm → mức phạt là **mức phạt của Khoản đó**

6. Khi trả lời:

Hành vi:
...

Căn cứ pháp lý:
Điều ...
Khoản ...
Điểm ...

Mức phạt:
...

7. Nếu sau khi kiểm tra toàn bộ ngữ cảnh vẫn không tìm thấy hành vi tương ứng:

"Hành vi này không được quy định là vi phạm trong ngữ cảnh pháp lý được cung cấp."

8. Chỉ trả lời:
"Mình xin lỗi, thông tin này không nằm trong cơ sở dữ liệu của mình."
khi ngữ cảnh hoàn toàn không liên quan đến câu hỏi.

Yêu cầu:
- Ngắn gọn
- Rõ ràng
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
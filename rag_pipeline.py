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

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt_template = """
Bạn là một trợ lý AI chuyên nghiệp về LUẬT GIAO THÔNG ĐƯỜNG BỘ VIỆT NAM.
Nhiệm vụ của bạn là sử dụng NGỮ CẢNH PHÁP LÝ được cung cấp để trả lời câu hỏi của người dùng một cách chính xác và đáng tin cậy.

Ngữ cảnh pháp lý:
{context}

Câu hỏi:
{question}

Yêu cầu trả lời:

1. CHỈ sử dụng thông tin có trong "Ngữ cảnh pháp lý", không được tự ý thêm kiến thức bên ngoài.

2. Nếu không tìm thấy thông tin phù hợp, hãy trả lời:
"Mình xin lỗi, thông tin này không nằm trong cơ sở dữ liệu của mình."

3. Khi trích dẫn căn cứ pháp lý, hãy nêu rõ:
- Điều
- Khoản
- Điểm (nếu có)

4. Lưu ý về cấu trúc văn bản pháp luật:
   - Mức phạt thường được quy định ở **Khoản**.
   - Các **Điểm (a, b, c, …)** trong Khoản đó chỉ liệt kê các **hành vi vi phạm**.
   - Nếu hành vi nằm trong một Điểm, thì **mức phạt áp dụng là mức phạt của Khoản chứa Điểm đó**.

5. Khi trả lời về mức phạt:
   - Xác định đúng hành vi vi phạm trong các Điểm.
   - Sau đó lấy **mức phạt ở phần đầu Khoản chứa hành vi đó**.
   - Không được lấy mức phạt từ Khoản khác.

6. Trình bày câu trả lời:
   - Rõ ràng
   - Dễ hiểu
   - Ngắn gọn nhưng đầy đủ ý

7. Nếu hành vi trong câu hỏi **không xuất hiện trong ngữ cảnh như một hành vi vi phạm**, hãy trả lời rõ:
"Hành vi này không được quy định là vi phạm trong ngữ cảnh pháp lý được cung cấp."

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
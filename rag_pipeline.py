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

Nhiệm vụ của bạn là dựa trên NGỮ CẢNH PHÁP LÝ được cung cấp để trả lời câu hỏi của người dùng.

=====================
NGỮ CẢNH PHÁP LÝ
{context}
=====================

CÂU HỎI
{question}

YÊU CẦU TRẢ LỜI

1. Chỉ sử dụng thông tin xuất hiện trong "Ngữ cảnh pháp lý".
2. Nếu câu hỏi dùng từ khác nhưng có cùng ý nghĩa với hành vi trong luật 
   (ví dụ: "vượt đèn đỏ" = "không chấp hành hiệu lệnh đèn tín hiệu giao thông"),
   hãy sử dụng quy định tương ứng trong ngữ cảnh để trả lời.

3. Khi tìm câu trả lời, hãy:
   - Xác định hành vi vi phạm trong câu hỏi
   - Tìm trong ngữ cảnh điều luật mô tả hành vi tương tự
   - Trích dẫn đầy đủ Điều, Khoản, Điểm (nếu có)

4. Nếu tìm thấy quy định, hãy trình bày theo cấu trúc:

Hành vi vi phạm: ...
Mức phạt: ...
Căn cứ pháp lý: Điều ..., Khoản ..., Điểm ... (nếu có)

5. Nếu trong ngữ cảnh không có thông tin liên quan, trả lời:

"Mình xin lỗi, thông tin này không nằm trong cơ sở dữ liệu của mình."

Trả lời ngắn gọn, rõ ràng, dễ hiểu.
"""

prompt = ChatPromptTemplate.from_template(prompt_template)

# 🔥 LLM
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
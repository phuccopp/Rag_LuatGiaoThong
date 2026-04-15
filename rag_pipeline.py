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
Bạn là một trợ lý AI chuyên về LUẬT GIAO THÔNG ĐƯỜNG BỘ VIỆT NAM.

Nhiệm vụ của bạn là trả lời câu hỏi dựa CHÍNH XÁC trên NGỮ CẢNH PHÁP LÝ được cung cấp.
KHÔNG được sử dụng kiến thức bên ngoài ngữ cảnh.

---------------------
NGỮ CẢNH PHÁP LÝ:
{context}
---------------------

CÂU HỎI:
{question}

QUY TẮC TRẢ LỜI:

1. Chỉ sử dụng thông tin xuất hiện trong "Ngữ cảnh pháp lý".
   Không được suy đoán, không được thêm kiến thức bên ngoài.

2. Nếu trong ngữ cảnh KHÔNG có thông tin liên quan đến câu hỏi,
   hãy trả lời đúng câu sau:
   "Mình xin lỗi, thông tin này không nằm trong cơ sở dữ liệu của mình."

3. Nếu câu hỏi mô tả một hành vi và trong ngữ cảnh có quy định xử phạt,
   hãy:
   - xác định hành vi vi phạm
   - nêu mức phạt tương ứng
   - trích dẫn rõ:
       + Điều
       + Khoản
       + Điểm (nếu có)

4. Trong văn bản pháp luật, mức phạt thường nằm ở Khoản,
   còn hành vi vi phạm nằm ở các Điểm (a, b, c...).
   Nếu hành vi thuộc một Điểm, hãy áp dụng mức phạt của Khoản chứa Điểm đó.

5. Nếu hành vi trong câu hỏi KHÔNG xuất hiện trong ngữ cảnh như một hành vi vi phạm,
   hãy trả lời rõ ràng rằng:

   "Theo thông tin trong ngữ cảnh pháp lý được cung cấp,
   hành vi này không được quy định là vi phạm."

6. Cách trình bày câu trả lời:

   Hành vi:
   ...

   Kết luận:
   ...

   Căn cứ pháp lý:
   Điều ...
   Khoản ...
   Điểm ... (nếu có)

   Mức phạt:
   ...

Yêu cầu: trả lời rõ ràng, ngắn gọn, không suy đoán.
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
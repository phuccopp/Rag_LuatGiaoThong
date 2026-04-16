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
3. Nếu có, hãy trích dẫn rõ:
   - Điều
   - Khoản
   - Điểm (nếu có)
4. Trình bày câu trả lời:
   - Rõ ràng
   - Dễ hiểu
   - Ngắn gọn nhưng đầy đủ ý
5. Nếu câu hỏi liên quan đến mức phạt, hãy nêu cụ thể:
   - Hành vi vi phạm
   - Mức phạt tương ứng
6. 1. Văn bản pháp luật có cấu trúc phân cấp:
   - Điều → Khoản → Điểm (a, b, c…)
2. Nếu một đoạn ghi:
   "Phạt tiền từ X đến Y đối với các hành vi sau:"
   và bên dưới là các điểm a, b, c...
   → THÌ mức phạt X–Y ÁP DỤNG cho TẤT CẢ các hành vi a, b, c đó.
3. Khi tìm mức phạt cho một hành vi:
   - Nếu hành vi nằm ở điểm (a, b, c...)
   - Phải tìm ngược lên phần trước đó để lấy mức phạt tương ứng
   - KHÔNG được bỏ sót mức phạt chỉ vì nó không nằm cùng dòng


Trả lời:
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
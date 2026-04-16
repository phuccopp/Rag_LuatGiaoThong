import gradio as gr
from rag_pipeline import ask

def chat(message, history):
    response = ask(message)
    return response

examples = [
    "Uống rượu bia rồi lái xe phạt bao nhiêu?",
    "Chuyển làn đường không đúng bị gì?",
    "Giấy phép lái xe gồm hạng nào?"
]

demo = gr.ChatInterface(
    fn=chat,
    examples=examples,  
    title="🚦 Chatbot Luật Giao Thông Việt Nam",
    description="Hỏi về luật giao thông Việt Nam. Bấm vào câu hỏi mẫu hoặc nhập câu hỏi của bạn.",
)

demo.launch(server_name="0.0.0.0", server_port=7860)

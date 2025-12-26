import gradio as gr
from rag_engine import RAGManager

# Initialize the RAG system

try:
    rag_system = RAGManager()
    print(" RAG system initialized successfully")
except Exception as e:
    print(f" Error initializing RAGManager: {e}")
    rag_system = None

# File upload handler

def process_uploads(files):
    if not files:
        return " No files uploaded."
    if rag_system is None:
        return " RAG system not initialized."

    try:
        processed_count = 0
        for file in files:
            rag_system.add_document(str(file))
            processed_count += 1

        return f" Successfully processed {processed_count} document(s)."
    except Exception as e:
        return f" Error: {str(e)}"


# Chat handler (Gradio 6.x native messages)

def chat_interface(message, history):
    if not message or message.strip() == "":
        return history

    if rag_system is None:
        history.append({
            "role": "assistant",
            "content": " RAG system not initialized."
        })
        return history

    if rag_system.vector_db is None:
        history.append({
            "role": "assistant",
            "content": " Please upload documents first."
        })
        return history

    try:
        answer = rag_system.query(message)

        history.append({
            "role": "user",
            "content": message
        })
        history.append({
            "role": "assistant",
            "content": answer
        })

        return history

    except Exception as e:
        history.append({
            "role": "assistant",
            "content": f" Error: {str(e)}"
        })
        return history


# -------------------------------------------------
# Build Gradio UI (NO theme here)
# -------------------------------------------------
with gr.Blocks() as demo:
    gr.Markdown("# Modular RAG-Based AI Assistant")
    gr.Markdown("Upload documents and ask questions about them!")

    with gr.Row():

        # Left column
        with gr.Column(scale=1):
            gr.Markdown("###  Document Upload")

            file_input = gr.File(
                label="Upload Documents",
                file_count="multiple",
                file_types=[".pdf", ".docx", ".txt"]
            )

            upload_btn = gr.Button(" Build Knowledge Base", variant="primary")

            status_box = gr.Textbox(
                label="System Status",
                interactive=False
            )

            upload_btn.click(
                fn=process_uploads,
                inputs=file_input,
                outputs=status_box
            )

        # Right column
        with gr.Column(scale=2):
            gr.Markdown("### Chat Interface")

            chatbot = gr.Chatbot(
                label="AI Assistant",
                height=400
            )

            user_input = gr.Textbox(
                label="Your Message",
                placeholder="Ask something about your documents...",
                lines=2
            )

            with gr.Row():
                send_btn = gr.Button(" Send", variant="primary")
                clear_btn = gr.Button(" Clear Chat")

            send_btn.click(
                chat_interface,
                inputs=[user_input, chatbot],
                outputs=chatbot
            ).then(lambda: "", outputs=user_input)

            user_input.submit(
                chat_interface,
                inputs=[user_input, chatbot],
                outputs=chatbot
            ).then(lambda: "", outputs=user_input)

            clear_btn.click(lambda: [], outputs=chatbot)


# Launch

if __name__ == "__main__":
    print(" Launching Gradio UI...")
    demo.launch(
        debug=True,
        share=False,
        theme=gr.themes.Soft()
    )


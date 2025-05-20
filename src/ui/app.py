import gradio as gr
from pathlib import Path
import shutil

from ..retrieval.rag import generate_answer, retrieve
from ..ingestion.ingestion import main as ingest_main
from ..embedding.index import main as index_main

DATA_RAW_DIR = Path(__file__).resolve().parents[2] / 'data' / 'raw'

def get_contexts(query: str, top_k: int) -> str:
    contexts = retrieve(query, top_k)
    return "\n\n".join(f"Category: {d['category']}\nQ: {d['question']}\nA: {d['answer']}" for d in contexts)

def get_answer(query: str, top_k: int) -> str:
    return generate_answer(query, top_k)

def upload_and_index(file) -> str:
    if file is None:
        return "No file selected."
    try:
        DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
        dest = DATA_RAW_DIR / Path(file.name).name
        shutil.copy(file.name, dest)
        ingest_main()
        index_main()
        return f"File '{Path(file.name).name}' uploaded and indexed successfully."
    except Exception as e:
        return f"Error during upload/index: {e}"

def launch_ui():
    with gr.Blocks(title="NUST Bank LLM Assistant") as demo:
        gr.Markdown("# NUST Bank Customer Service Assistant")
        with gr.Tab("Chat"):
            query_input = gr.Textbox(lines=2, label="Your Question", placeholder="Ask about bank services...")
            top_k_slider = gr.Slider(minimum=1, maximum=10, step=1, value=5, label="Top-K Contexts")
            retrieve_btn = gr.Button("Retrieve Contexts")
            answer_btn = gr.Button("Get Answer")
            contexts_output = gr.Textbox(lines=10, label="Retrieved Contexts")
            answer_output = gr.Textbox(lines=5, label="Assistant Response")
            retrieve_btn.click(fn=get_contexts, inputs=[query_input, top_k_slider], outputs=contexts_output)
            answer_btn.click(fn=get_answer, inputs=[query_input, top_k_slider], outputs=answer_output)
        with gr.Tab("Upload Data"):
            upload_file = gr.File(label="Upload JSON or Excel", file_types=['.json', '.xlsx'])
            upload_btn = gr.Button("Upload & Index")
            upload_status = gr.Textbox(label="Upload Status")
            upload_btn.click(fn=upload_and_index, inputs=upload_file, outputs=upload_status)
    demo.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == '__main__':
    launch_ui()

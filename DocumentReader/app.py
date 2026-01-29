# app.py
import gradio as gr
from config import users
from functions import process_pdf, delete_pdfs, update_checkbox_choices, chat_handler, clear_chat, update_file_list
from document_processor import DocumentProcessor

processor = DocumentProcessor()

# Gradio Interface
with gr.Blocks(title="Document Reader") as demo:
    gr.Markdown("# Document Reader with Qdrant")

    file_state = gr.State(value=processor.get_processed_files())
    user_state = gr.State(value="john.doe123@example.com")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Chat with Your Documents")
            chatbot = gr.Chatbot(label="Document Chatbot", type="messages", height=400)
            msg = gr.Textbox(
                label="Your Message",
                placeholder="Ask anything about your uploaded documents..."
            )
            with gr.Row():
                submit_btn = gr.Button("Submit")
                clear_btn = gr.Button("Clear Chat")
            submit_btn.click(
                fn=chat_handler,
                inputs=[msg, chatbot, user_state],
                outputs=[chatbot, msg]
            )
            clear_btn.click(
                fn=clear_chat,
                inputs=None,
                outputs=chatbot
            )

        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("## Select User")
                user_dropdown = gr.Dropdown(
                    label="User",
                    choices=list(users.keys()),
                    value="john.doe123@example.com",
                    interactive=True
                )

            with gr.Group():
                gr.Markdown("## Upload PDF")
                pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
                upload_btn = gr.Button("Process PDF")
                upload_output = gr.Textbox(label="Status")

            with gr.Group():
                gr.Markdown("## Manage Documents")
                delete_checkboxes = gr.CheckboxGroup(
                    label="Select PDFs to Delete",
                    value=[],
                    choices=processor.get_processed_files("john.doe123@example.com"),
                    interactive=True
                )
                delete_btn = gr.Button("Delete Selected PDFs")

    user_dropdown.change(
        fn=update_file_list,
        inputs=[user_dropdown],
        outputs=[delete_checkboxes]
    ).then(
        fn=lambda x: x,
        inputs=[user_dropdown],
        outputs=[user_state]
    )

    upload_btn.click(
        fn=process_pdf,
        inputs=[pdf_input, file_state, user_state],
        outputs=[upload_output, delete_checkboxes, file_state]
    ).then(
        fn=update_checkbox_choices,
        inputs=[file_state],
        outputs=[delete_checkboxes]
    )

    delete_btn.click(
        fn=delete_pdfs,
        inputs=[delete_checkboxes, file_state, user_state],
        outputs=[upload_output, delete_checkboxes, file_state]
    ).then(
        fn=update_checkbox_choices,
        inputs=[file_state],
        outputs=[delete_checkboxes]
    )

if __name__ == "__main__":
    demo.launch()
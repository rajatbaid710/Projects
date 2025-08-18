from dotenv import load_dotenv
import os
import gradio as gr
from langchain_openai import OpenAI
from langchain_experimental.sql.base import SQLDatabaseChain
from langchain.sql_database import SQLDatabase
from langchain.memory import ConversationBufferMemory
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate

# Format: "mysql+mysqlconnector://user:password@host:port/database"
db = SQLDatabase.from_uri("mysql+mysqlconnector://root:admin@localhost:3306/my_schema")

# Load environment variables from .env file
load_dotenv()
# Initialize LLM
open_api_key = os.getenv("OPENAI_API_KEY")
llm = OpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=open_api_key)

# Prompt tells the LLM how to safely generate SQL



# Memory for conversation
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


# SQLDatabaseChain without get_schema
db_chain = SQLDatabaseChain.from_llm(
    llm=llm,
    db=db,
    verbose=True,
    memory=memory
)

# Store chat history for Gradio
chat_history = []

def query_bot(user_input):
    try:
        raw_result = db_chain.run(user_input)
        
        # If result is structured, convert to table string
        if isinstance(raw_result, list) and len(raw_result) > 0 and isinstance(raw_result[0], dict):
            df = pd.DataFrame(raw_result)
            result_text = df.to_string(index=False)
        else:
            result_text = str(raw_result)
        
        # Update Gradio chat history
        chat_history.append((user_input, result_text))
        return chat_history, chat_history  # returns for Chatbot and Textbox if needed
    
    except Exception as e:
        chat_history.append((user_input, f"Error: {e}"))
        return chat_history, chat_history

# Gradio interface
with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    user_input = gr.Textbox(placeholder="Ask me about your database...", lines=2)
    submit_btn = gr.Button("Send")
    
    submit_btn.click(query_bot, inputs=user_input, outputs=[chatbot, chatbot])
    
demo.launch()
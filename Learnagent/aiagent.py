import gradio as gr
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain.tools import tool
import PyPDF2
import pandas as pd
import mysql.connector
from mysql.connector import Error
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
import logging

# Set up logging to debug Gradio output
logging.basicConfig(level=logging.DEBUG)

# Load environment variables from .env file
load_dotenv()

# Initialize LLM
open_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=open_api_key)

# Define tools
@tool
def read_pdf(file_path: str) -> str:
    """Extracts text from a PDF file."""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        text = "".join([page.extract_text() for page in reader.pages])
    return text

@tool
def read_excel(file_path: str) -> str:
    """Reads an Excel file and returns the first few rows as a string."""
    df = pd.read_excel(file_path)
    return df.head().to_string()

@tool
def query_mysql(query: str) -> str:
    """Executes a SQL query on the MySQL database's my_schema schema and returns results as a string."""
    connection = None
    cursor = None
    try:
        # Retrieve MySQL credentials from environment variables
        db_config = {
            "host": os.getenv("MYSQL_HOST", "127.0.0.1"),
            "port": os.getenv("MYSQL_PORT", "3306"),
            "user": os.getenv("MYSQL_USER", "root"),
            "password": os.getenv("MYSQL_PASSWORD", "admin"),
            "database": os.getenv("MYSQL_DATABASE", "my_schema")
        }
        
        # Establish connection
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor(dictionary=True, buffered=True)  # Buffered cursor to avoid unread results
        
        # Execute query
        cursor.execute(query)
        
        # Handle different query types
        if query.strip().upper().startswith("SELECT") or query.strip().upper().startswith("SHOW"):
            results = cursor.fetchall()
            if not results:
                return "No results found."
            if query.strip().upper().startswith("SHOW"):
                # For SHOW queries, extract table names or relevant fields
                result_str = "\n".join([str(row.get("Tables_in_my_schema", row)) for row in results])
            else:
                # For SELECT queries, return all fields
                result_str = "\n".join([str(row) for row in results])
        else:
            # Handle non-SELECT/SHOW queries (e.g., INSERT, UPDATE, CREATE)
            connection.commit()
            result_str = f"Query executed successfully. Affected rows: {cursor.rowcount}"
        
        return result_str
    
    except Error as e:
        return f"Error executing query: {str(e)}"
    
    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()

# Define tools list
tools = [read_pdf, read_excel, query_mysql]

# Define ReAct prompt template
prompt = PromptTemplate.from_template(
    """You are a helpful assistant with access to the following tools:

{tools}

Use these tools to answer questions or perform tasks. For database-related queries (e.g., fetching data, listing tables, creating tables), use the query_mysql tool with appropriate SQL queries targeting the my_schema schema. For PDF or Excel tasks, use read_pdf or read_excel with the file path. Follow this format:

Thought: [Your reasoning about what to do next]
Action: [The tool to use, one of {tool_names}]
Action Input: [Input for the tool]
Observation: [Result from the tool, if applicable]

Repeat the Thought/Action/Action Input/Observation cycle until you can provide a final answer. Then, conclude with:

Final Answer: [Your final response to the question/task]

If no tool is needed or the query is unclear, ask for clarification before proceeding.

Question/Task: {input}
{agent_scratchpad}
"""
)

# Create ReAct agent
converse_agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=converse_agent, tools=tools, verbose=True, handle_parsing_errors=True)

# Chat function
def chat_agent(history, message):
    # Pass input as a dictionary with 'input' key
    response = agent_executor.invoke({"input": message})["output"]
    # logging.debug(f"Agent response: {response}")
    # logging.debug(f"Current history before append: {history}")
    # Append user and assistant messages as individual dictionaries
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})
    # logging.debug(f"Updated history: {history}")
    return history, ""

# Gradio UI
with gr.Blocks() as demo:
    chatbot = gr.Chatbot(type="messages")  # Use messages format
    msg = gr.Textbox(placeholder="Type your question here...")
    clear = gr.Button("Clear")

    state = gr.State([])

    def respond(history, message):
        updated_history, _ = chat_agent(history, message)
        # logging.debug(f"Returning history to Gradio: {updated_history}")
        return updated_history, ""

    msg.submit(respond, [chatbot, msg], [chatbot, msg])
    clear.click(lambda: [], None, chatbot)

demo.launch()
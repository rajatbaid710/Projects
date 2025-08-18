import gradio as gr
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain.tools import tool
import mysql.connector
from mysql.connector import Error
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
import json

# Load environment variables from .env file
load_dotenv()

# Initialize LLM
open_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=open_api_key)

# Reusable function to get MySQL connection
def get_db_connection():
    """Returns a MySQL connection object using environment variables."""
    db_config = {
        "host": os.getenv("MYSQL_HOST", "127.0.0.1"),
        "port": os.getenv("MYSQL_PORT", "3306"),
        "user": os.getenv("MYSQL_USER", "root"),
        "password": os.getenv("MYSQL_PASSWORD", "admin"),
        "database": os.getenv("MYSQL_DATABASE", "my_schema")
    }
    return mysql.connector.connect(**db_config)

# Define tools

@tool
def query_mysql(query: str) -> str:
    """Executes a SQL query on the MySQL database's my_schema schema and returns results as a string."""
    connection = None
    cursor = None
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True, buffered=True)
        
        cursor.execute(query)
        
        if query.strip().upper().startswith("SELECT") or query.strip().upper().startswith("SHOW"):
            results = cursor.fetchall()
            if not results:
                return "No results found."
            if query.strip().upper().startswith("SHOW"):
                result_str = "\n".join([str(row.get("Tables_in_my_schema", row)) for row in results])
            else:
                headers = list(results[0].keys())
                rows = [list(row.values()) for row in results]
                header_row = "| " + " | ".join(headers) + " |"
                separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
                data_rows = ["| " + " | ".join(str(val) if val is not None else "" for val in row) + " |" for row in rows]
                result_str = "```markdown\n" + "\n".join([header_row, separator_row] + data_rows) + "\n```"
        else:
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

@tool
def fetch_schema(tool_input: str = "") -> str:
    """Fetches schema details for all tables in the my_schema database and returns them as a JSON string.
    If my_schema.json exists, loads the schema from the file. Otherwise, fetches from the database and saves to my_schema.json."""
    schema_file = "my_schema.json"
    
    if os.path.exists(schema_file):
        try:
            with open(schema_file, "r") as f:
                schema = json.load(f)
                return json.dumps(schema, indent=2)
        except (IOError, json.JSONDecodeError) as e:
            pass
    
    connection = None
    cursor = None
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True, buffered=True)
        
        query = """
        SELECT JSON_OBJECTAGG(
            t.table_name, JSON_OBJECT(
                'columns', t.table_columns,
                'constraints', t.table_constraints
            )
        ) AS schema_json
        FROM (
            SELECT 
                c.table_name,
                JSON_ARRAYAGG(
                    JSON_OBJECT(
                        'column_name', c.column_name,
                        'data_type', c.column_type,
                        'is_nullable', c.is_nullable,
                        'column_default', c.column_default,
                        'extra', c.extra,
                        'comment', c.column_comment
                    ) 
                ) AS table_columns,
                (
                    SELECT JSON_ARRAYAGG(
                        JSON_OBJECT(
                            'constraint_name', k.constraint_name,
                            'column_name', k.column_name,
                            'referenced_table', k.referenced_table_name,
                            'referenced_column', k.referenced_column_name
                        )
                    )
                    FROM information_schema.key_column_usage k
                    WHERE k.table_schema = c.table_schema
                      AND k.table_name = c.table_name
                      AND k.referenced_table_name IS NOT NULL
                ) AS table_constraints
            FROM information_schema.columns c
            WHERE c.table_schema = 'my_schema'
            GROUP BY c.table_name, c.table_schema
        ) t
        """
        cursor.execute(query)
        result = cursor.fetchone()
        
        if not result or not result["schema_json"]:
            schema = {"error": "No tables found in the my_schema database."}
        else:
            schema = json.loads(result["schema_json"])
        
        try:
            with open(schema_file, "w") as f:
                json.dump(schema, f, indent=2)
        except IOError as e:
            print(f"Warning: Failed to save schema to {schema_file}: {str(e)}")
        
        return json.dumps(schema, indent=2)
    
    except Error as e:
        return json.dumps({"error": f"Error fetching schema: {str(e)}"})
    
    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()

# Define tools list
tools = [query_mysql, fetch_schema]

# Fetch or load schema details and store as JSON
schema_json = fetch_schema("")

# Define ReAct prompt template with schema information and chat history
prompt = PromptTemplate.from_template(
    """You are a helpful assistant with access to the following tools:

{tools}

Chat History:
{chat_history}

Database Schema Information (JSON format):
{schema_json}

The schema JSON includes:
- Columns (name, data_type, is_nullable, column_default, extra, comment)
- Foreign key constraints (constraint_name, column_name, referenced_table, referenced_column)

Use the chat history to understand the context. Use query_mysql for database queries (e.g., fetching data, listing tables) targeting my_schema. Use fetch_schema to refresh the schema if needed. Ensure queries align with the schema (e.g., correct column names, data types, constraints). If the schema lacks primary keys or indexes, infer them or ask for clarification.

For query results:
- Format SELECT query results as a markdown table in a code block (```markdown) for UI display.
- For SHOW or non-data queries (e.g., INSERT, UPDATE), return plain text.
- If the output format is unclear, use table for data, text for messages.

Follow this format:

Thought: [Your reasoning]
Action: [Tool to use, one of {tool_names}]
Action Input: [Input for the tool]
Observation: [Result from the tool]

Repeat until you can provide a final answer, then conclude with:

Final Answer: [Your response, including relevant tables or results]

If the query is unclear, ask for clarification.

Question/Task: {input}
{agent_scratchpad}
"""
).partial(schema_json=schema_json)

# Create ReAct agent
converse_agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=converse_agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=30,
    max_execution_time=60
)

# Chat function
def chat_agent(history, message):
    # Format chat history for the prompt
    chat_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history])
    try:
        # Invoke agent and capture full trace
        result = agent_executor.invoke({
            "input": message,
            "chat_history": chat_history
        })
        # Combine intermediate steps (e.g., Observations with tables) and Final Answer
        response = ""
        if "intermediate_steps" in result:
            for step in result["intermediate_steps"]:
                action, observation = step
                if action.tool == "query_mysql" and observation.startswith("```markdown"):
                    response += f"{observation}\n"
        response += result["output"]
    except Exception as e:
        # Fallback for SELECT queries
        if message.strip().upper().startswith("SELECT"):
            response = query_mysql(message) + "\n\nNote: Agent stopped due to limits, but query results are shown above."
        else:
            response = f"Agent stopped due to limits: {str(e)}"
    # Append user and assistant messages to history
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})
    return history, ""

# Gradio UI
with gr.Blocks() as demo:
    chatbot = gr.Chatbot(type="messages", height=500)
    msg = gr.Textbox(placeholder="Type your question here...")
    clear = gr.Button("Clear")

    state = gr.State([])

    def respond(history, message):
        if not history:
            history = []
        updated_history, _ = chat_agent(history.copy(), message)
        print(f"Updated history: {updated_history}")  # Debugging
        return updated_history, ""

    msg.submit(respond, [state, msg], [chatbot, msg])
    clear.click(lambda: [], None, [chatbot, state])

demo.launch()
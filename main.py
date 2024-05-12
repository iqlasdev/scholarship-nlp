import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

env = os.getenv('APP_ENV', 'development')
if (env == 'development'):
    db_user = os.getenv("db_user")
    db_password = os.getenv("db_password")
    db_host = os.getenv("db_host")
    db_name = os.getenv("db_name")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
else:
    db_user = st.secrets["db_user"]
    db_password = st.secrets["db_password"]
    db_host = st.secrets["db_host"]
    db_name = st.secrets["db_name"]
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    LANGCHAIN_TRACING_V2 = st.secrets["LANGCHAIN_TRACING_V2"]
    LANGCHAIN_API_KEY = st.secrets["LANGCHAIN_API_KEY"]

print(f"Running in {env} mode")


if OPENAI_API_KEY:
    print("API Key is:", OPENAI_API_KEY)
else:
    print("API Key not found.")

from langchain_community.utilities.sql_database import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain.memory import ChatMessageHistory

from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI


st.title("Scholarship Chatbot")

@st.cache_resource
def get_chain():
    from langchain_openai import ChatOpenAI
    from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

    db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    generate_query = create_sql_query_chain(llm, db) 
    execute_query = QuerySQLDataBaseTool(db=db)
    rephrase_answer = answer_prompt | llm | StrOutputParser()

    chain = (
    RunnablePassthrough.assign(query=generate_query).assign(
        result=itemgetter("query") | execute_query
    )
    | rephrase_answer
)


    return chain

def create_history(messages):
    history = ChatMessageHistory()
    for message in messages:
        if message["role"] == "user":
            history.add_user_message(message["content"])
        else:
            history.add_ai_message(message["content"])
    return history

def invoke_chain(question):
    chain = get_chain()
   # history = create_history(messages)
    response = chain.invoke({"question": question}) #,"top_k":3,"messages":history.messages})
   # history.add_user_message(question)
    #history.add_ai_message(response)
    return response


from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder,FewShotChatMessagePromptTemplate,PromptTemplate

answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)

client = OpenAI()

# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

# Initialize chat history
if "messages" not in st.session_state:
    # print("Creating session state")
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
prompt = st.chat_input("What is up? just enter your questions..")
if prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.spinner("Generating response..."):
        with st.chat_message("assistant"):
            response = invoke_chain(prompt)
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
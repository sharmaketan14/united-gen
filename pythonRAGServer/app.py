from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_core.tools import Tool
from langchain import hub
from dotenv import load_dotenv
import os
from pydantic import BaseModel
from langchain.agents import AgentExecutor, create_react_agent, create_self_ask_with_search_agent, create_structured_chat_agent, create_openai_functions_agent
from langchain.agents.initialize import initialize_agent
from langchain.agents import create_tool_calling_agent
from langchain.agents import create_openai_tools_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langserve import add_routes
from langchain.agents.format_scratchpad.openai_tools import (format_to_openai_tool_messages,)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
import uvicorn

#Wiwipedia Wrapper
api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper)

#Google Search Tool
load_dotenv()
os.environ['GOOGLE_API_KEY']=os.getenv("GOOGLE_API_KEY")
os.environ['GOOGLE_CSE_ID']=os.getenv("GOOGLE_CSE_ID")

search = GoogleSearchAPIWrapper()

google_search = Tool(
    name="google_search",
    description="Search Google for recent results.",
    func=search.run,
)


# Document Retrieval
loader = PyPDFLoader("s41443-021-00485-w.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
text_splitter.split_documents(docs)[:5]

documents=text_splitter.split_documents(docs)
vectordb=FAISS.from_documents(documents,OllamaEmbeddings())
retriever=vectordb.as_retriever()

retriever_tool=create_retriever_tool(retriever,"document_search", "Search for information about Transgender and Intersex.")


#Archive Tool
arxiv_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)

#All Tools
tools=[google_search, wiki, arxiv]

#LLM
llm = Ollama(model="llama2")


#Prompt
prompt = hub.pull("hwchase17/react")

#Tool calling agent
agent = create_react_agent(llm, tools, prompt)

#Agent Executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# result = agent_executor.invoke({"input":"Give me some research on Transgender community."})
# print(result)

#API Rounting
app = FastAPI(
    title="UnitedGen"
)


#Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

class Question(BaseModel):
    topic: str

@app.post("/run/")
async def create_user(question : Question):
    topic = question.topic
    result = llm.invoke(topic)
    print(result)
    return {
        "msg": result
    }

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)


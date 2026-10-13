from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.tools import tool
from langchain.agents import create_agent
load=PyPDFLoader("./tollywood_movies_dataset.pdf")
v=load.load()
split=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
splitter=split.split_documents(v)
embeddings=GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vector=InMemoryVectorStore.from_documents(
    documents=splitter,
    embedding=embeddings)
@tool
def vamsi(query:str):
    """
    you are an helping agent to retrive the data based on the query 
    """
    docs=vector.similarity_search(query=query)
    context=""
    for i in docs:
        context+=i.page_content+"/n"
    return context
system_prompt="you are a agent that retrive the data based on the query and give correct" \
" answer and if u dont know jst say please ask related to the document"
llm=ChatGoogleGenerativeAI(model="gemini-3.6-flash")
agent=create_agent(
    model=llm,
    tools=[vamsi],
    system_prompt=system_prompt,

)
query="give me the details about film called eega"
res=agent.invoke({"messages":[{"role":"user","content":query}]})
print(res["messages"][-1].content)

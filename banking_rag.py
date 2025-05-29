from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import LLMChain, RetrievalQA, StuffDocumentsChain

# initialize the embedding model 
embeddings = HuggingFaceEmbeddings()

# load vector store
vector_store = FAISS.load_local("./vector_db", embeddings, allow_dangerous_deserialization=True)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# load local llm
llm = Ollama(model="deepseek-r1")

# define the prompt to feed for llm
prompt = """  
1. Rely strictly on the context provided below.
2. Respond clearly in no more than two sentences.
3. Do not refer to the context explicitly in your answer.
4. If the answer is not available, respond with: 'I don't know','I can only assist with SBI related queries.'
Context: {context}  
Question: {question}  
Answer:  
"""  
QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt)
llm_chain = LLMChain(llm=llm, prompt=QA_CHAIN_PROMPT)

# formats the documents chunk
document_prompt = PromptTemplate(
    template="Context:\ncontent:{page_content}\nsource:{source}",
    input_variables=["page_content", "source"]
)
qa = RetrievalQA(
    combine_documents_chain=StuffDocumentsChain(
        llm_chain=llm_chain,
        document_prompt=document_prompt,
        document_variable_name="context"
    ),
    retriever=retriever
)
def get_rag_answer(query):
    return qa.run(query)
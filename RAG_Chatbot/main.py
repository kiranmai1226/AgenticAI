from dotenv import load_dotenv
import os
import glob
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")
OPENAI_EMBEDDING_MODEL_NAME = os.getenv("OPENAI_EMBEDDING_MODEL_NAME")

from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI

def get_pdf_text(pdf_paths):
    text = ""
    for pdf_path in pdf_paths:
        with open(pdf_path, "rb") as pdf_file:
            reader = PdfReader(pdf_file)
            for page in reader.pages:
                text += page.extract_text()
    return text

def get_chunk_text(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vector_store):
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo-0", temperature=0)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    system_template  =  """
    Use the following pieces of context and chat history to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}

    Chat history: {chat_history}

    Question: {question}
    Helpful Answer:
    """
    prompt = PromptTemplate(
        template=system_template,
        input_variables=["context", "question",  "chat_history"],
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        verbose = True,
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )
    return conversation_chain
from langchain.evaluation.ragas import RagasEvaluator
import pandas as pd

def main():
    print("PDF QA CLI")
    pdf_input = input("Enter PDF file paths separated by commas (or leave blank to load all PDFs from a folder): ").strip()
    if pdf_input:
        pdf_paths = [p.strip() for p in pdf_input.split(",") if p.strip()]
    else:
        folder = input("Enter folder path to load all PDFs: ").strip()
        pdf_paths = glob.glob(os.path.join(folder, "*.pdf"))

    if not pdf_paths:
        print("No PDF files provided.")
        return

    print("Processing PDFs...")
    raw_text = get_pdf_text(pdf_paths)
    text_chunks = get_chunk_text(raw_text)
    vector_store = get_vector_store(text_chunks)
    conversation = get_conversation_chain(vector_store)
    print("Ready! Ask questions about your PDFs. Type 'exit' to quit.")

    qa_pairs = []

    while True:
        question = input("Your question: ")
        if question.lower() in ["exit", "quit"]:
            break
        try:
            response = conversation({'question': question})
            chat_history = response.get('chat_history', [])
            if chat_history:
                answer = chat_history[-1].content
                print("AI:", answer)
                reference = input("Enter the reference (expected) answer for evaluation (or leave blank to skip): ")
                if reference:
                    qa_pairs.append({"question": question, "answer": answer, "reference": reference})
            else:
                print("No answer.")
        except Exception as e:
            print(f"Error: {e}")

    # Evaluate with Ragas if any pairs collected
    if qa_pairs:
        df = pd.DataFrame(qa_pairs)
        evaluator = RagasEvaluator()
        results = evaluator.evaluate(df)
        print("\nRagas Evaluation Results:")
        print(results)
    else:
        print("No Q/A pairs to evaluate.")

if __name__ == '__main__':
    main()
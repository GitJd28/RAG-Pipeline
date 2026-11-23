import os 
# where you would read the document(text , files ppt etc.) from 
from langchain_community.document_loaders import TextLoader , DirectoryLoader 
# to chunk the data 
from langchain_text_splitters import CharacterTextSplitter 
# Using OpenAI embedding model to help it conver it to a vector DB
from langchain_openai import OpenAIEmbeddings 
# the vector db will be now stoired in chrome DB 
from langchain_chroma import Chroma 
#load all the environment variables 
from dotenv import load_dotenv 


load_dotenv() 

def load_documents(docs_path="docs"):
    """load all the files""" 
    print("\nLoading all the documents from {docs_path} .......")

    #check if the files exist in the given directory. 
    if not os.path.exists(docs_path) :
        raise FileNotFoundError(f"The given directory {docs_path} is not found. Please insert files and try again.")
  
    #Load all the .txt files from the docs directory
    loader= DirectoryLoader(
        path = docs_path , 
        glob= "*.txt", 
        loader_cls= TextLoader   #we have got several classes to load certail types of inpouts like text , pdf or website etc. 
    )

    documents= loader.load()  #gives us a list of langchain documents. where it keeps the source and the metadata. 

    if(len(documents)==0):
        raise FileNotFoundError(f"No .txt files found in {docs_path} . Please upload the files. ")

    for i , doc in enumerate(documents[ :3]) :
        print(f"\n Document {i+1} : ")
        print(f" Source:{doc.metadata['source']} ")
        print(f" Content length: {len(doc.page_content)} characters ")
        print(f" Content preview: {doc.page_content[:100]} ......")
        print(f" Metadata: {doc.metadata}\n" )

    return documents    


def split_documents(documents, chunk_size=900, chunk_overlap=0):
    """Split the documents into chunks with overlap"""
    print("\nSplitting the documents into chunks")

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap   #decides how much text is repeated between 2 consecutive chunks. Helps preserve contextual meaning. 
    )

    # chunk overlap - If not 0 helps imporve retreival quality. 

    chunks = text_splitter.split_documents(documents)

    if chunks:
        for i, chunk in enumerate(chunks[:6]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print("Content:")
            print(chunk.page_content)
            print("-" * 50)

        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks")

    return chunks

def main():
    print("\nMain func heyyy :) \n")

    #Step 1. Load the file 
    documents = load_documents(docs_path="docs")
    #Step 2. Chunking the files 
    chunks = split_documents(documents)
    #Setp 3. Embedding and storing in Vector DB
    


if __name__ == "__main__" : 
    main()
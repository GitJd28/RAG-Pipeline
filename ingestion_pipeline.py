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

def main():
    print("\n\nMain func heyyy :) ")


if __name__ == "__main__" : 
    main()
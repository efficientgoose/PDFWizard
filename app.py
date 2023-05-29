import streamlit as st 
# from dotenv import load_dotenv
import pickle
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

import os

# Coding the sidebar

os.environ['OPENAI_API_KEY']='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'


with st.sidebar:
    st.title("PDF Wizard 💬🤖")
    st.markdown('''
        ## About
        Experience the ultimate PDF companion with our user-friendly LLM-powered chatbot! Effortlessly upload your PDFs and ask any questions related to the PDF.
                ''')
    
    st.write("Made with ❤️ by [Ajinkya Kale](https://www.linkedin.com/in/ajinkode/)")
    
    

def main():
    st.header("Chat with PDF 💬")
    # load_dotenv()
    
    # upload a pdf file
    pdf = st.file_uploader("Upload your PDF", type='pdf')
    
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
    
        text = ""
        for page in pdf_reader.pages:
            text+=page.extract_text()
            
        # st.write(text)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap=200,
            length_function=len
        )
        
        chunks = text_splitter.split_text(text=text)
        
        # Create Embeddings
        store_name = pdf.name[:-4]
        
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
                # st.write("Embeddings loaded from the Disk! Money saved!")
                
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)
                
                
        # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file: ")
        # st.write(query)
        
        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
            
            llm = OpenAI(temperature=0)
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                # st.write("Total tokens and total cost associated: ")
                st.write(cb)
            
            st.subheader("Result: ")
            st.write(response)
            
                
                
if __name__ == '__main__':
    main()
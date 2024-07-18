# LLM App to extract answerd from CSV file

import os
import streamlit as st
from enum import Enum
from langchain_groq import ChatGroq
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter

CREATIVITY=0
# os.environ["TOKENIZERS_PARALLELISM"] = False
TEMPLATE = """
Given the following context and a question, generate an answer based on this context only.
In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
If the answer is not found in the context, respond "I don't know." Don't try to make up an answer.

CONTEXT: {context}

QUESTION: {question}
"""


class ModelType(Enum):
    GROQ='GroqCloud'
    OPENAI='OpenAI'


class LLMModel:
    def __init__(self, model_provider: str) -> None:
        self.model_provider = model_provider

    def load(self, api_key=str):
        try:
            if self.model_provider==ModelType.GROQ.value:
                llm = ChatGroq(temperature=CREATIVITY, model="llama3-70b-8192", api_key=api_key) # model="mixtral-8x7b-32768"
            if self.model_provider==ModelType.OPENAI.value:
                llm = OpenAI(temperature=CREATIVITY, api_key=api_key)
            return llm
        
        except Exception as e:
            raise e


class LLMStreamlitUI:
    def __init__(self) -> None:
        pass

    def validate_api_key(self, key:str):
        if not key:
            st.sidebar.warning("Please enter your API Key")
            # st.stop()
        else:    
            if (key.startswith("sk-") or key.startswith("gsk_")):
                st.sidebar.success("Received valid API Key!")
            else:
                st.sidebar.error("Invalid API Key!")

    def get_api_key(self):
        
        # Get the API Key to query the model
        input_text = st.sidebar.text_input(
            label="Your API Key",
            placeholder="Ex: sk-2twmA8tfCb8un4...",
            key="api_key_input",
            type="password"
        )

        # Validate the API key
        self.validate_api_key(input_text)
        return input_text
    
    def generage_vectordb(self, document, embeddings, fpath):
        # Save vecotr database locally
        vectordb = FAISS.from_documents(document, embeddings)
        vectordb.save_local(fpath)
        return vectordb
    
    def create(self):
        try:
            # Set the page title for blog post
            st.set_page_config(page_title="Ask from CSV File with FAQs about Napoleon")
            st.markdown("<h1 style='text-align: center;'>Ask from CSV File with FAQs about Napoleon</h1>", unsafe_allow_html=True)

            # Select the model provider
            option_model_provider = st.sidebar.selectbox(
                    'Select the model provider',
                    ('GroqCloud', 'OpenAI')
                )

            # Input API Key for model to query
            api_key = self.get_api_key()

            # Get the question from user
            question = st.text_input("Question: ")
            if question:
                if not api_key:
                    st.warning("Please insert your API Key", icon="⚠️")
                    st.stop()

                # Get current working directory
                cwd = os.getcwd()

                # Load th ecsv file
                loader = CSVLoader(file_path=os.path.join(cwd, 'napoleon-faqs.csv'), source_column="prompt")
                document = loader.load()

                # Create embeddings and store into FAISS vector db
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

                # Load the vector database from the local folder
                vectordb_file_path = os.path.join(cwd, "my_vectordb")
                if os.path.exists(vectordb_file_path):
                    vectordb = FAISS.load_local(vectordb_file_path, embeddings, allow_dangerous_deserialization=True)
                else:
                    # Generate and save vector database
                    vectordb = self.generage_vectordb(document, embeddings, vectordb_file_path)

                # Create a retriever for querying the vector database
                retriever = vectordb.as_retriever(score_threshold=0.7)

                # Get the prompt
                prompt = PromptTemplate.from_template(
                    # input_variables=["question", "context"],
                    template=TEMPLATE
                )
                
                # Load the LLM model
                llm_model = LLMModel(model_provider=option_model_provider)
                llm = llm_model.load(api_key=api_key)

                # Create retrieval QA chain
                chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    input_key="query",
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": prompt}
                )
                answer = chain.invoke(question)

                st.markdown("Here's your answer:")
                st.info(answer['result'])

                btn = st.button("Recreate Vector DB")
                if btn:
                    # Generate and save vector database
                    vectordb = self.generage_vectordb(document, embeddings, vectordb_file_path)
                    st.write("Vector DB Recreated!")


        except Exception as e:
            st.error(str(e), icon=":material/error:")



def main():
    # Create the streamlit UI
    st_ui = LLMStreamlitUI()
    st_ui.create()


if __name__ == "__main__":
    main()
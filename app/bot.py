#pip install atlassian-python-api pytesseract Pillow svglib
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import ConfluenceLoader
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Chroma


from langchain.llms import AzureOpenAI
# from streamlit_chat import message
# import streamlit as st
import pandas as pd
import random 
import os



# st.set_page_config(page_title="Confluence investigator Chatbot", page_icon=":robot_face:")
# st.title("Confluence investigator Chatbot")
# st.markdown("This is a chatbot that can answer questions about with confluence pages ")

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2022-12-01"
os.environ["OPENAI_API_BASE"] = "https://xxxx.openai.azure.com/"
os.environ["OPENAI_API_KEY"] = "xxxxx"



loader = ConfluenceLoader(
    url="https://xxx.atlassian.net/wiki",
    username="xxxx@gmail.com",
    api_key="xxxxxx"
)
documents = loader.load(space_key="lola", include_attachments=True, limit=50)
print(documents)
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
documents = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(deployment='text-embedding-ada-002')

vectorstore = Chroma.from_documents(documents, embeddings)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa = ConversationalRetrievalChain.from_llm(AzureOpenAI(temperature=0,
                                                       deployment_name='text-davinci-002',
                                                       model_name="text-davinci-002"), vectorstore.as_retriever(), memory=memory)
query = "what is this doc ?"
result = qa({"question": query})
print(result["answer"])


# if input_file is not None:


#     agent = create_csv_agent(AzureOpenAI(temperature=0 ,
#                                         verbose=True,
#                                         deployment_name='text-davinci-002',
#                                         model_name="text-davinci-002", 
#                                         max_tokens=1000),files_data)
#     agent.agent.llm_chain.prompt.template


#     st.session_state['generated'] = []
#     st.session_state['past'] = []


#     if 'generated' not in st.session_state:
#         st.session_state['generated'] = []
#     if 'past' not in st.session_state:
#         st.session_state['past'] = []
        
        

#     user_input=st.text_input("You:",key='input')

#     if user_input:
#         output=agent.run(user_input)
#         st.session_state['past'].append(user_input)
#         st.session_state['generated'].append(output)
#         if st.session_state['generated']:
#             for i in range(len(st.session_state['generated'])-1, -1, -1):
#                 message(st.session_state["generated"][i], key=str(i))
#                 message(st.session_state['past'][i], is_user=True, key=str(i) + '_user') 
#                 st.session_state.generated = ''


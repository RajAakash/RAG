# Install required libraries
!pip install -q langchain
!pip install -q torch
!pip install -q transformers
!pip install -q sentence-transformers
!pip install -q datasets
!pip install -q faiss-cpu

# Import necessary libraries
from langchain.document_loaders import HuggingFaceDatasetLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import pipeline
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA

# Specify the dataset name and the column containing the content
dataset_name = "nhankins/legal_contracts"
page_content_column = "text"  # Adjust based on actual dataset structure

# Create a loader instance
loader = HuggingFaceDatasetLoader(dataset_name, page_content_column)

# Load the data
data = loader.load()

# Display the first few entries
print(data[:2])

# Create an instance of the RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

# Split the text into documents using the text splitter
docs = text_splitter.split_documents(data)

# Define the model for embeddings - choose a model fine-tuned on legal texts if available
modelPath = "nlpaueb/legal-bert-small-uncased"

# Initialize an instance of HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)

# Use FAISS to create a vector store from the documents
db = FAISS.from_documents(docs, embeddings)

# Setup for retrieval and question-answering
tokenizer = AutoTokenizer.from_pretrained(modelPath)
model = AutoModelForQuestionAnswering.from_pretrained(modelPath)

question_answerer = pipeline(
    "question-answering", 
    model=model,
    tokenizer=tokenizer,
    return_tensors='pt'
)

llm = HuggingFacePipeline(pipeline=question_answerer, model_kwargs={"temperature": 0.7, "max_length": 512})

# Create a retriever object
retriever = db.as_retriever()

# Create a RetrievalQA instance
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="refine", retriever=retriever, return_source_documents=False)

# Example query
question = "What does the confidentiality clause state?"
result = qa.run({"query": question})
print(result["result"])

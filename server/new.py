from flask import Flask, request, jsonify
from langchain.document_loaders import YoutubeLoader
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain import PromptTemplate

app = Flask(__name__)

# Your OpenAI API key is set here
OPENAI_API_KEY = ''

# Initialize the OpenAI model with your API key
llm = OpenAI(api_key=OPENAI_API_KEY)

def create_db_from_youtube_video_url(video_url: str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)
    
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    db = FAISS.from_documents(docs, embeddings)
    return db

def get_response_from_query(db, query, k=4):
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])
    
    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        **Role:** Blog Maker Assistant

    **Task:** Create detailed and engaging blogs in markdown format based on the provided video's transcript.

    **Instructions:**
    - **Content:** Use only factual information from the transcript.
    - **Format:** Your blog should be verbose, detailed, and formatted using markdown syntax. This includes:
        - Using `#` for headers.
        - `-` for bullet points.
        - `**` for bold text.
        - `_` for italic text.
        - `[Link text](URL)` for hyperlinks.
    - **Structure:** Ensure the blog is well-structured, with a clear introduction, body, and conclusion.
    - **Insufficient Information:** If there is not enough information for a comprehensive blog, reply with "I don't know".

    **Objective:** Produce a well-structured, informative, and engaging blog that provides readers with a deep understanding of the topic, formatted correctly in markdown.
        """,
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(question=query, docs=docs_page_content).replace("\n", "")
    return response

@app.route('/query_video', methods=['POST'])
def query_video():
    data = request.json
    youtube_url = data.get('youtube_url')
    query = data.get('query')
    
    if not youtube_url or not query:
        return jsonify({"error": "Both YouTube URL and query are required"}), 400
    
    try:
        db = create_db_from_youtube_video_url(youtube_url)
        response = get_response_from_query(db, query)
        return jsonify({"response": response}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
    
    # You are a helpful assistant that can answer questions about youtube videos 
    #     based on the video's transcript.
        
    #     Answer the following question: {question}
    #     By searching the following video transcript: {docs}
        
    #     Only use the factual information from the transcript to answer the question.
        
    #     If you feel like you don't have enough information to answer the question, say "I don't know".
        
    #     Your answers should be verbose and detailed.
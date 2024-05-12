from flask import Flask, request, jsonify
from langchain.document_loaders import YoutubeLoader
from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter

app = Flask(__name__)

OPENAI_API_KEY = ''  # Your OpenAI API key here

# Initialize the OpenAI model
llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

@app.route('/summarize/video', methods=['POST'])
def summarize_video():
    data = request.json
    youtube_url = data.get('youtube_url')
    chain_type = data.get('chain_type', 'stuff')  # Default to 'stuff'

    try:
        loader = YoutubeLoader.from_youtube_url(youtube_url, add_video_info=True)
        result = loader.load()
        
        chain = load_summarize_chain(llm, chain_type=chain_type, verbose=False)
        summary = chain.run(result)
        
        return jsonify({'summary': summary}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/summarize/multiple_videos', methods=['POST'])
def summarize_multiple_videos():
    data = request.json
    youtube_urls = data.get('youtube_urls')
    texts = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)

    try:
        for url in youtube_urls:
            loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
            result = loader.load()
            texts.extend(text_splitter.split_documents(result))
        
        chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=False)
        summary = chain.run(texts)
        
        return jsonify({'summary': summary}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
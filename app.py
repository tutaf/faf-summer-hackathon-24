import concurrent.futures
import json
import os
import html2text
import aiohttp
import asyncio

import vertexai
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_together import ChatTogether
from flask import Flask, jsonify, request
import asyncio
from langchain_google_vertexai import VertexAI



# Initialize the models for natural language processing
chat = ChatTogether(
    model="meta-llama/Llama-3-70b-chat-hf",
    temperature=0.0,
    together_api_key=os.environ['TOGETHER_API_KEY']
)

mistral_chat = ChatTogether(
    model="mistralai/Mixtral-8x22B-Instruct-v0.1",
    temperature=0.02,
    together_api_key=os.environ['TOGETHER_API_KEY']
)

vertexai.init(project="deductive-mix-407619")

comparison_chat = VertexAI(
    model="gemini-1.5-flash",
    project_id="deductive-mix-407619",
)

search_template_text = """<s>[INST] Evaluate the usefulness of these search results for the query "{query}". Make sure the name of the product matches the query:
    {results}
    Output a JSON list containing exactly 3 links to the most relevant results. Follow these rules strictly:
    1. Only include links to comprehensive text reviews of the product.
    2. Do NOT include links to videos, video reviews, or multimedia content.
    3. Prioritize reviews from independent creators. Avoid commercial websites with generic articles, unless they specialize in this domain.
    4. Exclude product reveals, announcements, and any publications that do not provide a thorough user experience overview.
    5. Exclude links to websites that only list product specifications and nothing else, wiki pages, or stores/marketplaces.
    Your output must strictly be a JSON list with exactly 3 items, each being a link from the search results. Do not include any other text or explanation, only the JSON list.
    Output:[/INST]"""

comparison_template_text = """You will need to help user compare {product1_name}  and {product2_name}. 
    Your user is not a professional, so you shouldn't overwhelm them with technical terms and numbers. Focus on what's important for this user, on their experience with the product.
    
    Also, user has shared some info about themselves. It is paramount to look at the choice between these two products through user's eyes. Keep in mind that the user is not tech-savvy, so instead of overwhelming them with numbers, specs and technical terms, explain what it means to for them. Explain what user's experience will be like, not how many megapixels a camera has (this is just an example of a useless specification). 
    Here's the information shared by user: "{user_request}"
    You MUST keep user's usecase in mind when comparing the products. 

    You will output JSON containing two things:
    1. A JSON list named "comparisons". The list will contain comparisons by a few categories. You are free create from 2 to 5 categories. Each list entry will have:
        1.1. A "category_title" - 1-3 word long category name;
        1.2. "category_description" - 1-2 sentences telling what this is and why this comparison criterion is important;
        1.3. "product1_text" - A few sentences, explaining how the first product performs in this category, what its downsides and advantages are compared to product2. Use clear language, remember you're trying to be useful, but don't make things complicated for user. Do NOT overwhelm user with numbers and technical terms. Remember, you are trying to help the user, so keep his usecase in mind when comparing the products.
        1.4. "product2_text" - Same as for product above, but for product 2. Do NOT overwhelm user with numbers and technical terms. Do not use the exact same wordings as in "product1_text", show a little creativity.     
    2. A JSON item called "final_verdict" - explain what is the better choice overall and why, up to 3 sentences long. Do NOT overwhelm user with numbers and technical terms. No ambiguity or "it's up to you" is allowed here, you must provide a definitive answer.
    
    Remember, your comparison descriptions and verdict must be easy to comprehend for a user, who is new to the topic and does not know all the intricacies of technology.
    Also, you MUST output your answer in JSON format. 

    Products to compare:
    ===== {product1_name} REVIEWS START HERE =====
    {product1_content}
    ===== {product1_name} REVIEWS END HERE =====
    
    ===== {product2_name} REVIEWS START HERE =====:
    {product2_content}
    ===== {product2_name} REVIEWS END HERE =====
    You HAVE TO output your answer in JSON.
    You MUST ensure your output is a VALID JSON.
    Your response must contain only the JSON and nothing else. Do NOT envelop the json contents in triple backticks, do NOT output markdown.
    JSON OUTPUT:"""

all_links = []

# Async function to fetch search results using aiohttp
async def fetch_search_results(session, query):
    search_url = f'https://api.search.brave.com/res/v1/web/search?q={query}'
    headers = {
        'Accept': 'application/json',
        'Accept-Encoding': 'gzip',
        'X-Subscription-Token': os.environ['BRAVE_SEARCH_API_KEY']
    }
    async with session.get(search_url, headers=headers) as response:
        return await response.json()

# Async function to perform web scraping and return cleaned content
async def do_webscraping(session, link):
    print(link)
    if not link in all_links:
        print(f"АХТУНГ - {link}")


    # Creating an executor to run synchronous code in an async manner
    with concurrent.futures.ThreadPoolExecutor() as pool:
        loader = AsyncHtmlLoader([link])
        # Run the synchronous load function in an executor
        docs = await asyncio.get_event_loop().run_in_executor(pool, loader.load)

        html2text_transformer = Html2TextTransformer()
        # Similarly, transforming documents
        docs_transformed = await asyncio.get_event_loop().run_in_executor(pool, html2text_transformer.transform_documents, docs)

        if docs_transformed:
            metadata = docs_transformed[0].metadata
            return {
                'summary': docs_transformed[0].page_content,
                'title': metadata.get('title', ''),
                'metadata': metadata,
                'clean_content': html2text.html2text(docs_transformed[0].page_content)
            }
        return None

# Function to parse the search results
def parse_search_results(response):
    results = []
    for result in response.get('web', {}).get('results', []):
        results.append({
            'title': result['title'],
            'snippet': result['description'],
            'url': result['url']
        })
        all_links.append(result['url'])
        print(results[-1])
    return results

# Function to evaluate the relevance of the search results
def evaluate_results(query, results):
    results_text = json.dumps([{"title": result["title"], "url": result["url"], "snippet": result["snippet"]} for result in results], indent=0)
    input_data = {"query": query, "results": results_text}
    final_prompt = search_template_text.format(query=input_data["query"], results=input_data["results"])
    output = chat.invoke(final_prompt)
    return json.loads(output.content)



# Function to generate a comparison based on two sets of product reviews
async def generate_comparison(product1, product2, user_request):
    prompt = comparison_template_text.format(product1_name=product1['name'], product1_content=product1['content'],
                                             product2_name=product2['name'], product2_content=product2['content'],
                                             user_request=user_request)

    with open('comparison_prompt.txt', 'w', encoding='utf-8') as f:
        print(prompt, file=f)

    # Using ThreadPoolExecutor to run synchronous code asynchronously
    with concurrent.futures.ThreadPoolExecutor() as pool:
        comparison = await asyncio.get_event_loop().run_in_executor(
            pool, comparison_chat.invoke, prompt
        )
        print(comparison)
        return comparison


# Main workflow to compare two products, using async
async def compare_two_products(product1, product2, user_request):
    async with aiohttp.ClientSession() as session:
        product1_response = await fetch_search_results(session, product1+" review")
        product1_results = parse_search_results(product1_response)
        product1_evaluated = evaluate_results(product1+" review", product1_results)

        product2_response = await fetch_search_results(session, product2+" review")
        product2_results = parse_search_results(product2_response)
        product2_evaluated = evaluate_results(product2+" review", product2_results)

        # Scrape reviews concurrently
        product1_reviews = await asyncio.gather(*[do_webscraping(session, url) for url in product1_evaluated])
        product2_reviews = await asyncio.gather(*[do_webscraping(session, url) for url in product2_evaluated])

        # Combine and clean review contents for comparison
        product1_content = " ".join([review['clean_content'] for review in product1_reviews if review])
        product2_content = " ".join([review['clean_content'] for review in product2_reviews if review])

        # Prepare products data for comparison
        product1_data = {'name': product1, 'content': product1_content}
        product2_data = {'name': product2, 'content': product2_content}

        # Generate and return the comparison
        return await generate_comparison(product1_data, product2_data, user_request)


# Main async function
app = Flask(__name__)


# This is your existing asynchronous function wrapped in a new function to handle Flask requests.
@app.route('/compare_products', methods=['POST'])
def compare_products():
    data = request.get_json()
    product1 = data.get('product1')
    product2 = data.get('product2')
    user_request = data.get('user_request')

    # We wrap the async call with asyncio.run to ensure it is run in an event loop
    comparison_result = asyncio.run(compare_two_products(product1, product2, user_request))

    # Return JSON result
    return comparison_result


# Ensure to have your asyncio logic functions here or import them if they are in another file.

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)  # Turn off reloader if running in production
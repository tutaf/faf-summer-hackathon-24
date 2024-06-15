import json

import html2text
import requests
import os
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_together import ChatTogether

# Define the model for invoking advanced natural language processing tasks
chat = ChatTogether(
    model="meta-llama/Llama-3-70b-chat-hf",
    temperature=0.0,
    together_api_key=os.environ['TOGETHER_API_KEY']
)

comparison_chat = ChatTogether(
    # model="Qwen/Qwen2-72B-Instruct",
    model="mistralai/Mixtral-8x22B-Instruct-v0.1",
    temperature=0.0,
    together_api_key=os.environ['TOGETHER_API_KEY']
)

search_template_text = """<s>[INST] Evaluate the usefulness of these search results for the query "{query}":
{results}
Output a JSON list containing exactly 3 links to the most relevant results. Follow these rules strictly:
1. Only include links to comprehensive text reviews of the product.
2. Do NOT include links to videos, video reviews, or multimedia content.
3. Prioritize reviews from independent creators. Avoid big news agencies or commercial websites with generic articles.
4. Exclude product reveals, announcements, and any publications that do not provide a thorough user experience overview.
5. Exclude links to websites that only list product specifications, wiki pages, news websites, or stores/marketplaces.
6. Ensure the chosen reviews are objective, unbiased, and not sponsored.

Your output must strictly be a JSON list with exactly 3 items, each being a link from the search results. Do not include any other text or explanation, only the JSON list.

Output:[/INST]"""

all_links = []

# Function to fetch search results from Brave search API
def fetch_search_results(query):
    search_url = f'https://api.search.brave.com/res/v1/web/search?q={query}'
    headers = {
        'Accept': 'application/json',
        'Accept-Encoding': 'gzip',
        'X-Subscription-Token': os.environ['BRAVE_SEARCH_API_KEY']
    }
    response = requests.get(search_url, headers=headers)
    return response.json()

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
    return results

# Function to evaluate the relevance of the search results
def evaluate_results(query, results):
    results_text = json.dumps([{"title": result["title"], "url": result["url"], "snippet": result["snippet"]} for result in results], indent=0)
    input_data = {"query": query, "results": results_text}
    final_prompt = search_template_text.format(query=input_data["query"], results=input_data["results"])
    output = chat.invoke(final_prompt)
    return json.loads(output.content)

# Function to perform web scraping and return cleaned content
def do_webscraping(link):
    urls = [link]
    if not link in all_links:
        print(f"AAAAAAAAA - {link}")
    loader = AsyncHtmlLoader(urls)
    docs = loader.load()
    html2text_transformer = Html2TextTransformer()
    docs_transformed = html2text_transformer.transform_documents(docs)
    if docs_transformed:
        metadata = docs_transformed[0].metadata
        return {
            'summary': docs_transformed[0].page_content,
            'title': metadata.get('title', ''),
            'metadata': metadata,
            'clean_content': html2text.html2text(docs_transformed[0].page_content)
        }
    else:
        return None


comparison_template_text = """<s>[INST] You will need to help user compare {product1_name} and {product2_name}. 
    Your user is not a professional, so you shouldn't overwhelm them with technical terms and numbers. Focus on what's important for this user, on their experience with the product.

    You will output JSON containing two things:
    1. A list. The list will contain comparisons by a few categories. You are free create from 2 to 5 categories. Each list entry will have:
        1.1. A "category_title" - 1-3 word long category name;
        1.2. "category_description" - 1-3 sentences telling what this is and why this comparison criterion is important;
        1.3. "product1_text" - A few sentences, explaining how the first product performs in this category, what its downsides and advantages are. Use clear language, remember you're trying to be useful, but don't make things complicated for user. 
        1.4. "product2_text" - Same, but for product 2. 
    2. An item called "final_verdict" - explain what is the better choice overall and why. 1-3 sentences long.


    Products to compare:
    {product1_name}:
    {product1_content}
    =====
    {product2_name}:
    {product2_content}
    JSON OUTPUT:[/INST]"""

# Function to generate a comparison based on two sets of product reviews
def generate_comparison(product1, product2):
    prompt = comparison_template_text.format(product1_name=product1['name'], product1_content=product1['content'],
                                             product2_name=product2['name'], product2_content=product2['content'])
    comparison = comparison_chat.invoke(prompt)
    return comparison

# Main workflow to compare two products
def compare_two_products(product1_query, product2_query):
    # Fetch and evaluate results for both products
    product1_response = fetch_search_results(product1_query)
    product1_results = parse_search_results(product1_response)
    product1_evaluated = evaluate_results(product1_query, product1_results)

    product2_response = fetch_search_results(product2_query)
    product2_results = parse_search_results(product2_response)
    product2_evaluated = evaluate_results(product2_query, product2_results)

    # Scrape the best sources for each product
    product1_reviews = [do_webscraping(url) for url in product1_evaluated]
    product2_reviews = [do_webscraping(url) for url in product2_evaluated]

    # Combine and clean review contents for comparison
    product1_content = " ".join([review['clean_content'] for review in product1_reviews if review])
    product2_content = " ".join([review['clean_content'] for review in product2_reviews if review])

    # Prepare products data for comparison
    product1_data = {'name': product1_query, 'content': product1_content}
    product2_data = {'name': product2_query, 'content': product2_content}

    # Generate and return the comparison
    return generate_comparison(product1_data, product2_data)

# Usage
product1_query = "realme 10 pro review"
product2_query = "samsung galaxy s22 review"
comparison_result = compare_two_products(product1_query, product2_query)
print(comparison_result.content)

import json
import requests
import os
import html2text
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_together import ChatTogether
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Fetch search results from Brave API
def fetch_search_results(query):
    print(f"searching {query}")
    search_url = f'https://api.search.brave.com/res/v1/web/search?q={query}'
    headers = {
        'Accept': 'application/json',
        'Accept-Encoding': 'gzip',
        'X-Subscription-Token': os.environ['BRAVE_SEARCH_API_KEY']
    }
    response = requests.get(search_url, headers=headers)
    print("search complete")
    return response.json()

# Parse search results
def parse_search_results(response):
    results = []
    for result in response.get('web', {}).get('results', []):
        results.append({
            'title': result['title'],
            'snippet': result['description'],
            'url': result['url']
        })
    return results

# Scrape and transform webpage content
def do_webscraping(link):
    print(f"scraping {link}")
    try:
        response = requests.get(link)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        content = ' '.join([para.get_text() for para in paragraphs])
        clean_content = html2text.html2text(content)
        print(f"scraping complete - {link}")
        return clean_content
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

# Evaluate search results using LangChain and Together API
def evaluate_results(query, results):
    print(f"evaluating search results - {query}")
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

    results_text = json.dumps(
        [{"title": result["title"], "url": result["url"], "snippet": result["snippet"]} for result in results],
        indent=0
    )

    input_data = {
        "query": query,
        "results": results_text
    }

    final_prompt = search_template_text.format(query=input_data["query"], results=input_data["results"])
    output = chat.invoke(final_prompt)
    print(f"search evaluation complete - {query}")
    return output

# Compare two products using scraped text data
def compare_products(product1_name, product2_name, product1_content, product2_content):
    print("comparing products")
    comparison_template_text = f"""<s>[INST] You will need to help user compare the following to products. 
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

    print(comparison_template_text)
    # final_prompt = comparison_template_text.format(product1_content=product1_content, product2_content=product2_content)
    output = comparison_chat.invoke(comparison_template_text)
    return output

# Initialize LangChain's Together model for search
chat = ChatTogether(
    model="meta-llama/Llama-3-70b-chat-hf",
    together_api_key=os.environ['TOGETHER_API_KEY']
)

# Initialize LangChain's Together model for comparison
comparison_chat = ChatTogether(
    model="mistralai/Mixtral-8x22B-Instruct-v0.1",
    # model="Qwen/Qwen2-72B-Instruct",
    together_api_key=os.environ['TOGETHER_API_KEY']
)
print("initialization complete")


# Define a function to handle the complete process for one product
def process_product(query):
    search_response = fetch_search_results(query)
    search_results = parse_search_results(search_response)
    evaluated_results = evaluate_results(query, search_results)
    relevant_urls = json.loads(evaluated_results.content)
    review_content = ""
    for url in relevant_urls:
        response = do_webscraping(url)
        if response:
            review_content += response + "\n\n-----\n\n"
    return review_content


product1_name = "moto g34"
product2_name = "nokia g42"
# Queries for the two products
query1 = f"{product1_name} review"
query2 = f"{product2_name} review"


# Use ThreadPoolExecutor to process both products in parallel
with ThreadPoolExecutor() as executor:
    future1 = executor.submit(process_product, query1)
    future2 = executor.submit(process_product, query2)

    review_content1 = future1.result()
    review_content2 = future2.result()

# Compare the two products using Together API
comparison_result = compare_products(product1_name, product2_name, review_content1, review_content2)
print(comparison_result)

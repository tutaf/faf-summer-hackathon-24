import json

import requests
import os
from getpass import getpass



def fetch_search_results(query):
    search_url = f'https://api.search.brave.com/res/v1/web/search?q={query}'
    headers = {
        'Accept': 'application/json',
        'Accept-Encoding': 'gzip',
        'X-Subscription-Token': os.environ['BRAVE_SEARCH_API_KEY']
    }
    response = requests.get(search_url, headers=headers)
    return response.json()


def parse_search_results(response):
    results = []
    for result in response.get('web', {}).get('results', []):
        results.append({
            'title': result['title'],
            'snippet': result['description'],  # or 'snippet' if description is not available
            'url': result['url']
        })
    return results

query = "realme 10 pro review"
search_response = fetch_search_results(query)
search_results = parse_search_results(search_response)

print(search_results)







from langchain_together import ChatTogether
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Define the model
chat = ChatTogether(
    model="meta-llama/Llama-3-70b-chat-hf",
    together_api_key=os.environ['TOGETHER_API_KEY']
)


template_text = """<s>[INST] Evaluate the usefulness of these search results for the query "{query}":
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


# Define a function to evaluate results
def evaluate_results(query, results):
    results_text = json.dumps(
        [{"title": result["title"], "url": result["url"], "snippet": result["snippet"]} for result in results],
        indent=0)
    print(results_text)
    input_data = {
        "query": query,
        "results": results_text
    }

    # Format the prompt
    final_prompt = template_text.format(query=input_data["query"], results=input_data["results"])

    # Invoke the model with the formatted prompt
    output = chat.invoke(final_prompt)
    return output

# Example usage
query = f"{query}"
evaluated_results = evaluate_results(query, search_results)
print(evaluated_results)
print(evaluated_results.content)


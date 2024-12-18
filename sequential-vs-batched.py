# Standard library imports
import time
from typing import List, Tuple

# Third-party imports
from ollama import Client

def fetch_query_response(client: Client, query: str) -> str:
    """
    Makes a single API call to the LLM model and returns the response
    - Uses the 3B parameter version of Llama 3.2
    - Includes a system message to specify plain text output
    - Includes the user's query
    Args:
        client: Ollama client
        query: The query to send to the LLM
    Returns:
        The response string from the LLM
    """
    return client.chat(
        model='llama3.2:3b',
        messages=[
            {
                'role': 'system',
                'content': 'Respond to these queries in plain text format. Be descriptive wherever it makes sense.',
            },
            {
                'role': 'user',
                'content': query,
            },
        ],
    )

def process_sequential(queries: List[str]) -> Tuple[List[str], float]:
    """
    Process queries one after another in sequential order
    Args:
        queries: List of questions to ask the LLM
    Returns:
        tuple: (list_of_responses, execution_time)
    """
    client = Client()
    start_time = time.time()
    responses = []  # Store responses

    for query in queries:
        response = fetch_query_response(client, query)
        responses.append(response['message']['content'])

    return responses, time.time() - start_time

def process_batched(queries: List[str], batch_size: int = 3) -> Tuple[List[str], float]:
    """
    Process queries in batches, which might be more efficient for some scenarios
    Args:
        queries: List of questions to ask the LLM
        batch_size: Number of queries to process in each batch
    Returns:
        tuple: (list_of_responses, execution_time)
    """
    client = Client()
    start_time = time.time()
    responses = []  # Store responses

    for i in range(0, len(queries), batch_size):
        batch = queries[i:i + batch_size]
        for query in batch:
            response = fetch_query_response(client, query)
            responses.append(response['message']['content'])

    return responses, time.time() - start_time

# Test data: variety of questions to compare processing methods
queries = [
    "What is 2+2?",
    "What is the capital of France?",
    "Who wrote Romeo and Juliet?",
    "What is the speed of light?",
    "What is the largest planet?"
]

# Run both processing methods and measure their execution times
sequential_responses, sequential_time = process_sequential(queries)
batched_responses, batched_time = process_batched(queries, batch_size=3)

# Output the timing results
print(f"Sequential processing time: {sequential_time:.2f} seconds")
print(f"Batched processing time: {batched_time:.2f} seconds", end="\n\n")

# Display all query-response pairs
for query, response in zip(queries, sequential_responses):
    print(f"Query: {query}")
    print(f"Response: {response}", end="\n\n")
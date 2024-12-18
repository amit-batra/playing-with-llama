# Import required functions from the ollama package
from ollama import chat, Client, ChatResponse

# List of test questions to send to the LLM
test_queries = [
  "What is 2 + 2?",
  "What is the capital of France?",
  "Who wrote Romeo and Juliet?",
  "What is the speed of light?",
  "What is the largest planet?"
]

# Initialize the Ollama client
client = Client()

# Iterate through each query and get a response from the LLM
for query in test_queries:
    # Make an API call to the Llama model
    # - Uses the 3B parameter version of Llama 3.2
    # - Includes a system message to specify plain text output
    # - Includes the user's query
    response: ChatResponse = client.chat(model='llama3.2:3b', messages=[
        {
            'role': 'system',
            'content': 'Respond to these queries in plain text format. Be descriptive wherever it makes sense.',
        },
        {
            'role': 'user',
            'content': query,
        },
    ])
    # Print both the original query and the model's response
    print(f"Query: {query}")
    print(f"Response: {response.message.content}", end="\n\n")
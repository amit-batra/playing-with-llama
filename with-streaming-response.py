# Import the chat function from ollama package
from ollama import chat, Client

# List of test questions to send to the LLM
test_queries = [
    "What is 2+2?",
    "What is the capital of France?",
    "Who wrote Romeo and Juliet?",
    "What is the speed of light?",
    "What is the largest planet?"
]

# Initialize the Ollama client
client = Client()

# Process each query with streaming responses
for query in test_queries:
    # Print the current query being processed
    print(f"Query: {query}")
    # Start the response line (end='' prevents newline)
    print("Response: ", end='')

    # Initialize streaming response from the LLM
    # stream=True enables chunk-by-chunk response delivery
    stream = client.chat(
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
        stream=True,
    )

    # Process and display each chunk of the response as it arrives
    # flush=True ensures immediate display of each character
    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)

    print("\n\n")
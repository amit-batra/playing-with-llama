# Playing with LLAMA
This project demonstrates different ways of interacting with the LLAMA 3.2 language model through the Ollama API. It includes several Python examples:

1. **Single Response Mode** (`with-single-response.py`):
   - Makes basic API calls to LLAMA 3.2 (3B parameter model)
   - Processes a set of test queries sequentially
   - Returns complete responses for each query

2. **Streaming Response Mode** (`with-streaming-response.py`):
   - Demonstrates real-time streaming of responses from LLAMA
   - Shows responses chunk-by-chunk as they are generated
   - Provides a more interactive experience

3. **Sequential vs Parallel Processing** (`sequential-vs-parallel.py`):
   - Implements both sequential and parallel query processing
   - Uses ThreadPoolExecutor and asyncio for concurrent API calls
   - Includes benchmarking to compare performance between methods
   - Features a `ParallelOllamaProcessor` class for efficient parallel processing

4. **Sequential vs Batched Processing** (`sequential-vs-batched.py`):
   - Compares sequential processing against batched query processing
   - Demonstrates how to efficiently batch multiple queries together
   - Includes performance metrics to analyze the benefits of batching

The examples showcase different approaches to working with LLAMA, from basic usage to advanced parallel and batch processing techniques. Each implementation includes proper error handling and follows Python best practices. The project serves as a practical demonstration of integrating large language models into Python applications using the Ollama client API.

# Running Locally
You need the following on your machine to run these examples locally:
1. Clone this repository: `git clone git@github.com:amit-batra/playing-with-llama.git`.
2. Install Ollama utility from https://ollama.com/.
3. Ensure that you have a functional Python 3 installation (we tested this app with Python 3.12). On macOS, you can use Homebrew to install Python 3.12 like so: `brew install python@3.12`.
4. Create a Python virtual environment and activate it with these commands:
   1. `cd playing-with-llama`
   2. `python3 -m venv .venv`
   3. `source .venv/bin/activate`
5. At this point, you should see the name of the virtual environment printed in brackets `(.venv)` before your actual command prompt.
6. Now install the required Python libraries inside your virtual environment with these commands:
   1. `pip install --upgrade pip`
   2. `pip install ollama`
7. Download the LLAMA 3.2 3B parameter model on your machine with this command: `ollama pull llama3.2:3b`. This model has a size of approximately 2GB, so it will take some time to download it. This step is required only once.
8. Launch each of the Python apps as follows:
   1. `python3 with-single-response.py`
   2. `python3 with-streaming-response.py`
   3. `python3 sequential-vs-parallel.py`
   4. `python3 sequential-vs-batched.py`
9. Deactivate your Python virtual environment using the command `deactivate`.
10. (Optional) Remove the LLAMA 3.2 3B parameter model from your machine with this command: `ollama delete llama3.2:3b`.
# Standard library imports
from concurrent.futures import ThreadPoolExecutor
import asyncio
import time
from typing import List, Tuple

# Third-party library imports
from ollama import Client

class ParallelOllamaProcessor:
	"""
	A class that handles parallel processing of queries to the Ollama API.
	Uses a combination of threading and async/await for efficient concurrent processing.
	"""
	def __init__(self, max_workers: int = 3):
		# Initialize Ollama client for API communication
		self.client = Client()
		# Maximum number of concurrent workers
		self.max_workers = max_workers
		# Create a thread pool for parallel execution
		self.executor = ThreadPoolExecutor(max_workers=max_workers)

	def process_single_query(self, query: str) -> str:
		"""
		Process a single query using the Ollama API.
		Args:
			query (str): The input query to process
		Returns:
			str: The model's response content
		"""
		return self.client.chat(
			model='llama3.2:3b',
			messages=[{'role': 'user', 'content': query}]
		)['message']['content']

	async def process_query_async(self, query: str) -> str:
		"""
		Asynchronously process a single query using the thread pool.
		This prevents blocking while waiting for API responses.
		"""
		loop = asyncio.get_running_loop()
		return await loop.run_in_executor(
			self.executor,
			self.process_single_query,
			query
		)

	async def process_queries_parallel(self, queries: List[str]) -> List[str]:
		"""
		Process multiple queries in parallel.
		Args:
			queries (List[str]): List of queries to process
		Returns:
			List[str]: List of responses in the same order as queries
		"""
		tasks = [self.process_query_async(query) for query in queries]
		return await asyncio.gather(*tasks)

# Comparison code for benchmarking
def run_sequential(queries: List[str]) -> Tuple[List[str], float]:
	"""
	Process queries sequentially (one after another) for comparison.
	Returns results and total processing time.
	"""
	client = Client()
	start_time = time.time()
	results = []

	for query in queries:
		response = client.chat(
			model='llama3.2:3b',
			messages=[{'role': 'user', 'content': query}]
		)
		results.append(response['message']['content'])

	end_time = time.time()
	return results, end_time - start_time

def run_parallel_processing(queries: List[str], max_workers: int = 3) -> Tuple[List[str], float]:
	"""
	Process queries in parallel using the ParallelOllamaProcessor.
	Returns results and total processing time.
	"""
	processor = ParallelOllamaProcessor(max_workers=max_workers)
	start_time = time.time()

	# Run the async processing
	results = asyncio.run(processor.process_queries_parallel(queries))

	end_time = time.time()
	return results, end_time - start_time

# Example usage and benchmarking
if __name__ == "__main__":
	# Test queries to compare sequential vs parallel processing
	test_queries = [
		"What is 2+2?",
		"What is the capital of France?",
		"Who wrote Romeo and Juliet?",
		"What is the speed of light?",
		"What is the largest planet?"
	]

	# Run and time sequential processing
	sequential_results, sequential_time = run_sequential(test_queries)
	# Run and time parallel processing
	parallel_results, parallel_time = run_parallel_processing(test_queries, max_workers=3)

	# Print timing comparison
	print(f"Sequential time: {sequential_time:.2f} seconds")
	print(f"Parallel time: {parallel_time:.2f} seconds")

	# Print detailed results from sequential processing
	print("\nSequential Results:")
	for query, response in zip(test_queries, sequential_results):
		print(f"\nQuery: {query}")
		print(f"Response: {response}")
import os
import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, BrowserConfig, CacheMode
from typing import Dict
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from dotenv import load_dotenv

load_dotenv()

async def extract_structured_data_using_llm(
    provider: str, api_token: str = None, extra_headers: Dict[str, str] = None
):
    print(f"\n--- Extracting Structured Data with {provider} ---")

    if api_token is None and provider != "ollama":
        print(f"API token is required for {provider}. Skipping this example.")
        return

    browser_config = BrowserConfig(headless=True)

    extra_args = {"temperature": 0, "top_p": 0.9, "max_tokens": 2000}
    if extra_headers:
        extra_args["extra_headers"] = extra_headers

    instruction = """
    You are a machine that extracts structured data from a website.
    Extract all the showcase site names and their target valid URLs.
    """

    crawler_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        word_count_threshold=1,
        page_timeout=80000,
        extraction_strategy=LLMExtractionStrategy(
            provider=provider,
            api_token=api_token,
            extraction_type="schema",
            schema= {
                "site_name": "string",
                "site_url": "string"
            },
            instruction=instruction,
            extra_args=extra_args,
        ),
    )
    
    crawler_config_no_extraction = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        word_count_threshold=1,
        page_timeout=80000,
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(
            url="https://erkabased.com/showcase", config=crawler_config
        )
        print(result.extracted_content)

if __name__ == "__main__":
    asyncio.run(
        extract_structured_data_using_llm(
            provider="openai/gpt-4o", api_token= os.getenv('OPENAI_API_KEY')
        )
    )
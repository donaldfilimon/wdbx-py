"""
Web scraper plugin for WDBX.

This plugin provides web scraping capabilities, allowing WDBX to extract
content from web pages and convert it to vector embeddings.
"""

import os
import logging
import asyncio
import re
import json
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from urllib.parse import urlparse, urljoin
import numpy as np

from ..plugins.base import WDBXPlugin, PluginError

logger = logging.getLogger(__name__)


class WebScraperPlugin(WDBXPlugin):
    """
    Web scraper plugin for WDBX.

    This plugin allows WDBX to extract content from web pages,
    respecting robots.txt and rate limits.

    Attributes:
        wdbx: Reference to the WDBX instance
        user_agent: User agent string for HTTP requests
        respect_robots_txt: Whether to respect robots.txt directives
        timeout: Request timeout in seconds
        max_depth: Maximum crawl depth for recursive scraping
        concurrency: Maximum number of concurrent requests
    """

    def __init__(self, wdbx):
        """
        Initialize the web scraper plugin.

        Args:
            wdbx: Reference to the WDBX instance
        """
        super().__init__(wdbx)

        # Load configuration
        self.user_agent = self.get_config("USER_AGENT", "WDBX WebScraper/0.1.0")
        self.respect_robots_txt = self.get_config("RESPECT_ROBOTS_TXT", True)
        self.timeout = float(self.get_config("TIMEOUT", 10.0))
        self.max_depth = int(self.get_config("MAX_DEPTH", 1))
        self.concurrency = int(self.get_config("CONCURRENCY", 5))

        # Initialize session
        self.session = None

        # Rate limiting
        self.rate_limit = float(
            self.get_config("RATE_LIMIT", 1.0)
        )  # requests per second
        # domain -> (last_request_time, rate_limit)
        self.domain_rate_limits = {}

        # Robots.txt cache
        self.robots_cache = {}  # domain -> (last_checked, can_fetch)

        # Embedding model
        self.embedding_model = None
        self.embedding_batch_size = int(self.get_config("EMBEDDING_BATCH_SIZE", 8))

        logger.info(f"Initialized WebScraperPlugin with user_agent={self.user_agent}")

    @property
    def name(self) -> str:
        """Return the name of the plugin."""
        return "webscraper"

    @property
    def description(self) -> str:
        """Return a description of the plugin."""
        return (
            "Web scraper plugin for WDBX, allowing content extraction from web pages."
        )

    @property
    def version(self) -> str:
        """Return the version of the plugin."""
        return "0.2.0"

    async def initialize(self) -> None:
        """Initialize the plugin."""
        try:
            # Import required packages
            import aiohttp
            from bs4 import BeautifulSoup
            import numpy as np

            # Create session
            self.session = aiohttp.ClientSession(
                headers={"User-Agent": self.user_agent},
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            )

            # Initialize embedding model if available
            try:
                # Try to get embedding model from another plugin
                for plugin_name in [
                    "openai",
                    "ollama",
                    "huggingface",
                    "sentencetransformers",
                ]:
                    if plugin_name in self.wdbx.plugins:
                        self.embedding_model = self.wdbx.plugins[plugin_name]
                        logger.info(f"Using {plugin_name} plugin for embeddings")
                        break

                # If no plugin is available, try to load a local model
                if not self.embedding_model:
                    try:
                        from sentence_transformers import SentenceTransformer

                        model_name = self.get_config(
                            "EMBEDDING_MODEL", "all-MiniLM-L6-v2"
                        )
                        self.embedding_model = SentenceTransformer(model_name)
                        logger.info(
                            f"Using local SentenceTransformer model: {model_name}"
                        )
                    except ImportError:
                        logger.warning(
                            "SentenceTransformer not available, embeddings will be delegated"
                        )
            except Exception as e:
                logger.warning(f"Error initializing embedding model: {e}")
                self.embedding_model = None

            logger.info(f"WebScraperPlugin initialized successfully")
        except ImportError as e:
            logger.error(f"Required package not installed: {e}")
            raise PluginError(f"Required package not installed: {e}")
        except Exception as e:
            logger.error(f"Error initializing WebScraperPlugin: {e}")
            raise PluginError(f"Error initializing WebScraperPlugin: {e}")

    async def shutdown(self) -> None:
        """Clean up resources when shutting down."""
        if self.session:
            await self.session.close()
            self.session = None
        logger.info("WebScraperPlugin shut down")

    async def extract_content(
        self, url: str, depth: int = 0, follow_links: bool = False
    ) -> str:
        """
        Extract content from a web page.

        Args:
            url: URL of the page to extract content from
            depth: Current crawl depth
            follow_links: Whether to follow links on the page

        Returns:
            Extracted text content

        Raises:
            PluginError: If content extraction fails
        """
        if not self.session:
            raise PluginError("WebScraperPlugin not initialized")

        try:
            # Parse URL
            parsed_url = urlparse(url)
            domain = parsed_url.netloc

            # Check robots.txt
            if self.respect_robots_txt and not await self._can_fetch(url):
                raise PluginError(f"Access to {url} disallowed by robots.txt")

            # Apply rate limiting
            await self._apply_rate_limit(domain)

            # Fetch page
            async with self.session.get(url) as response:
                if response.status != 200:
                    raise PluginError(
                        f"Failed to fetch {url}: {response.status} {response.reason}"
                    )

                # Get content type
                content_type = response.headers.get("Content-Type", "").lower()

                # Extract text based on content type
                if "text/html" in content_type:
                    # Parse HTML
                    html = await response.text()
                    content = await self._extract_html_content(html, url)

                    # Follow links if needed
                    if follow_links and depth < self.max_depth:
                        await self._follow_links(html, url, depth + 1)

                    return content
                elif "application/pdf" in content_type:
                    # Extract PDF content
                    pdf_content = await response.read()
                    return await self._extract_pdf_content(pdf_content)
                elif "text/" in content_type:
                    # Plain text
                    return await response.text()
                else:
                    raise PluginError(f"Unsupported content type: {content_type}")
        except PluginError:
            raise
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {e}")
            raise PluginError(f"Error extracting content from {url}: {e}")

    async def _extract_html_content(self, html: str, url: str) -> str:
        """
        Extract text content from HTML.

        Args:
            html: HTML content
            url: URL of the page

        Returns:
            Extracted text content
        """
        from bs4 import BeautifulSoup

        try:
            # Parse HTML
            soup = BeautifulSoup(html, "html.parser")

            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()

            # Get main content area, if any
            main_content = (
                soup.find("main")
                or soup.find("article")
                or soup.find("div", {"id": "content"})
            )

            if main_content:
                # Extract text from main content
                text = main_content.get_text(separator="\n", strip=True)
            else:
                # Extract text from body
                text = soup.body.get_text(separator="\n", strip=True)

            # Clean up whitespace
            text = re.sub(r"\s+", " ", text)
            text = re.sub(r"\n+", "\n", text)

            # Add metadata
            title = soup.title.string if soup.title else ""
            metadata = f"URL: {url}\nTitle: {title}\n\n"

            return metadata + text
        except Exception as e:
            logger.error(f"Error extracting HTML content: {e}")
            raise PluginError(f"Error extracting HTML content: {e}")

    async def _extract_pdf_content(self, pdf_content: bytes) -> str:
        """
        Extract text content from PDF.

        Args:
            pdf_content: PDF content as bytes

        Returns:
            Extracted text content

        Raises:
            PluginError: If PDF extraction fails or required package is not installed
        """
        try:
            import io
            from PyPDF2 import PdfReader

            # Parse PDF
            pdf = PdfReader(io.BytesIO(pdf_content))

            # Extract text from each page
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n\n"

            return text
        except ImportError:
            logger.warning("PyPDF2 not installed, PDF extraction not available")
            raise PluginError("PyPDF2 not installed, PDF extraction not available")
        except Exception as e:
            logger.error(f"Error extracting PDF content: {e}")
            raise PluginError(f"Error extracting PDF content: {e}")

    async def _follow_links(self, html: str, base_url: str, depth: int) -> None:
        """
        Follow links in HTML content.

        Args:
            html: HTML content
            base_url: Base URL for resolving relative links
            depth: Current crawl depth
        """
        from bs4 import BeautifulSoup

        try:
            # Parse HTML
            soup = BeautifulSoup(html, "html.parser")

            # Find all links
            links = []
            for link in soup.find_all("a", href=True):
                href = link["href"]

                # Resolve relative URLs
                if not urlparse(href).netloc:
                    href = urljoin(base_url, href)

                # Only follow links to the same domain
                if urlparse(href).netloc == urlparse(base_url).netloc:
                    links.append(href)

            # Limit number of links
            max_links = int(self.get_config("MAX_LINKS", 5))
            links = links[:max_links]

            # Follow links concurrently
            tasks = []
            for link in links:
                task = asyncio.create_task(
                    self.extract_content(link, depth, follow_links=False)
                )
                tasks.append(task)

            # Wait for all tasks to complete
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Error following links: {e}")

    async def _can_fetch(self, url: str) -> bool:
        """
        Check if a URL can be fetched according to robots.txt.

        Args:
            url: URL to check

        Returns:
            True if the URL can be fetched, False otherwise
        """
        try:
            import urllib.robotparser

            # Parse URL
            parsed_url = urlparse(url)
            domain = parsed_url.netloc

            # Check cache
            now = time.time()
            if domain in self.robots_cache:
                last_checked, robots_parser = self.robots_cache[domain]

                # Cache expires after 24 hours
                if now - last_checked < 86400:
                    return robots_parser.can_fetch(self.user_agent, url)

            # Create robots parser
            robots_parser = urllib.robotparser.RobotFileParser()
            robots_url = f"{parsed_url.scheme}://{domain}/robots.txt"

            # Fetch robots.txt
            try:
                async with self.session.get(robots_url, timeout=5) as response:
                    if response.status == 200:
                        robots_content = await response.text()

                        # Parse robots.txt
                        robots_parser.parse(robots_content.splitlines())
                    else:
                        # If robots.txt is not available, assume all URLs are allowed
                        robots_parser.allow_all = True
            except Exception:
                # If there's an error fetching robots.txt, assume all URLs are allowed
                robots_parser.allow_all = True

            # Update cache
            self.robots_cache[domain] = (now, robots_parser)

            # Check if the URL can be fetched
            return robots_parser.can_fetch(self.user_agent, url)
        except ImportError:
            # If robotparser is not available, assume all URLs are allowed
            logger.warning(
                "urllib.robotparser not available, robots.txt checking disabled"
            )
            return True
        except Exception as e:
            logger.error(f"Error checking robots.txt: {e}")
            # If there's an error checking robots.txt, assume all URLs are allowed
            return True

    async def _apply_rate_limit(self, domain: str) -> None:
        """
        Apply rate limiting for a domain.

        Args:
            domain: Domain to apply rate limiting for
        """
        now = time.time()

        # Get domain-specific rate limit
        if domain in self.domain_rate_limits:
            last_request, rate_limit = self.domain_rate_limits[domain]
        else:
            last_request = 0
            rate_limit = self.rate_limit

        # Calculate delay
        min_interval = 1.0 / rate_limit
        elapsed = now - last_request
        delay = max(0, min_interval - elapsed)

        # Wait if needed
        if delay > 0:
            await asyncio.sleep(delay)

        # Update last request time
        self.domain_rate_limits[domain] = (time.time(), rate_limit)

    async def create_embedding(self, text: str) -> List[float]:
        """
        Create an embedding vector for the given text.

        Args:
            text: The input text

        Returns:
            Embedding vector as a list of floats

        Raises:
            PluginError: If embedding creation fails
        """
        try:
            # If we have an embedding model plugin, use it
            if self.embedding_model and hasattr(
                self.embedding_model, "create_embedding"
            ):
                return await self.embedding_model.create_embedding(text)

            # If we have a local SentenceTransformer model, use it
            if self.embedding_model and hasattr(self.embedding_model, "encode"):
                embedding = self.embedding_model.encode(text)
                return embedding.tolist()

            # If no embedding model is available, delegate to WDBX
            logger.warning("No embedding model available, delegating to wdbx")
            raise PluginError("No embedding model available")
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            raise PluginError(f"Error creating embedding: {e}")

    async def crawl(self, start_url: str, max_pages: int = 10) -> Dict[str, str]:
        """
        Crawl a website starting from a URL.

        Args:
            start_url: URL to start crawling from
            max_pages: Maximum number of pages to crawl

        Returns:
            Dictionary mapping URLs to extracted content

        Raises:
            PluginError: If crawling fails
        """
        if not self.session:
            raise PluginError("WebScraperPlugin not initialized")

        try:
            # Parse start URL
            parsed_url = urlparse(start_url)
            domain = parsed_url.netloc

            # Initialize crawl state
            visited = set()
            queue = [start_url]
            results = {}

            # Create semaphore for concurrency control
            semaphore = asyncio.Semaphore(self.concurrency)

            async def _crawl_url(url):
                async with semaphore:
                    # Extract content
                    try:
                        content = await self.extract_content(url, follow_links=False)
                        results[url] = content

                        # Extract links
                        from bs4 import BeautifulSoup

                        soup = BeautifulSoup(content, "html.parser")

                        # Find all links
                        new_links = []
                        for link in soup.find_all("a", href=True):
                            href = link["href"]

                            # Resolve relative URLs
                            if not urlparse(href).netloc:
                                href = urljoin(url, href)

                            # Only follow links to the same domain
                            if urlparse(href).netloc == domain:
                                new_links.append(href)

                        return new_links
                    except Exception as e:
                        logger.error(f"Error crawling {url}: {e}")
                        return []

            # Crawl pages
            while queue and len(results) < max_pages:
                # Get next batch of URLs
                batch_size = min(self.concurrency, max_pages - len(results))
                batch = queue[:batch_size]
                queue = queue[batch_size:]

                # Crawl URLs in parallel
                tasks = []
                for url in batch:
                    if url not in visited:
                        visited.add(url)
                        tasks.append(_crawl_url(url))

                # Wait for all tasks to complete
                new_links_lists = await asyncio.gather(*tasks)

                # Add new links to queue
                for new_links in new_links_lists:
                    for link in new_links:
                        if link not in visited and link not in queue:
                            queue.append(link)

            return results
        except Exception as e:
            logger.error(f"Error crawling website: {e}")
            raise PluginError(f"Error crawling website: {e}")

    async def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for a batch of texts.

        Args:
            texts: List of texts to create embeddings for

        Returns:
            List of embedding vectors

        Raises:
            PluginError: If embedding creation fails
        """
        try:
            # If we have an embedding model plugin, use it
            if self.embedding_model and hasattr(
                self.embedding_model, "create_embeddings_batch"
            ):
                return await self.embedding_model.create_embeddings_batch(texts)

            # If we have a local SentenceTransformer model, use it
            if self.embedding_model and hasattr(self.embedding_model, "encode"):
                embeddings = self.embedding_model.encode(texts)
                return embeddings.tolist()

            # If no batch method is available, fall back to individual embedding
            embeddings = []
            for text in texts:
                embedding = await self.create_embedding(text)
                embeddings.append(embedding)

            return embeddings
        except Exception as e:
            logger.error(f"Error creating embeddings batch: {e}")
            raise PluginError(f"Error creating embeddings batch: {e}")

    async def store_webpage(
        self, url: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Extract content from a webpage and store it in the vector database.

        Args:
            url: URL of the webpage
            metadata: Optional additional metadata

        Returns:
            ID of the stored vector

        Raises:
            PluginError: If content extraction or storage fails
        """
        try:
            # Extract content
            content = await self.extract_content(url)

            # Create embedding
            embedding = await self.create_embedding(content)

            # Prepare metadata
            if metadata is None:
                metadata = {}

            metadata["url"] = url
            metadata["source"] = "webscraper"
            metadata["extraction_time"] = time.time()
            metadata["content"] = content

            # Store in vector database
            vector_id = await self.wdbx.vector_store_async(embedding, metadata)

            return vector_id
        except Exception as e:
            logger.error(f"Error storing webpage {url}: {e}")
            raise PluginError(f"Error storing webpage {url}: {e}")

    async def store_website(self, start_url: str, max_pages: int = 10) -> List[str]:
        """
        Crawl a website and store all pages in the vector database.

        Args:
            start_url: URL to start crawling from
            max_pages: Maximum number of pages to crawl

        Returns:
            List of vector IDs for stored pages

        Raises:
            PluginError: If crawling or storage fails
        """
        try:
            # Crawl website
            pages = await self.crawl(start_url, max_pages)

            # Store pages
            vector_ids = []
            for url, content in pages.items():
                # Create embedding
                embedding = await self.create_embedding(content)

                # Prepare metadata
                metadata = {
                    "url": url,
                    "source": "webscraper",
                    "extraction_time": time.time(),
                    "content": content,
                }

                # Store in vector database
                vector_id = await self.wdbx.vector_store_async(embedding, metadata)
                vector_ids.append(vector_id)

            return vector_ids
        except Exception as e:
            logger.error(f"Error storing website {start_url}: {e}")
            raise PluginError(f"Error storing website {start_url}: {e}")

    def register_commands(self) -> None:
        """Register commands with the WDBX CLI."""
        if hasattr(self.wdbx, "register_command"):
            self.wdbx.register_command(
                "scrape",
                self._cmd_scrape,
                "Extract content from a webpage",
                {
                    "--url": "URL to scrape",
                    "--follow": "Follow links on the page",
                    "--depth": "Maximum crawl depth",
                    "--store": "Store in vector database",
                },
            )

            self.wdbx.register_command(
                "crawl",
                self._cmd_crawl,
                "Crawl a website and store pages in the vector database",
                {
                    "--url": "URL to start crawling from",
                    "--max-pages": "Maximum number of pages to crawl",
                },
            )

    async def _cmd_scrape(self, args):
        """Command handler for the scrape command."""
        import argparse

        # Parse arguments
        parser = argparse.ArgumentParser(description="Extract content from a webpage")
        parser.add_argument("--url", required=True, help="URL to scrape")
        parser.add_argument(
            "--follow", action="store_true", help="Follow links on the page"
        )
        parser.add_argument("--depth", type=int, default=1, help="Maximum crawl depth")
        parser.add_argument(
            "--store", action="store_true", help="Store in vector database"
        )

        try:
            parsed_args = parser.parse_args(args.split())
        except SystemExit:
            return

        try:
            # Extract content
            content = await self.extract_content(
                parsed_args.url, depth=0, follow_links=parsed_args.follow
            )

            # Print content summary
            print(f"Extracted {len(content)} characters from {parsed_args.url}")
            print("Content snippet:")
            print(content[:500] + "..." if len(content) > 500 else content)

            # Store in vector database if requested
            if parsed_args.store:
                vector_id = await self.store_webpage(parsed_args.url)
                print(f"Stored in vector database with ID: {vector_id}")
        except Exception as e:
            print(f"Error: {e}")

    async def _cmd_crawl(self, args):
        """Command handler for the crawl command."""
        import argparse

        # Parse arguments
        parser = argparse.ArgumentParser(
            description="Crawl a website and store pages in the vector database"
        )
        parser.add_argument("--url", required=True, help="URL to start crawling from")
        parser.add_argument(
            "--max-pages", type=int, default=10, help="Maximum number of pages to crawl"
        )

        try:
            parsed_args = parser.parse_args(args.split())
        except SystemExit:
            return

        try:
            # Crawl website
            print(f"Crawling {parsed_args.url} (max {parsed_args.max_pages} pages)")
            vector_ids = await self.store_website(
                parsed_args.url, parsed_args.max_pages
            )

            # Print results
            print(f"Stored {len(vector_ids)} pages in vector database")
            print("Vector IDs:")
            for vector_id in vector_ids:
                print(f"  {vector_id}")
        except Exception as e:
            print(f"Error: {e}")


# Examples and guides for using the web scraper plugin


# Example 1: Extracting content from a single webpage
async def example_extract_content():
    wdbx_instance = WDBX(vector_dimension=384, enable_plugins=True)
    await wdbx_instance.initialize()

    webscraper = wdbx_instance.get_plugin("webscraper")
    if webscraper:
        url = "https://example.com"
        content = await webscraper.extract_content(url)
        print(f"Extracted content from {url}:\n{content}")

    await wdbx_instance.shutdown()


# Example 2: Storing a webpage in the vector database
async def example_store_webpage():
    wdbx_instance = WDBX(vector_dimension=384, enable_plugins=True)
    await wdbx_instance.initialize()

    webscraper = wdbx_instance.get_plugin("webscraper")
    if webscraper:
        url = "https://example.com"
        vector_id = await webscraper.store_webpage(url)
        print(f"Stored webpage {url} with vector ID: {vector_id}")

    await wdbx_instance.shutdown()


# Example 3: Crawling a website and storing pages in the vector database
async def example_store_website():
    wdbx_instance = WDBX(vector_dimension=384, enable_plugins=True)
    await wdbx_instance.initialize()

    webscraper = wdbx_instance.get_plugin("webscraper")
    if webscraper:
        start_url = "https://example.com"
        vector_ids = await webscraper.store_website(start_url, max_pages=5)
        print(f"Stored {len(vector_ids)} pages from {start_url} in vector database")

    await wdbx_instance.shutdown()

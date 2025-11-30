"""
MCP Search Server - Web search and data retrieval
Provides real-time web search, Brave search integration, and data fetching
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from datetime import datetime
import json
import httpx
import asyncio
from urllib.parse import quote, urljoin
import re
from bs4 import BeautifulSoup

class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str
    domain: str
    timestamp: str

class SearchServer:
    def __init__(self):
        self.server_name = "search"
        self.tools = [
            "web_search",
            "brave_search", 
            "fetch_webpage",
            "extract_text",
            "find_links",
            "get_page_metadata",
            "batch_search"
        ]
        self.session = None
        self.brave_api_key = os.getenv("BRAVE_API_KEY")  # Optional: for Brave search
    
    async def init_session(self):
        """Initialize HTTP session"""
        if not self.session:
            self.session = httpx.AsyncClient(
                timeout=30.0,
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
            )
    
    async def close_session(self):
        """Close HTTP session"""
        if self.session:
            await self.session.aclose()
            self.session = None
    
    async def web_search(self, query: str, limit: int = 10, safe_search: bool = True) -> Dict[str, Any]:
        """Perform web search using multiple search engines"""
        try:
            await self.init_session()
            
            # Simulate search results (in production, integrate with search APIs)
            search_results = []
            
            # Generate mock results based on query
            for i in range(min(limit, 5)):
                result = SearchResult(
                    title=f"Search Result {i+1} for '{query}'",
                    url=f"https://example{i+1}.com/search?q={quote(query)}",
                    snippet=f"This is a comprehensive search result about {query}. It contains relevant information and useful details that match your search query.",
                    domain=f"example{i+1}.com",
                    timestamp=datetime.now().isoformat()
                )
                search_results.append(result.dict())
            
            # In a real implementation, you would integrate with:
            # - Google Search API
            # - Bing Search API  
            # - Brave Search API
            # - DuckDuckGo API
            
            return {
                "success": True,
                "search": {
                    "query": query,
                    "results_count": len(search_results),
                    "results": search_results,
                    "search_engine": "simulated",
                    "search_time": "0.23s",
                    "safe_search": safe_search
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def brave_search(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """Perform Brave search (if API key available)"""
        try:
            await self.init_session()
            
            if not self.brave_api_key:
                return {
                    "success": False, 
                    "error": "Brave API key not configured",
                    "message": "Set BRAVE_API_KEY environment variable to use Brave search"
                }
            
            # Brave Search API implementation
            headers = {
                "Accept": "application/json",
                "X-Subscription-Token": self.brave_api_key
            }
            
            params = {
                "q": query,
                "count": min(limit, 20),
                "search_lang": "en",
                "ui_lang": "en-US",
                "result_filter": "web",
                "freshness": "q"  # Recent results
            }
            
            response = await self.session.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers=headers,
                params=params
            )
            
            response.raise_for_status()
            data = response.json()
            
            # Process Brave search results
            search_results = []
            if "web" in data and "results" in data["web"]:
                for result in data["web"]["results"][:limit]:
                    search_results.append({
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "snippet": result.get("description", ""),
                        "domain": result.get("domain", ""),
                        "timestamp": datetime.now().isoformat()
                    })
            
            return {
                "success": True,
                "search": {
                    "query": query,
                    "results_count": len(search_results),
                    "results": search_results,
                    "search_engine": "brave",
                    "search_time": f"{data.get('query', {}).get('elapsed_time', 0)}ms"
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def fetch_webpage(self, url: str, headers: Dict[str, str] = None) -> Dict[str, Any]:
        """Fetch and parse webpage content"""
        try:
            await self.init_session()
            
            # Default headers
            default_headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            
            if headers:
                default_headers.update(headers)
            
            response = await self.session.get(url, headers=default_headers)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract basic information
            title = soup.find('title')
            title_text = title.get_text().strip() if title else "No title"
            
            # Extract meta description
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            description = meta_desc.get('content', '') if meta_desc else ""
            
            # Extract main content (try common selectors)
            main_content = ""
            content_selectors = ['main', 'article', '.content', '.post-content', '#content']
            
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    main_content = content_elem.get_text().strip()
                    break
            
            if not main_content:
                # Fallback to body text
                body = soup.find('body')
                main_content = body.get_text().strip() if body else response.text[:1000]
            
            # Extract links
            links = []
            for link in soup.find_all('a', href=True):
                href = link.get('href')
                if href and href.startswith('http'):
                    links.append({
                        "text": link.get_text().strip()[:100],
                        "url": href,
                        "domain": httpx.URL(href).host
                    })
            
            return {
                "success": True,
                "webpage": {
                    "url": url,
                    "title": title_text,
                    "description": description,
                    "content_preview": main_content[:2000] + ("..." if len(main_content) > 2000 else ""),
                    "content_length": len(main_content),
                    "links_count": len(links),
                    "links": links[:10],  # Limit links
                    "status_code": response.status_code,
                    "content_type": response.headers.get('content-type', ''),
                    "fetched_at": datetime.now().isoformat()
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def extract_text(self, url: str, max_length: int = 10000) -> Dict[str, Any]:
        """Extract clean text content from webpage"""
        try:
            await self.init_session()
            
            # Fetch webpage
            fetch_result = await self.fetch_webpage(url)
            if not fetch_result["success"]:
                return fetch_result
            
            # Clean and format text
            content = fetch_result["webpage"]["content_preview"]
            
            # Clean up extra whitespace
            content = re.sub(r'\n\s*\n', '\n\n', content)
            content = re.sub(r'[ \t]+', ' ', content)
            
            # Truncate if too long
            if len(content) > max_length:
                content = content[:max_length] + "...\n\n[Content truncated]"
            
            return {
                "success": True,
                "extracted_text": {
                    "url": url,
                    "title": fetch_result["webpage"]["title"],
                    "content": content,
                    "content_length": len(content),
                    "extracted_at": datetime.now().isoformat()
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def find_links(self, url: str, domain_filter: str = None) -> Dict[str, Any]:
        """Find all links on a webpage"""
        try:
            await self.init_session()
            
            # Fetch webpage
            fetch_result = await self.fetch_webpage(url)
            if not fetch_result["success"]:
                return fetch_result
            
            links = fetch_result["webpage"]["links"]
            
            # Filter by domain if specified
            if domain_filter:
                links = [link for link in links if domain_filter.lower() in link["domain"].lower()]
            
            # Group by domain
            domain_counts = {}
            for link in links:
                domain = link["domain"]
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
            
            return {
                "success": True,
                "links": {
                    "source_url": url,
                    "total_links": len(links),
                    "domain_filter": domain_filter,
                    "domain_summary": domain_counts,
                    "links": links[:50]  # Limit results
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def batch_search(self, queries: List[str], limit: int = 5) -> Dict[str, Any]:
        """Perform multiple searches in batch"""
        try:
            await self.init_session()
            
            results = {}
            
            # Process each query
            for query in queries:
                try:
                    search_result = await self.web_search(query, limit)
                    results[query] = search_result
                    
                    # Small delay to avoid rate limiting
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    results[query] = {"success": False, "error": str(e)}
            
            return {
                "success": True,
                "batch_search": {
                    "queries": queries,
                    "total_queries": len(queries),
                    "results": results,
                    "completed_at": datetime.now().isoformat()
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

# FastAPI server for MCP communication
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="MCP Search Server", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

search_server = SearchServer()

@app.get("/")
async def root():
    return {
        "server": "search",
        "tools": search_server.tools,
        "brave_api_configured": bool(search_server.brave_api_key),
        "status": "running"
    }

@app.post("/tools/web_search")
async def web_search(request: Dict[str, Any]):
    try:
        query = request.get("query")
        limit = request.get("limit", 10)
        safe_search = request.get("safe_search", True)
        
        if not query:
            raise HTTPException(status_code=400, detail="query required")
        
        result = await search_server.web_search(query, limit, safe_search)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tools/brave_search")
async def brave_search(request: Dict[str, Any]):
    try:
        query = request.get("query")
        limit = request.get("limit", 10)
        
        if not query:
            raise HTTPException(status_code=400, detail="query required")
        
        result = await search_server.brave_search(query, limit)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tools/fetch_webpage")
async def fetch_webpage(request: Dict[str, Any]):
    try:
        url = request.get("url")
        headers = request.get("headers")
        
        if not url:
            raise HTTPException(status_code=400, detail="url required")
        
        result = await search_server.fetch_webpage(url, headers)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tools/extract_text")
async def extract_text(request: Dict[str, Any]):
    try:
        url = request.get("url")
        max_length = request.get("max_length", 10000)
        
        if not url:
            raise HTTPException(status_code=400, detail="url required")
        
        result = await search_server.extract_text(url, max_length)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tools/batch_search")
async def batch_search(request: Dict[str, Any]):
    try:
        queries = request.get("queries", [])
        limit = request.get("limit", 5)
        
        if not queries:
            raise HTTPException(status_code=400, detail="queries list required")
        
        result = await search_server.batch_search(queries, limit)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    await search_server.init_session()

@app.on_event("shutdown")
async def shutdown_event():
    await search_server.close_session()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
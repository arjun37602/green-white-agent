"""Utility functions for green-white-agent."""

import re
import httpx
import asyncio
import uuid
from typing import Dict

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    AgentCard,
    Part,
    TextPart,
    MessageSendParams,
    Message,
    Role,
    SendMessageRequest,
    SendMessageResponse,
)

# Global cache for agent cards - avoids fetching on every request
_agent_card_cache: Dict[str, AgentCard] = {}
_agent_card_lock: asyncio.Lock | None = None


def _get_agent_card_lock() -> asyncio.Lock:
    """Get or create the agent card lock (lazy initialization for async compatibility)."""
    global _agent_card_lock
    if _agent_card_lock is None:
        _agent_card_lock = asyncio.Lock()
    return _agent_card_lock


def parse_tags(str_with_tags: str) -> Dict[str, str]:
    """the target str contains tags in the format of <tag_name> ... </tag_name>, parse them out and return a dict"""
    tags = re.findall(r"<(.*?)>(.*?)</\1>", str_with_tags, re.DOTALL)
    return {tag: content.strip() for tag, content in tags}


def parse_answer(text: str) -> str:
    """Parse <json>...</json> tags specifically from assistant answer"""
    json_match = re.search(r"<json>(.*?)</json>", text, re.DOTALL)
    if json_match:
        return json_match.group(1).strip()
    return ""


async def get_agent_card(url: str, httpx_client: httpx.AsyncClient | None = None, use_cache: bool = True) -> AgentCard | None:
    """Get agent card, using cache to avoid repeated network requests."""
    global _agent_card_cache
    
    # Check cache first (outside lock for fast path)
    if use_cache and url in _agent_card_cache:
        return _agent_card_cache[url]
    
    # Use lock to prevent multiple simultaneous fetches for the same URL
    async with _get_agent_card_lock():
        # Double-check after acquiring lock
        if use_cache and url in _agent_card_cache:
            return _agent_card_cache[url]
        
        # Fetch the card
        if httpx_client is None:
            async with httpx.AsyncClient() as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=url)
                card: AgentCard | None = await resolver.get_agent_card()
        else:
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url=url)
            card: AgentCard | None = await resolver.get_agent_card()
        
        # Cache the result
        if card is not None and use_cache:
            _agent_card_cache[url] = card
        
        return card


def clear_agent_card_cache():
    """Clear the agent card cache (useful for testing or reconnection)."""
    global _agent_card_cache, _agent_card_lock
    _agent_card_cache = {}
    _agent_card_lock = None  # Reset lock for new event loop


async def wait_agent_ready(url, timeout=10):
    # wait until the A2A server is ready, check by getting the agent card
    retry_cnt = 0
    while retry_cnt < timeout:
        retry_cnt += 1
        try:
            card = await get_agent_card(url)
            if card is not None:
                return True
            else:
                print(
                    f"Agent card not available yet..., retrying {retry_cnt}/{timeout}"
                )
        except Exception:
            pass
        await asyncio.sleep(1)
    return False


import logging
import traceback

_send_message_logger = logging.getLogger("utils.send_message")


async def send_message(
    url, message, task_id=None, context_id=None, timeout=600.0, httpx_client=None
) -> SendMessageResponse:
    """Send a message to an A2A agent with comprehensive error logging."""
    import time
    start_time = time.time()
    
    try:
        # Pass shared client to get_agent_card to avoid creating extra connections
        card = await get_agent_card(url, httpx_client=httpx_client)
        
        if card is None:
            _send_message_logger.error(f"🔴 Failed to get agent card from {url}")
            raise Exception(f"Could not fetch agent card from {url}")
        
        if httpx_client is None:
            # Create new client with proper cleanup if none provided
            async with httpx.AsyncClient(timeout=timeout) as httpx_client:
                client = A2AClient(httpx_client=httpx_client, agent_card=card)
                
                message_id = uuid.uuid4().hex
                params = MessageSendParams(
                    message=Message(
                        role=Role.user,
                        parts=[Part(TextPart(text=message))],
                        message_id=message_id,
                        task_id=task_id,
                        context_id=context_id,
                    )
                )
                request_id = uuid.uuid4().hex
                req = SendMessageRequest(id=request_id, params=params)
                
                _send_message_logger.debug(f"Sending message to {url} (context={context_id}, timeout={timeout})")
                response = await client.send_message(request=req)
                
                elapsed = time.time() - start_time
                _send_message_logger.debug(f"Response received from {url} in {elapsed:.2f}s")
                return response
        else:
            # Use provided shared client (don't modify its timeout - use as-is)
            client = A2AClient(httpx_client=httpx_client, agent_card=card)
            
            message_id = uuid.uuid4().hex
            params = MessageSendParams(
                message=Message(
                    role=Role.user,
                    parts=[Part(TextPart(text=message))],
                    message_id=message_id,
                    task_id=task_id,
                    context_id=context_id,
                )
            )
            request_id = uuid.uuid4().hex
            req = SendMessageRequest(id=request_id, params=params)
            
            _send_message_logger.debug(f"Sending message to {url} (context={context_id}, using shared client)")
            response = await client.send_message(request=req)
            
            elapsed = time.time() - start_time
            _send_message_logger.debug(f"Response received from {url} in {elapsed:.2f}s")
            return response
            
    except httpx.HTTPStatusError as e:
        elapsed = time.time() - start_time
        _send_message_logger.error(f"🔴 HTTP error from {url} after {elapsed:.2f}s")
        _send_message_logger.error(f"🔴 Status code: {e.response.status_code}")
        _send_message_logger.error(f"🔴 Response body: {e.response.text[:500] if e.response.text else '(empty)'}")
        raise Exception(f"HTTP Error {e.response.status_code}: {e.response.text[:200]}")
        
    except httpx.ConnectTimeout as e:
        elapsed = time.time() - start_time
        _send_message_logger.error(f"🔴 CONNECT TIMEOUT to {url} after {elapsed:.2f}s")
        _send_message_logger.error(f"🔴 This usually means the server's accept queue is full or server is temporarily unavailable")
        _send_message_logger.error(f"🔴 This often happens during batch transitions when many requests complete simultaneously")
        _send_message_logger.error(f"🔴 Error: {e}")
        raise Exception(f"Connect Timeout Error: Could not connect to {url} within timeout - server may be overloaded")
        
    except httpx.ConnectError as e:
        elapsed = time.time() - start_time
        _send_message_logger.error(f"🔴 CONNECTION ERROR to {url} after {elapsed:.2f}s")
        _send_message_logger.error(f"🔴 Error: {e}")
        raise Exception(f"Connection Error: Could not connect to {url}: {e}")
        
    except httpx.WriteTimeout as e:
        elapsed = time.time() - start_time
        _send_message_logger.error(f"🔴 WRITE TIMEOUT to {url} after {elapsed:.2f}s")
        _send_message_logger.error(f"🔴 This usually means the server is too busy to read the request")
        _send_message_logger.error(f"🔴 Consider reducing parallelism or increasing write timeout")
        _send_message_logger.error(f"🔴 Error: {e}")
        raise Exception(f"Write Timeout Error: Server at {url} too busy to accept request after {elapsed:.2f}s")
        
    except httpx.ReadTimeout as e:
        elapsed = time.time() - start_time
        _send_message_logger.error(f"🔴 READ TIMEOUT from {url} after {elapsed:.2f}s")
        _send_message_logger.error(f"🔴 This usually means the LLM call is taking too long")
        _send_message_logger.error(f"🔴 Error: {e}")
        raise Exception(f"Read Timeout Error: Request to {url} timed out waiting for response")
        
    except httpx.TimeoutException as e:
        elapsed = time.time() - start_time
        timeout_type = type(e).__name__
        _send_message_logger.error(f"🔴 Timeout ({timeout_type}) from {url} after {elapsed:.2f}s")
        _send_message_logger.error(f"🔴 Error: {e}")
        raise Exception(f"Timeout Error ({timeout_type}): Request to {url} timed out after {elapsed:.2f}s")
        
    except Exception as e:
        elapsed = time.time() - start_time
        error_type = type(e).__name__
        error_msg = str(e)
        
        _send_message_logger.error(f"🔴 Error sending message to {url} after {elapsed:.2f}s")
        _send_message_logger.error(f"🔴 Error type: {error_type}")
        _send_message_logger.error(f"🔴 Error message: {error_msg}")
        
        # Check for A2A client errors that wrap underlying issues
        if "A2AClient" in error_type:
            _send_message_logger.error(f"🔴 A2A Client Error - check underlying network/server issues")
            if "503" in error_msg:
                _send_message_logger.error(f"🔴 503 errors from A2A usually indicate WriteTimeout or server overload")
        
        _send_message_logger.error(f"🔴 Full traceback:\n{traceback.format_exc()}")
        
        # Re-raise with more context
        raise Exception(f"Network communication error: {e}")


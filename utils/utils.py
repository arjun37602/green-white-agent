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

# Cache agent cards to avoid refetching on every parallel request
_agent_card_cache: Dict[str, AgentCard] = {}


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


async def get_agent_card(url: str, use_cache: bool = True) -> AgentCard | None:
    """
    Get agent card, with optional caching to prevent excessive requests.
    
    Args:
        url: The base URL of the agent
        use_cache: If True, use cached card if available (default: True)
    
    Returns:
        AgentCard or None
    """
    if use_cache and url in _agent_card_cache:
        return _agent_card_cache[url]
    
    # Use context manager to ensure client is properly closed
    async with httpx.AsyncClient() as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=url)
        card: AgentCard | None = await resolver.get_agent_card()
    
    if card is not None and use_cache:
        _agent_card_cache[url] = card
    
    return card


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


async def send_message(
    url, message, task_id=None, context_id=None, use_cached_card=True, agent_card=None
) -> SendMessageResponse:
    """
    Send a message to an A2A agent.
    
    Args:
        url: The base URL of the agent
        message: The message text to send
        task_id: Optional task ID
        context_id: Optional context ID
        use_cached_card: If True, use cached agent card (default: True for parallel execution)
        agent_card: Optional pre-fetched AgentCard to avoid fetching (recommended for parallel execution)
    
    Returns:
        SendMessageResponse
    """
    # Use provided agent_card if available, otherwise fetch it
    if agent_card is None:
        card = await get_agent_card(url, use_cache=use_cached_card)
    else:
        card = agent_card
    
    # Use context manager to ensure client is properly closed
    async with httpx.AsyncClient(timeout=600.0) as httpx_client:
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
        response = await client.send_message(request=req)
    
    return response


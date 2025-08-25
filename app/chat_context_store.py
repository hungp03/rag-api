import os
import json
import redis.asyncio as redis
from typing import List, Dict, Any


class ChatContextStore:
    _instance = None  # giữ singleton instance

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ChatContextStore, cls).__new__(cls)
        return cls._instance

    def __init__(self,
                 redis_url: str = None,
                 max_messages: int = 10,
                 ttl_seconds: int = 3600):
        # tránh re-init nhiều lần
        if hasattr(self, "_initialized") and self._initialized:
            return

        self.redis_url = redis_url or os.getenv("REDIS_URL")
        self.max_messages = int(os.getenv("CHAT_HISTORY_MAX_MESSAGES", max_messages))
        self.ttl_seconds = int(os.getenv("CHAT_HISTORY_TTL_SECONDS", ttl_seconds))
        self.redis: redis.Redis | None = None
        self._initialized = True

    async def init(self):
        if self.redis is None:
            self.redis = redis.from_url(self.redis_url, decode_responses=True)

    async def keep_alive(self) -> bool:
        await self.init()
        try:
            await self.redis.set("keepalive", "1", ex=1000)
            return True
        except Exception as e:
            print(f"Redis keep_alive failed: {e}")
            return False

    def key(self, session_id: str) -> str:
        return f"chat:ctx:{session_id}"

    async def append(self, session_id: str, message: Dict[str, Any]):
        """ 
        Save the latest message to the list, trimming to keep max_messages, and set TTL.
        message format: {"role": "user"/"model", "parts": [{"text": "..."}]} 
        """
        await self.init()
        k = self.key(session_id)

        await self.redis.lpush(k, json.dumps(message))
        await self.redis.ltrim(k, 0, self.max_messages - 1)
        await self.redis.expire(k, self.ttl_seconds)

    async def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        await self.init()
        k = self.key(session_id)
        raw = await self.redis.lrange(k, 0, self.max_messages - 1)

        if not raw:
            return []

        messages = [json.loads(s) for s in raw]
        return list(reversed(messages))

    async def clear(self, session_id: str):
        await self.init()
        await self.redis.delete(self.key(session_id))

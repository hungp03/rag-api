from typing import Optional
from pydantic import BaseModel


class ChatResponse(BaseModel):
    session_id: str
    response: str
    message: Optional[str] = None
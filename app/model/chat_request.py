from pydantic import BaseModel, Field
from typing import Optional


class ChatRequest(BaseModel):
    session_id: str = Field(None, description="Session ID for chat history")
    user_message: str = Field(..., description="User's message/question")
    use_rag: Optional[bool] = Field(False, description="Whether to use RAG pipeline")
    use_stream: Optional[bool] = Field(True, description="Whether to stream response")
    use_advanced_rag: Optional[bool] = Field(False, description="Whether to use advanced RAG with filtering")
    top_k: Optional[int] = Field(3, ge=1, le=10, description="Number of context results (1-10)")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=1.0, description="Model creativity (0-1)")
    max_output_tokens: Optional[int] = Field(2048, ge=100, le=4096, description="Max response length")
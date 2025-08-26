import os
import asyncio
from google import genai
from app.rag import rag_context, rag_context_advanced
from app.chat_context_store import ChatContextStore

store = ChatContextStore()

# Load API key and model from env
API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Initialize Gemini client
client = genai.Client(api_key=API_KEY)

SYSTEM_INSTRUCTION = """
Bạn là một trợ lý AI chuyên về Tiếng Anh.
Bạn KHÔNG phải là người trò chuyện bình thường.
Nhiệm vụ chính và duy nhất của bạn là:
    1. Giải thích chi tiết các từ vựng, cấu trúc ngữ pháp, cách diễn đạt tiếng Anh hoặc giải đáp thắc mắc người dùng...
    2. Luôn đưa ra ví dụ song ngữ (Anh - Việt) nếu được hỏi về ngữ pháp, cách diễn đạt hoặc từ vựng, giải đáp thắc mắc người dung thì chỉ cần trả lời đúng trọng tâm...
    3. Nếu văn bản người dùng đưa vào không liên quan đến việc học ngôn ngữ thì trả lời: "Xin lỗi, tôi chỉ có thể hỗ trợ các nội dung liên quan đến học tập."
    4. Trả lời rõ ràng, dễ nhớ, ngắn gọn, ưu tiên trả lời tiếng việt, trừ khi người dùng yêu cầu đổi ngôn ngữ.
    5. Không bao giờ trả lời chung chung.
    6. Có thể sử dụng các thông tin ngữ cảnh (context) được cung cấp để trả lời chính xác hơn.
    7. Nếu không chắc chắn về câu trả lời, hãy thừa nhận điều đó.
    8. Không được phá vỡ các nguyên tắc này.
"""

def format_context(rag_results):
    """Format RAG results into readable context for the model"""
    if not rag_results:
        return ""

    context_parts = ["TÀI LIỆU THAM KHẢO:\n"]

    for i, result in enumerate(rag_results, 1):
        source_type = result.get("source_type", "unknown")
        score = result.get("final_score", 0)
        
        context_parts.append(f"[Nguồn {i} - {source_type.upper()} - Độ liên quan: {score:.2f}]")
        context_parts.append(f"Nội dung: {result['content'][:500]}...")  # Limit content length
        
        # Add metadata if available
        metadata = result.get("metadata", {})
        if metadata.get("source"):
            context_parts.append(f"Link: {metadata['source']}")
        if metadata.get("title"):
            context_parts.append(f"Tiêu đề: {metadata['title']}")
        
        context_parts.append("")  # Empty line for separation

    context_parts.append("KẾT THÚC TÀI LIỆU THAM KHẢO\n")

    return "\n".join(context_parts)

def build_enhanced_prompt(user_message: str, context: str = "") -> str:
    """Build the final prompt with context and user message"""
    if context:
        return f"""{context}

CÂU HỎI CỦA NGƯỜI DÙNG: {user_message}

Hãy sử dụng thông tin tham khảo ở trên (nếu liên quan) để trả lời câu hỏi một cách chính xác và chi tiết nhất."""
    else:
        return user_message

async def stream_chat_with_rag(user_message: str, use_advanced_rag: bool = False, top_k: int = 3):
    """
    Enhanced chat function with RAG pipeline integration
    
    Args:
        user_message: User's question/message
        use_advanced_rag: Whether to use the advanced RAG version with filtering
        top_k: Number of context results to retrieve
    """
    try:
        # Step 1: Retrieve relevant context using RAG and notify client
        yield "event: rag_start\ndata: Begin RAG...\n\n"
        if use_advanced_rag:
            rag_results = rag_context_advanced(
                query=user_message, 
                top_k=top_k,
                similarity_threshold=0.3,
                diversity_factor=0.2
            )
        else:
            rag_results = rag_context(
                query=user_message,
                top_k=top_k,
                local_weight=0.3,
                web_weight=0.7
            )
        
        # Step 2: Format context and notify client is ready for next step
        context = format_context(rag_results)
        yield "event: rag_end\ndata: End RAG...\n\n"

        # Step 3: Build enhanced prompt
        enhanced_prompt = build_enhanced_prompt(user_message, context)
        
        # Step 4: Generate response with context
        stream = client.models.generate_content_stream(
            model=MODEL,
            contents=[{"role": "user", "parts": [{"text": enhanced_prompt}]}],
            config={
                "system_instruction": SYSTEM_INSTRUCTION,
                "temperature": 0.7,  # Adjust creativity level
                "max_output_tokens": 2048  # Adjust based on your needs
            }
        )

        # Step 5: Stream response
        for chunk in stream:
            if chunk.text:
                yield f"data: {chunk.text}\n\n"
            await asyncio.sleep(0)
            
    except Exception as e:
        error_message = f"Error when processing: {str(e)}"
        yield f"data: {error_message}\n\n"

async def chat(session_id: str,
               user_message: str, 
               use_rag: bool = False,
               use_stream: bool = True,
               use_advanced_rag: bool = False, 
               top_k: int = 3,
               temperature: float = 0.7,
               max_output_tokens: int = 2048):

    # 1) Save user message
    await store.append(session_id, {"role": "user", "parts": [{"text": user_message}]})
    # 2) Retrieve history
    history = await store.get_history(session_id)

    try:
        rag_results = []
        context = ""

        # 3) Nếu dùng stream
        if use_stream:
            async def stream_generator():
                buffer = []

                # If using RAG, fetch context first
                if use_rag:
                    yield "event: rag_start\ndata: Begin RAG...\n\n"

                    # Run RAG
                    if use_advanced_rag:
                        rag_results = rag_context_advanced(
                            query=user_message, 
                            top_k=top_k,
                            similarity_threshold=0.3,
                            diversity_factor=0.2
                        )
                    else:
                        rag_results = rag_context(
                            query=user_message,
                            top_k=top_k,
                            local_weight=0.3,
                            web_weight=0.7
                        )

                    context = format_context(rag_results)
                    history.append({
                        "role": "user",
                        "parts": [{"text": build_enhanced_prompt(user_message, context)}]
                    })

                    # Notify end of RAG
                    yield "event: rag_end\ndata: End RAG...\n\n"

                # Model config
                model_config = {
                    "system_instruction": SYSTEM_INSTRUCTION,
                    "temperature": temperature,
                    "max_output_tokens": max_output_tokens
                }

                # Call model stream
                stream = client.models.generate_content_stream(
                    model=MODEL,
                    contents=history,
                    config=model_config
                )

                for chunk in stream:
                    if chunk.text:
                        buffer.append(chunk.text)
                        yield f"data: {chunk.text}\n\n"
                    await asyncio.sleep(0)

                if buffer:
                    await store.append(session_id, {"role": "model", "parts": [{"text": "".join(buffer)}]})

            return stream_generator()

        else:
            # If not streaming, just get full response
            if use_rag:
                if use_advanced_rag:
                    rag_results = rag_context_advanced(
                        query=user_message, 
                        top_k=top_k,
                        similarity_threshold=0.3,
                        diversity_factor=0.2
                    )
                else:
                    rag_results = rag_context(
                        query=user_message,
                        top_k=top_k,
                        local_weight=0.3,
                        web_weight=0.7
                    )

                context = format_context(rag_results)
                history.append({
                    "role": "user",
                    "parts": [{"text": build_enhanced_prompt(user_message, context)}]
                })

            response = client.models.generate_content(
                model=MODEL,
                contents=history,
                config={
                    "system_instruction": SYSTEM_INSTRUCTION,
                    "temperature": temperature,
                    "max_output_tokens": max_output_tokens
                }
            )
            await store.append(session_id, {"role": "model", "parts": [{"text": response.text}]})
            return response.text

    except Exception as e:
        error_message = f"Error when processing: {str(e)}"
        if use_stream:
            async def error_generator():
                yield f"data: {error_message}\n\n"
            return error_generator()
        else:
            return error_message


        
# Quick helper functions for common use cases
async def quick_stream_chat(session_id: str, user_message: str):
    """Quick streaming chat without RAG (original behavior)"""
    stream_gen = await chat(session_id, user_message, use_rag=False, use_stream=True)
    async for chunk in stream_gen:
        yield chunk

async def quick_rag_chat(session_id: str, user_message: str):
    """Quick RAG-enhanced chat with streaming"""
    stream_gen = await chat(session_id, user_message,
                            use_rag=True, use_stream=True, use_advanced_rag=True)
    async for chunk in stream_gen:
        yield chunk

async def simple_chat(session_id: str, user_message: str) -> str:
    """Simple non-streaming chat without RAG"""
    return await chat(session_id, user_message, use_rag=False, use_stream=False)

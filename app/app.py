from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Depends
from pydantic import BaseModel
from app.rag import rag_context_advanced
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.config import vectorstore
from app.chat_context_store import ChatContextStore
import os
from dotenv import load_dotenv
import uuid
from fastapi.responses import StreamingResponse
from app.gemini_service import chat
from app.model.chat_request import ChatRequest
from app.model.chat_response import ChatResponse

load_dotenv()

app = FastAPI()
store = ChatContextStore()

API_KEY = os.getenv("RAG_API_KEY")
MAX_FILE_SIZE = 2 * 1024 * 1024  # 2 MB limit

# ===== Middleware to check API key =====
def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")

# ===== RAG SEARCH =====
class QueryRequest(BaseModel):
    query: str

@app.get("/health")
async def health_check():
    await store.keep_alive()
    return {"status": "ok", "requested_at": os.times(), "request_id": uuid.uuid4().hex}

@app.post("/rag", dependencies=[Depends(verify_api_key)])
async def rag_endpoint(req: QueryRequest):
    """Search top-k contexts from vectorstore based on query"""
    return {"contexts": rag_context_advanced(req.query)}

# ===== INGEST / UPLOAD =====
@app.post("/upload", dependencies=[Depends(verify_api_key)])
async def upload_file(file: UploadFile = File(...)):
    """Upload a document (txt, pdf, docx), chunk, embed and store in Qdrant"""

    # Check file size (< 2 MB)
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large (max 2MB)")

    # Check extension
    suffix = file.filename.split(".")[-1].lower()
    if suffix not in ["txt", "pdf", "docx"]:
        raise HTTPException(status_code=400, detail="Only .txt, .pdf, .docx files are allowed")

    # Save file into ./docs folder (same level as app.py)
    docs_dir = os.path.join(os.path.dirname(__file__), "docs")
    os.makedirs(docs_dir, exist_ok=True)
    saved_path = os.path.join(docs_dir, file.filename)

    with open(saved_path, "wb") as f:
        f.write(contents)

    # Choose loader based on extension
    if suffix == "txt":
        loader = TextLoader(saved_path, encoding="utf-8")
    elif suffix == "pdf":
        loader = PyPDFLoader(saved_path)
    elif suffix == "docx":
        loader = UnstructuredWordDocumentLoader(saved_path)

    # Load document
    docs = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # Upsert into Qdrant vectorstore
    vectorstore.add_documents(chunks)

    return {
        "filename": file.filename,
        "saved_path": saved_path,
        "chunks_ingested": len(chunks)
    }


@app.post("/chat", dependencies=[Depends(verify_api_key)])
async def chat_endpoint(request: ChatRequest):
    """
    Unified chat endpoint supporting all 4 modes:
    1. Stream + RAG
    2. No Stream + RAG  
    3. Stream + No RAG
    4. No Stream + No RAG
    
    Body parameters:
    - session_id: Session ID for chat history
    - user_message: Required. The user's question/message
    - use_rag: Optional (default: false). Enable RAG pipeline
    - use_stream: Optional (default: true). Enable streaming response
    - use_advanced_rag: Optional (default: false). Use advanced RAG with filtering
    - top_k: Optional (default: 3). Number of context results to retrieve
    - temperature: Optional (default: 0.7). Model creativity level
    - max_output_tokens: Optional (default: 2048). Maximum response length
    
    Returns:
    - If use_stream=true: Server-sent events stream
    - If use_stream=false: JSON with complete response
    """
    if not request.user_message.strip():
        raise HTTPException(status_code=400, detail="user_message cannot be empty")

    session_id = request.session_id or str(uuid.uuid4())

    try:
        # Call the unified chat function
        result = await chat(
            session_id=session_id,
            user_message=request.user_message,
            use_rag=request.use_rag,
            use_stream=request.use_stream,
            use_advanced_rag=request.use_advanced_rag,
            top_k=request.top_k,
            temperature=request.temperature,
            max_output_tokens=request.max_output_tokens
        )
        
        if request.use_stream:
            # Return streaming response
            return StreamingResponse(
                result,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                    "X-Session-ID": session_id,
                }
            )
        else:
            # Return complete response as JSON
            return ChatResponse(
                session_id=session_id,          
                response=result
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

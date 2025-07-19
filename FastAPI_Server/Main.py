import os
import uuid
import tempfile
import shutil
from contextlib import asynccontextmanager
from typing import Any, List

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from dotenv import load_dotenv

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.pregel import Pregel
from mem0 import MemoryClient

from Graph import BuildGraph, GraphState

load_dotenv()

SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "checkpoints.sqlite")

sqlite_checkpointer: AsyncSqliteSaver | None = None
graph: Pregel | None = None
memory_client: MemoryClient | None = None
memory_manager = None  # Add this global

class ConversationMemoryManager:
    def __init__(self, memory_client: MemoryClient):
        self.memory_client = memory_client
        self.conversation_messages = {}
    
    def get_conversation_key(self, user_id: str, session_id: str) -> str:
        return f"{user_id}_{session_id}"
    
    async def load_conversation_context(self, user_id: str, session_id: str) -> dict:
        try:
            key = self.get_conversation_key(user_id, session_id)
            
            all_memories = self.memory_client.get_all(user_id=user_id)
            
            if not all_memories:
                self.conversation_messages[key] = []
                return {"past_memory": "This is a fresh conversation", "messages": []}
            
            session_memories = [
                m for m in all_memories 
                if m.get("metadata", {}).get("session_id") == session_id
            ]
            
            summaries = []
            for mem in session_memories:
                summaries.append(mem.get("memory", ""))
            
            if key not in self.conversation_messages:
                self.conversation_messages[key] = []
            
            combined_context = "\n\n".join([
                "Summarized memory:\n" + "\n".join(summaries)
            ]) if summaries else "This is a fresh conversation"
            
            return {"past_memory": combined_context, "messages": self.conversation_messages[key]}
            
        except Exception as e:
            print(f"Error loading conversation context: {e}")
            key = self.get_conversation_key(user_id, session_id)
            self.conversation_messages[key] = []
            return {"past_memory": "Error loading conversation context", "messages": []}
    
    async def save_conversation_turn(self, user_id: str, session_id: str, 
                                   user_message: str, ai_response: str):
        try:
            conversation_data = [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": ai_response}
            ]
            
            result = self.memory_client.add(
                conversation_data,
                user_id=user_id,
                metadata={"session_id": session_id}
            )
            print(f"Conversation saved to Mem0: {result}")
            
        except Exception as e:
            print(f"Error saving conversation: {e}")
    
    def add_to_growing_conversation(self, user_id: str, session_id: str, 
                                   user_message: str, ai_response: str):
        key = self.get_conversation_key(user_id, session_id)
        
        if key not in self.conversation_messages:
            self.conversation_messages[key] = []
        
        self.conversation_messages[key].extend([
            HumanMessage(content=user_message),
            AIMessage(content=ai_response)
        ])
        
        if len(self.conversation_messages[key]) > 10:
            self.conversation_messages[key] = self.conversation_messages[key][-10:]
    
    def get_current_messages(self, user_id: str, session_id: str) -> List[BaseMessage]:
        key = self.get_conversation_key(user_id, session_id)
        return self.conversation_messages.get(key, [])

@asynccontextmanager
async def lifespan(app: FastAPI):
    global sqlite_checkpointer, graph, memory_client, memory_manager
    
    checkpointer_cm = AsyncSqliteSaver.from_conn_string(SQLITE_DB_PATH)
    try:
        sqlite_checkpointer = await checkpointer_cm.__aenter__()
        graph = BuildGraph(sqlite_checkpointer)
        memory_client = MemoryClient()
        memory_manager = ConversationMemoryManager(memory_client)  # Initialize here
        os.makedirs("uploads", exist_ok=True)
        yield
    finally:
        if sqlite_checkpointer:
            await checkpointer_cm.__aexit__(None, None, None)

app = FastAPI(lifespan=lifespan)

# Remove the @app.on_event("startup") section entirely

class MessageRequest(BaseModel):
    user_id: str
    session_id: str
    message: str

def generate_thread_id(user_id: str, session_id: str) -> str:
    return f"{user_id}-{session_id}"

@app.post("/invoke")
async def invoke_agent(request: MessageRequest):
    thread_id = generate_thread_id(request.user_id, request.session_id)
    config = {"configurable": {"thread_id": thread_id}}
    
    context = await memory_manager.load_conversation_context(
        request.user_id, request.session_id
    )
    
    current_messages = memory_manager.get_current_messages(
        request.user_id, request.session_id
    )
    
    initial_state: GraphState = {
        "input": request.message,
        "user_id": request.user_id,
        "session_id": request.session_id,
        "messages": current_messages,
        "past_memory": context["past_memory"]
    }

    try:
        final_state = None
        async for event in graph.astream(initial_state, config=config):
            if "Aggregator" in event:
                final_state = event["Aggregator"]

        if not final_state or "final_response" not in final_state:
            raise HTTPException(status_code=500, detail="Graph did not produce a final response.")

        ai_response = final_state["final_response"]
        
        memory_manager.add_to_growing_conversation(
            request.user_id, request.session_id, request.message, ai_response
        )
        
        await memory_manager.save_conversation_turn(
            request.user_id, request.session_id, request.message, ai_response
        )

        return {"response": ai_response}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph execution error: {e}")

@app.post("/invoke_with_files")
async def invoke_agent_with_files(
    user_id: str = Form(...),
    session_id: str = Form(...),
    message: str = Form(...),
    files: List[UploadFile] = File(...)
):
    if not files:
        raise HTTPException(status_code=400, detail="No files were provided.")

    file_paths = {"uploaded_doc": "", "uploaded_img": ""}
    
    for file in files:
        unique_filename = f"{uuid.uuid4()}-{file.filename}"
        local_path = os.path.join("uploads", unique_filename)
        with open(local_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if 'image' in (file.content_type or ""):
            file_paths["uploaded_img"] = local_path
        else:
            file_paths["uploaded_doc"] = local_path
    
    thread_id = generate_thread_id(user_id, session_id)
    config = {"configurable": {"thread_id": thread_id}}
    
    context = await memory_manager.load_conversation_context(user_id, session_id)
    current_messages = memory_manager.get_current_messages(user_id, session_id)

    initial_state: GraphState = {
        "input": message,
        "user_id": user_id,
        "session_id": session_id,
        "uploaded_doc": file_paths.get("uploaded_doc"),
        "uploaded_img": file_paths.get("uploaded_img"),
        "messages": current_messages,
        "past_memory": context["past_memory"]
    }

    try:
        final_state = None
        async for event in graph.astream(initial_state, config=config):
            if "Aggregator" in event:
                final_state = event["Aggregator"]

        if not final_state or "final_response" not in final_state:
            raise HTTPException(status_code=500, detail="Graph did not produce a final response.")

        ai_response = final_state["final_response"]
        
        memory_manager.add_to_growing_conversation(
            user_id, session_id, message, ai_response
        )
        
        await memory_manager.save_conversation_turn(
            user_id, session_id, message, ai_response
        )

        return {"response": ai_response}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph execution error with files: {e}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("Main:app", host="0.0.0.0", port=port, reload=True)  # Updated module name

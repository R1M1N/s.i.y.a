"""
MCP Memory Server - Persistent conversation and context management
Provides long-term memory, user preferences, and conversation history
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from datetime import datetime, timedelta
import json
import uuid
import sqlite3
import os
from pathlib import Path

class MemoryEntry(BaseModel):
    id: str
    user_id: str
    session_id: str
    content_type: str  # conversation, preference, fact, context
    key: str
    value: Any
    metadata: Dict[str, Any] = {}
    created_at: str
    updated_at: str
    expires_at: Optional[str] = None

class UserPreferences(BaseModel):
    user_id: str
    language: str = "en"
    timezone: str = "UTC"
    response_style: str = "friendly"
    capabilities: List[str] = []
    custom_settings: Dict[str, Any] = {}

class MemoryServer:
    def __init__(self, db_path: str = "siya_memory.db"):
        self.server_name = "memory"
        self.tools = [
            "store_memory",
            "retrieve_memory", 
            "search_memory",
            "update_memory",
            "delete_memory",
            "get_conversation_history",
            "store_user_preferences",
            "get_user_preferences",
            "get_context_summary"
        ]
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for persistent storage"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create memory entries table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS memory_entries (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    expires_at TEXT
                )
            ''')
            
            # Create user preferences table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_preferences (
                    user_id TEXT PRIMARY KEY,
                    language TEXT,
                    timezone TEXT,
                    response_style TEXT,
                    capabilities TEXT,
                    custom_settings TEXT,
                    updated_at TEXT NOT NULL
                )
            ''')
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Database initialization error: {e}")
    
    def store_memory(self, user_id: str, session_id: str, content_type: str, 
                    key: str, value: Any, metadata: Dict[str, Any] = None, 
                    expires_in_hours: int = None) -> Dict[str, Any]:
        """Store a memory entry"""
        try:
            entry_id = str(uuid.uuid4())
            now = datetime.now().isoformat()
            
            # Calculate expiration
            expires_at = None
            if expires_in_hours:
                expires_at = (datetime.now() + timedelta(hours=expires_in_hours)).isoformat()
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO memory_entries 
                (id, user_id, session_id, content_type, key, value, metadata, created_at, updated_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                entry_id, user_id, session_id, content_type, key, 
                json.dumps(value), json.dumps(metadata or {}), now, now, expires_at
            ))
            
            conn.commit()
            conn.close()
            
            return {
                "success": True,
                "memory_id": entry_id,
                "key": key,
                "content_type": content_type,
                "expires_at": expires_at
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def retrieve_memory(self, user_id: str, key: str, content_type: str = None) -> Dict[str, Any]:
        """Retrieve memory entries by key and optional type"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Build query
            query = "SELECT * FROM memory_entries WHERE user_id = ? AND key = ?"
            params = [user_id, key]
            
            if content_type:
                query += " AND content_type = ?"
                params.append(content_type)
            
            query += " ORDER BY updated_at DESC LIMIT 10"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            if not rows:
                return {"success": False, "error": "No memory entries found"}
            
            entries = []
            for row in rows:
                entries.append({
                    "id": row[0],
                    "user_id": row[1],
                    "session_id": row[2],
                    "content_type": row[3],
                    "key": row[4],
                    "value": json.loads(row[5]),
                    "metadata": json.loads(row[6] or "{}"),
                    "created_at": row[7],
                    "updated_at": row[8],
                    "expires_at": row[9]
                })
            
            return {
                "success": True,
                "key": key,
                "content_type": content_type,
                "entries": entries
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def search_memory(self, user_id: str, query: str, content_type: str = None) -> Dict[str, Any]:
        """Search memory entries by content"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Build search query
            search_query = "SELECT * FROM memory_entries WHERE user_id = ? AND (key LIKE ? OR value LIKE ?)"
            search_param = f"%{query}%"
            params = [user_id, search_param, search_param]
            
            if content_type:
                search_query += " AND content_type = ?"
                params.append(content_type)
            
            search_query += " ORDER BY updated_at DESC LIMIT 20"
            
            cursor.execute(search_query, params)
            rows = cursor.fetchall()
            conn.close()
            
            entries = []
            for row in rows:
                entries.append({
                    "id": row[0],
                    "user_id": row[1],
                    "session_id": row[2],
                    "content_type": row[3],
                    "key": row[4],
                    "value": json.loads(row[5]),
                    "metadata": json.loads(row[6] or "{}"),
                    "created_at": row[7],
                    "updated_at": row[8],
                    "expires_at": row[9]
                })
            
            return {
                "success": True,
                "query": query,
                "content_type": content_type,
                "results_count": len(entries),
                "entries": entries
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_conversation_history(self, user_id: str, session_id: str = None, 
                               limit: int = 50) -> Dict[str, Any]:
        """Get conversation history for user/session"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Build query
            query = "SELECT * FROM memory_entries WHERE user_id = ? AND content_type = 'conversation'"
            params = [user_id]
            
            if session_id:
                query += " AND session_id = ?"
                params.append(session_id)
            
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            conversations = []
            for row in reversed(rows):  # Reverse to get chronological order
                conversations.append({
                    "id": row[0],
                    "session_id": row[2],
                    "key": row[4],
                    "value": json.loads(row[5]),
                    "metadata": json.loads(row[6] or "{}"),
                    "created_at": row[7]
                })
            
            return {
                "success": True,
                "user_id": user_id,
                "session_id": session_id,
                "conversations_count": len(conversations),
                "conversations": conversations
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def store_user_preferences(self, user_id: str, preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Store user preferences"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            now = datetime.now().isoformat()
            
            # Extract preference fields
            language = preferences.get("language", "en")
            timezone = preferences.get("timezone", "UTC")
            response_style = preferences.get("response_style", "friendly")
            capabilities = json.dumps(preferences.get("capabilities", []))
            custom_settings = json.dumps(preferences.get("custom_settings", {}))
            
            # Insert or update
            cursor.execute('''
                INSERT OR REPLACE INTO user_preferences 
                (user_id, language, timezone, response_style, capabilities, custom_settings, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (user_id, language, timezone, response_style, capabilities, custom_settings, now))
            
            conn.commit()
            conn.close()
            
            return {
                "success": True,
                "user_id": user_id,
                "preferences": preferences,
                "updated_at": now
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user preferences"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM user_preferences WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()
            conn.close()
            
            if not row:
                return {
                    "success": True,
                    "user_id": user_id,
                    "preferences": {
                        "language": "en",
                        "timezone": "UTC", 
                        "response_style": "friendly",
                        "capabilities": [],
                        "custom_settings": {}
                    },
                    "created": True
                }
            
            return {
                "success": True,
                "user_id": user_id,
                "preferences": {
                    "language": row[1],
                    "timezone": row[2],
                    "response_style": row[3],
                    "capabilities": json.loads(row[4] or "[]"),
                    "custom_settings": json.loads(row[5] or "{}")
                },
                "updated_at": row[6]
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_context_summary(self, user_id: str, session_id: str = None) -> Dict[str, Any]:
        """Get a summary of user's context and recent activity"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get memory statistics
            cursor.execute('''
                SELECT content_type, COUNT(*) 
                FROM memory_entries 
                WHERE user_id = ? 
                GROUP BY content_type
            ''', (user_id,))
            
            type_counts = dict(cursor.fetchall())
            
            # Get recent conversations
            query = "SELECT key, value, created_at FROM memory_entries WHERE user_id = ? AND content_type = 'conversation'"
            params = [user_id]
            
            if session_id:
                query += " AND session_id = ?"
                params.append(session_id)
            
            query += " ORDER BY created_at DESC LIMIT 10"
            
            cursor.execute(query, params)
            recent_conversations = cursor.fetchall()
            conn.close()
            
            # Process recent conversations
            conversations = []
            for row in recent_conversations:
                conversations.append({
                    "type": row[0],
                    "content": json.loads(row[1]),
                    "timestamp": row[2]
                })
            
            return {
                "success": True,
                "user_id": user_id,
                "session_id": session_id,
                "summary": {
                    "memory_counts": type_counts,
                    "total_memories": sum(type_counts.values()),
                    "recent_conversations": conversations,
                    "context_types": list(type_counts.keys())
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

# FastAPI server for MCP communication
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="MCP Memory Server", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

memory_server = MemoryServer()

@app.get("/")
async def root():
    return {
        "server": "memory",
        "tools": memory_server.tools,
        "db_path": memory_server.db_path,
        "status": "running"
    }

@app.post("/tools/store_memory")
async def store_memory(request: Dict[str, Any]):
    try:
        user_id = request.get("user_id")
        session_id = request.get("session_id")
        content_type = request.get("content_type")
        key = request.get("key")
        value = request.get("value")
        metadata = request.get("metadata")
        expires_in_hours = request.get("expires_in_hours")
        
        if not all([user_id, session_id, content_type, key, value]):
            raise HTTPException(status_code=400, detail="user_id, session_id, content_type, key, value required")
        
        result = memory_server.store_memory(
            user_id, session_id, content_type, key, value, metadata, expires_in_hours
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tools/retrieve_memory")
async def retrieve_memory(request: Dict[str, Any]):
    try:
        user_id = request.get("user_id")
        key = request.get("key")
        content_type = request.get("content_type")
        
        if not all([user_id, key]):
            raise HTTPException(status_code=400, detail="user_id and key required")
        
        result = memory_server.retrieve_memory(user_id, key, content_type)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tools/get_conversation_history")
async def get_conversation_history(request: Dict[str, Any]):
    try:
        user_id = request.get("user_id")
        session_id = request.get("session_id")
        limit = request.get("limit", 50)
        
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id required")
        
        result = memory_server.get_conversation_history(user_id, session_id, limit)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tools/get_user_preferences")
async def get_user_preferences(request: Dict[str, Any]):
    try:
        user_id = request.get("user_id")
        
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id required")
        
        result = memory_server.get_user_preferences(user_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
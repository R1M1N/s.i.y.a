"""
MCP Time Server - Real-time date and time operations
Provides current time, timezone conversions, and scheduling capabilities
"""

from datetime import datetime, timezone, timedelta
import json
from typing import Dict, Any, Optional
from pydantic import BaseModel

class TimeRequest(BaseModel):
    timezone: Optional[str] = "UTC"
    format: Optional[str] = "iso"

class TimeServer:
    def __init__(self):
        self.server_name = "time"
        self.tools = [
            "current_time",
            "timezone_convert", 
            "time_until",
            "schedule_reminder"
        ]
    
    def get_current_time(self, timezone: str = "UTC", format: str = "iso") -> Dict[str, Any]:
        """Get current time in specified timezone and format"""
        try:
            if timezone.upper() == "UTC":
                now = datetime.now(timezone.utc)
            else:
                # Simple timezone handling (in real implementation, use pytz)
                offset_map = {
                    "EST": -5, "EDT": -4, "CST": -6, "CDT": -5,
                    "MST": -7, "MDT": -6, "PST": -8, "PDT": -7
                }
                offset = offset_map.get(timezone.upper(), 0)
                now = datetime.now(timezone.utc) + timedelta(hours=offset)
            
            if format == "iso":
                time_str = now.isoformat()
            elif format == "human":
                time_str = now.strftime("%A, %B %d, %Y at %I:%M %p")
            elif format == "24h":
                time_str = now.strftime("%Y-%m-%d %H:%M:%S")
            else:
                time_str = now.isoformat()
            
            return {
                "success": True,
                "time": time_str,
                "timezone": timezone,
                "unix_timestamp": int(now.timestamp()),
                "iso_format": now.isoformat(),
                "weekday": now.strftime("%A"),
                "date": now.strftime("%Y-%m-%d"),
                "time_24h": now.strftime("%H:%M:%S"),
                "time_12h": now.strftime("%I:%M %p")
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def time_until(self, target_time: str, from_time: str = None) -> Dict[str, Any]:
        """Calculate time until a target time"""
        try:
            if from_time is None:
                from_time = datetime.now(timezone.utc).isoformat()
            
            target = datetime.fromisoformat(target_time.replace('Z', '+00:00'))
            from_dt = datetime.fromisoformat(from_time.replace('Z', '+00:00'))
            
            diff = target - from_dt
            
            if diff.total_seconds() < 0:
                return {"success": False, "error": "Target time is in the past"}
            
            days = diff.days
            hours = diff.seconds // 3600
            minutes = (diff.seconds % 3600) // 60
            seconds = diff.seconds % 60
            
            return {
                "success": True,
                "time_until": {
                    "days": days,
                    "hours": hours, 
                    "minutes": minutes,
                    "seconds": seconds,
                    "total_seconds": int(diff.total_seconds()),
                    "human_readable": f"{days}d {hours}h {minutes}m {seconds}s"
                },
                "target_time": target_time,
                "from_time": from_time
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def schedule_reminder(self, message: str, time_offset: str) -> Dict[str, Any]:
        """Schedule a reminder (basic implementation)"""
        try:
            # Parse time offset (e.g., "5 minutes", "2 hours", "1 day")
            parts = time_offset.lower().split()
            if len(parts) < 2:
                return {"success": False, "error": "Invalid time format"}
            
            amount = int(parts[0])
            unit = parts[1]
            
            if unit.startswith("minute"):
                seconds = amount * 60
            elif unit.startswith("hour"):
                seconds = amount * 3600
            elif unit.startswith("day"):
                seconds = amount * 86400
            else:
                return {"success": False, "error": "Unsupported time unit"}
            
            reminder_time = datetime.now(timezone.utc) + timedelta(seconds=seconds)
            
            return {
                "success": True,
                "reminder": {
                    "message": message,
                    "scheduled_for": reminder_time.isoformat(),
                    "time_until": seconds,
                    "human_readable": f"{amount} {unit}"
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

# FastAPI server for MCP communication
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="MCP Time Server", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

time_server = TimeServer()

@app.get("/")
async def root():
    return {
        "server": "time",
        "tools": time_server.tools,
        "status": "running"
    }

@app.post("/tools/current_time")
async def current_time(request: Dict[str, Any]):
    try:
        timezone = request.get("timezone", "UTC")
        format_type = request.get("format", "iso")
        result = time_server.get_current_time(timezone, format_type)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tools/time_until")
async def time_until(request: Dict[str, Any]):
    try:
        target_time = request.get("target_time")
        from_time = request.get("from_time")
        if not target_time:
            raise HTTPException(status_code=400, detail="target_time required")
        result = time_server.time_until(target_time, from_time)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tools/schedule_reminder")
async def schedule_reminder(request: Dict[str, Any]):
    try:
        message = request.get("message")
        time_offset = request.get("time_offset")
        if not message or not time_offset:
            raise HTTPException(status_code=400, detail="message and time_offset required")
        result = time_server.schedule_reminder(message, time_offset)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
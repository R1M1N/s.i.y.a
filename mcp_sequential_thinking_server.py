"""
MCP Sequential Thinking Server - Break down complex tasks into steps
Provides structured reasoning and task planning capabilities
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from datetime import datetime
import json
import uuid

class ThinkingStep(BaseModel):
    id: str
    step_number: int
    description: str
    status: str = "pending"  # pending, in_progress, completed, failed
    result: Optional[str] = None
    error: Optional[str] = None
    timestamp: str = None

class TaskPlan(BaseModel):
    id: str
    title: str
    description: str
    steps: List[ThinkingStep]
    status: str = "planning"  # planning, executing, completed, failed
    created_at: str
    completed_at: Optional[str] = None

class SequentialThinkingServer:
    def __init__(self):
        self.server_name = "sequentialthinking"
        self.tools = [
            "create_plan",
            "add_step", 
            "execute_step",
            "get_plan_status",
            "modify_step",
            "complete_plan"
        ]
        self.active_plans: Dict[str, TaskPlan] = {}
    
    def create_plan(self, title: str, description: str, steps: List[str] = None) -> Dict[str, Any]:
        """Create a new task plan with optional initial steps"""
        try:
            plan_id = str(uuid.uuid4())
            created_at = datetime.now().isoformat()
            
            plan_steps = []
            if steps:
                for i, step_desc in enumerate(steps):
                    step = ThinkingStep(
                        id=str(uuid.uuid4()),
                        step_number=i + 1,
                        description=step_desc,
                        timestamp=created_at
                    )
                    plan_steps.append(step)
            
            plan = TaskPlan(
                id=plan_id,
                title=title,
                description=description,
                steps=plan_steps,
                created_at=created_at
            )
            
            self.active_plans[plan_id] = plan
            
            return {
                "success": True,
                "plan": {
                    "id": plan_id,
                    "title": title,
                    "description": description,
                    "steps_count": len(plan_steps),
                    "status": plan.status,
                    "created_at": created_at
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def add_step(self, plan_id: str, description: str, step_number: int = None) -> Dict[str, Any]:
        """Add a step to an existing plan"""
        try:
            if plan_id not in self.active_plans:
                return {"success": False, "error": "Plan not found"}
            
            plan = self.active_plans[plan_id]
            
            # Determine step number
            if step_number is None:
                step_number = len(plan.steps) + 1
            
            # Reorder steps if needed
            if step_number <= len(plan.steps):
                for step in plan.steps[step_number-1:]:
                    step.step_number += 1
            
            # Create new step
            new_step = ThinkingStep(
                id=str(uuid.uuid4()),
                step_number=step_number,
                description=description,
                timestamp=datetime.now().isoformat()
            )
            
            # Insert step at correct position
            plan.steps.insert(step_number - 1, new_step)
            
            return {
                "success": True,
                "step": {
                    "id": new_step.id,
                    "step_number": new_step.step_number,
                    "description": new_step.description,
                    "status": new_step.status
                },
                "plan_steps_count": len(plan.steps)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def execute_step(self, plan_id: str, step_id: str, result: str = None, error: str = None) -> Dict[str, Any]:
        """Execute a step and update its status"""
        try:
            if plan_id not in self.active_plans:
                return {"success": False, "error": "Plan not found"}
            
            plan = self.active_plans[plan_id]
            step = next((s for s in plan.steps if s.id == step_id), None)
            
            if not step:
                return {"success": False, "error": "Step not found"}
            
            # Update step status
            if error:
                step.status = "failed"
                step.error = error
            else:
                step.status = "completed"
                step.result = result
            
            # Update plan status
            all_completed = all(s.status == "completed" for s in plan.steps)
            any_failed = any(s.status == "failed" for s in plan.steps)
            
            if all_completed:
                plan.status = "completed"
                plan.completed_at = datetime.now().isoformat()
            elif any_failed:
                plan.status = "failed"
            else:
                plan.status = "executing"
            
            return {
                "success": True,
                "step": {
                    "id": step.id,
                    "step_number": step.step_number,
                    "description": step.description,
                    "status": step.status,
                    "result": step.result,
                    "error": step.error
                },
                "plan_status": plan.status,
                "progress": f"{sum(1 for s in plan.steps if s.status == 'completed')}/{len(plan.steps)} steps completed"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_plan_status(self, plan_id: str) -> Dict[str, Any]:
        """Get detailed status of a plan"""
        try:
            if plan_id not in self.active_plans:
                return {"success": False, "error": "Plan not found"}
            
            plan = self.active_plans[plan_id]
            
            steps_status = []
            for step in plan.steps:
                steps_status.append({
                    "id": step.id,
                    "step_number": step.step_number,
                    "description": step.description,
                    "status": step.status,
                    "result": step.result,
                    "error": step.error,
                    "timestamp": step.timestamp
                })
            
            progress_percent = (sum(1 for s in plan.steps if s.status == "completed") / len(plan.steps)) * 100 if plan.steps else 0
            
            return {
                "success": True,
                "plan": {
                    "id": plan.id,
                    "title": plan.title,
                    "description": plan.description,
                    "status": plan.status,
                    "progress_percent": round(progress_percent, 1),
                    "steps_count": len(plan.steps),
                    "completed_steps": sum(1 for s in plan.steps if s.status == "completed"),
                    "failed_steps": sum(1 for s in plan.steps if s.status == "failed"),
                    "created_at": plan.created_at,
                    "completed_at": plan.completed_at,
                    "steps": steps_status
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def modify_step(self, plan_id: str, step_id: str, new_description: str = None, new_number: int = None) -> Dict[str, Any]:
        """Modify an existing step"""
        try:
            if plan_id not in self.active_plans:
                return {"success": False, "error": "Plan not found"}
            
            plan = self.active_plans[plan_id]
            step = next((s for s in plan.steps if s.id == step_id), None)
            
            if not step:
                return {"success": False, "error": "Step not found"}
            
            old_description = step.description
            old_number = step.step_number
            
            if new_description:
                step.description = new_description
            
            if new_number and new_number != step.step_number:
                # Remove from current position
                plan.steps.remove(step)
                
                # Adjust other step numbers
                for s in plan.steps:
                    if s.step_number >= new_number:
                        s.step_number += 1
                
                # Insert at new position
                step.step_number = new_number
                plan.steps.insert(new_number - 1, step)
            
            return {
                "success": True,
                "step": {
                    "id": step.id,
                    "step_number": step.step_number,
                    "description": step.description,
                    "status": step.status
                },
                "changes": {
                    "description_changed": new_description is not None,
                    "number_changed": new_number is not None,
                    "old_description": old_description,
                    "old_number": old_number
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def complete_plan(self, plan_id: str) -> Dict[str, Any]:
        """Mark a plan as completed"""
        try:
            if plan_id not in self.active_plans:
                return {"success": False, "error": "Plan not found"}
            
            plan = self.active_plans[plan_id]
            plan.status = "completed"
            plan.completed_at = datetime.now().isoformat()
            
            # Mark all pending steps as completed
            for step in plan.steps:
                if step.status == "pending":
                    step.status = "completed"
                    step.result = "Auto-completed when plan finished"
            
            return {
                "success": True,
                "plan": {
                    "id": plan.id,
                    "title": plan.title,
                    "status": plan.status,
                    "completed_at": plan.completed_at,
                    "final_steps_count": len(plan.steps)
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

# FastAPI server for MCP communication
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="MCP Sequential Thinking Server", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

thinking_server = SequentialThinkingServer()

@app.get("/")
async def root():
    return {
        "server": "sequentialthinking",
        "tools": thinking_server.tools,
        "active_plans": len(thinking_server.active_plans),
        "status": "running"
    }

@app.post("/tools/create_plan")
async def create_plan(request: Dict[str, Any]):
    try:
        title = request.get("title")
        description = request.get("description", "")
        steps = request.get("steps", [])
        
        if not title:
            raise HTTPException(status_code=400, detail="title required")
        
        result = thinking_server.create_plan(title, description, steps)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tools/add_step")
async def add_step(request: Dict[str, Any]):
    try:
        plan_id = request.get("plan_id")
        description = request.get("description")
        step_number = request.get("step_number")
        
        if not plan_id or not description:
            raise HTTPException(status_code=400, detail="plan_id and description required")
        
        result = thinking_server.add_step(plan_id, description, step_number)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tools/execute_step")
async def execute_step(request: Dict[str, Any]):
    try:
        plan_id = request.get("plan_id")
        step_id = request.get("step_id")
        result = request.get("result")
        error = request.get("error")
        
        if not plan_id or not step_id:
            raise HTTPException(status_code=400, detail="plan_id and step_id required")
        
        result = thinking_server.execute_step(plan_id, step_id, result, error)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tools/get_plan_status")
async def get_plan_status(request: Dict[str, Any]):
    try:
        plan_id = request.get("plan_id")
        
        if not plan_id:
            raise HTTPException(status_code=400, detail="plan_id required")
        
        result = thinking_server.get_plan_status(plan_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
"""FastAPI service for RT auto-contouring pipeline.

This module provides a REST API for triggering auto-contouring tasks
and integrating with workflow systems like n8n.
"""

from pathlib import Path
from typing import Optional, Dict, List
import logging
from datetime import datetime
import asyncio
from enum import Enum

from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import uvicorn

from config.paths import get_path_config
from src.inference.predict import InferenceEngine
from src.inference.nifti_to_rtstruct import NiftiToRtstructConverter

logger = logging.getLogger(__name__)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


# Pydantic models for API
class TaskStatus(str, Enum):
    """Task status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ContoringRequest(BaseModel):
    """Request model for auto-contouring."""
    case_id: str = Field(..., description="Unique case identifier")
    ct_series_path: str = Field(..., description="Path to CT DICOM series")
    output_dir: Optional[str] = Field(None, description="Output directory path")
    create_rtstruct: bool = Field(True, description="Create DICOM RTSTRUCT output")
    notify_url: Optional[str] = Field(None, description="Webhook URL for completion notification")


class ContoringResponse(BaseModel):
    """Response model for auto-contouring task."""
    task_id: str
    status: TaskStatus
    message: str
    case_id: str


class TaskStatusResponse(BaseModel):
    """Response model for task status query."""
    task_id: str
    status: TaskStatus
    case_id: str
    created_at: str
    updated_at: str
    result: Optional[Dict] = None
    error: Optional[str] = None


# Global task storage (in production, use Redis or database)
tasks: Dict[str, Dict] = {}


# Initialize FastAPI app
app = FastAPI(
    title="RT Auto-Contouring API",
    description="API for radiotherapy auto-contouring using nnU-Net",
    version="0.1.0"
)


# Global inference engine (initialized on startup)
inference_engine: Optional[InferenceEngine] = None
rtstruct_converter: Optional[NiftiToRtstructConverter] = None


@app.on_event("startup")
async def startup_event():
    """Initialize inference engine on startup."""
    global inference_engine, rtstruct_converter
    
    logger.info("Initializing inference engine...")
    
    try:
        path_config = get_path_config()
        
        # Initialize inference engine with default settings
        # In production, these should be configurable via environment variables
        inference_engine = InferenceEngine(
            dataset_id=1,  # Default dataset
            path_config=path_config,
            configuration="3d_fullres",
            trainer="nnUNetTrainer",
            plans="nnUNetPlans",
            folds="all"
        )
        
        rtstruct_converter = NiftiToRtstructConverter()
        
        logger.info("Inference engine initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize inference engine: {e}")
        # In production, you might want to prevent app startup on failure


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "RT Auto-Contouring API",
        "version": "0.1.0",
        "status": "running",
        "inference_ready": inference_engine is not None
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "inference_engine": "ready" if inference_engine is not None else "not_ready"
    }


@app.post("/contour", response_model=ContoringResponse)
async def create_contouring_task(
    request: ContoringRequest,
    background_tasks: BackgroundTasks
):
    """
    Create a new auto-contouring task.
    
    This endpoint accepts a contouring request and starts processing
    in the background. Returns immediately with a task ID for tracking.
    """
    if inference_engine is None:
        raise HTTPException(
            status_code=503,
            detail="Inference engine not initialized"
        )
    
    # Generate task ID
    task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{request.case_id}"
    
    # Create task record
    task_record = {
        "task_id": task_id,
        "status": TaskStatus.PENDING,
        "case_id": request.case_id,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "request": request.dict(),
        "result": None,
        "error": None
    }
    
    tasks[task_id] = task_record
    
    # Add task to background processing
    background_tasks.add_task(
        process_contouring_task,
        task_id=task_id,
        request=request
    )
    
    logger.info(f"Created task {task_id} for case {request.case_id}")
    
    return ContoringResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        message="Task created successfully",
        case_id=request.case_id
    )


@app.get("/task/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """
    Get status of a contouring task.
    """
    if task_id not in tasks:
        raise HTTPException(
            status_code=404,
            detail=f"Task {task_id} not found"
        )
    
    task = tasks[task_id]
    
    return TaskStatusResponse(
        task_id=task["task_id"],
        status=task["status"],
        case_id=task["case_id"],
        created_at=task["created_at"],
        updated_at=task["updated_at"],
        result=task.get("result"),
        error=task.get("error")
    )


@app.get("/task/{task_id}/result")
async def get_task_result(task_id: str):
    """
    Download result files for a completed task.
    """
    if task_id not in tasks:
        raise HTTPException(
            status_code=404,
            detail=f"Task {task_id} not found"
        )
    
    task = tasks[task_id]
    
    if task["status"] != TaskStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Task is not completed (status: {task['status']})"
        )
    
    if task["result"] is None or "rtstruct" not in task["result"]:
        raise HTTPException(
            status_code=404,
            detail="Result file not available"
        )
    
    rtstruct_path = Path(task["result"]["rtstruct"])
    
    if not rtstruct_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Result file not found on disk"
        )
    
    return FileResponse(
        path=str(rtstruct_path),
        filename=rtstruct_path.name,
        media_type="application/dicom"
    )


@app.get("/tasks")
async def list_tasks(
    status: Optional[TaskStatus] = None,
    limit: int = 100
):
    """
    List all tasks with optional filtering.
    """
    filtered_tasks = []
    
    for task in tasks.values():
        if status is None or task["status"] == status:
            filtered_tasks.append({
                "task_id": task["task_id"],
                "status": task["status"],
                "case_id": task["case_id"],
                "created_at": task["created_at"],
                "updated_at": task["updated_at"]
            })
    
    # Sort by created_at descending
    filtered_tasks.sort(key=lambda x: x["created_at"], reverse=True)
    
    return {
        "tasks": filtered_tasks[:limit],
        "total": len(filtered_tasks)
    }


async def process_contouring_task(
    task_id: str,
    request: ContoringRequest
):
    """
    Background task for processing contouring request.
    """
    logger.info(f"Starting processing for task {task_id}")
    
    # Update task status
    tasks[task_id]["status"] = TaskStatus.RUNNING
    tasks[task_id]["updated_at"] = datetime.now().isoformat()
    
    try:
        ct_series_path = Path(request.ct_series_path)
        
        if not ct_series_path.exists():
            raise ValueError(f"CT series path does not exist: {ct_series_path}")
        
        # Run inference
        logger.info(f"Running inference for case {request.case_id}")
        
        inference_result = inference_engine.predict_from_dicom(
            ct_series_path=ct_series_path,
            output_dir=Path(request.output_dir) if request.output_dir else None,
            case_id=request.case_id
        )
        
        result = {
            "case_id": inference_result["case_id"],
            "segmentation": str(inference_result["segmentation"]),
            "ct": str(inference_result["ct"]),
            "output_dir": str(inference_result["output_dir"])
        }
        
        # Create RTSTRUCT if requested
        if request.create_rtstruct:
            logger.info(f"Creating RTSTRUCT for case {request.case_id}")
            
            rtstruct_path = inference_result["output_dir"] / f"{request.case_id}_RTSTRUCT.dcm"
            
            rtstruct_converter.convert(
                segmentation_path=inference_result["segmentation"],
                ct_series_path=ct_series_path,
                output_path=rtstruct_path
            )
            
            result["rtstruct"] = str(rtstruct_path)
        
        # Update task with results
        tasks[task_id]["status"] = TaskStatus.COMPLETED
        tasks[task_id]["result"] = result
        tasks[task_id]["updated_at"] = datetime.now().isoformat()
        
        logger.info(f"Task {task_id} completed successfully")
        
        # TODO: Send webhook notification if notify_url is provided
        if request.notify_url:
            logger.info(f"Would send notification to {request.notify_url}")
        
    except Exception as e:
        logger.error(f"Task {task_id} failed: {e}", exc_info=True)
        
        tasks[task_id]["status"] = TaskStatus.FAILED
        tasks[task_id]["error"] = str(e)
        tasks[task_id]["updated_at"] = datetime.now().isoformat()


def start_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False
):
    """
    Start the FastAPI server.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        reload: Enable auto-reload for development
    """
    uvicorn.run(
        "src.service.pipeline:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    start_server()

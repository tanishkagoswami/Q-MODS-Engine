from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import List, Dict, Any

router = APIRouter(
    prefix="/api/roles",
    tags=["Role Management & Role-Based Views"]
)

# Mock User-Role DB mapping
USER_ROLES = {
    "physician_john": "clinician",
    "researcher_mary": "researcher",
    "sys_admin": "admin"
}

class RoleAccessRequest(BaseModel):
    username: str

@router.post("/verify")
def verify_role_access(req: RoleAccessRequest):
    """
    Verifies user identity and assigns appropriate interface permissions.
    """
    role = USER_ROLES.get(req.username.lower())
    if not role:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username. Registered users: 'physician_john', 'researcher_mary', 'sys_admin'"
        )
    return {"username": req.username, "role": role}

@router.get("/view/clinician")
def get_clinician_dashboard():
    """
    Exposes granular clinical metrics, early-warning system organ indicators, 
    and therapeutic treatment simulation panels.
    """
    return {
        "view_permissions": "clinician",
        "allowed_modules": ["Early Warning", "Dynamic Cascade Graphs", "Intervention Sandbox"],
        "instructions": "Early interventions focus on fluid control and MAP preservation."
    }

@router.get("/view/researcher")
def get_researcher_dashboard():
    """
    Exposes raw high-frequency cohort waveforms, deep Quantum VQC weights, 
    and model hyperparameter tuning panels.
    """
    return {
        "view_permissions": "researcher",
        "allowed_modules": ["Quantum Feature State Space", "TCN Embeddings", "Explainability Weights"],
        "instructions": "Supports export to PyTorch / PennyLane tensor formats."
    }

@router.get("/physician")
async def get_physician_config():
    """Fetches configuration and permissions for the Physician dashboard."""
    return {
        "role": "physician",
        "permissions": ["simulate_therapy", "view_cascade_velocity", "override_alarms"],
        "active_modules": ["digital_twin", "hazard_deep_dive"]
    }

@router.get("/nurse")
async def get_nurse_config():
    """Fetches configuration and permissions for the Nurse dashboard."""
    return {
        "role": "nurse",
        "permissions": ["view_task_queue", "acknowledge_alerts", "update_chart"],
        "active_modules": ["task_reranker", "supply_staging"]
    }

@router.get("/patient")
async def get_patient_config():
    """Fetches configuration and permissions for the Patient/Family dashboard."""
    return {
        "role": "patient",
        "permissions": ["view_summary"],
        "active_modules": ["traffic_light_board", "plain_english_summary"]
    }
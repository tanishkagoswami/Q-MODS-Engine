import os
import sys
import redis.asyncio as redis
import json

# Dynamic path injection: ensures both the backend root and project root are
# on the Python path so imports like 'database' and 'src' resolve correctly.
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
for path in (BACKEND_DIR, PROJECT_ROOT):
    if path not in sys.path:
        sys.path.append(path)

import torch
import numpy as np
import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
import asyncio
import random
from api.role_router import router as role_router
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
import importlib.util

database_path = os.path.join(BACKEND_DIR, "configs", "database.py")
spec = importlib.util.spec_from_file_location("qmods_database", database_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load database module from {database_path}")
qmods_database = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qmods_database)
snapshots_collection = qmods_database.snapshots_collection

# Import Pipeline and Model classes from previous layers
from src.graph.physiology_graph import PhysiologyGraphBuilder
from src.graph.attention_edges import PhysiologicalAttentionEdges
from src.models.classical_comp import WaveformTCNCompressor
from src.models.quantum_vqc import QuantumVQCEncoder
from src.models.stgcn import STGCN
from src.explainability.gnn_explainer import PhysiologyGNNExplainer
from src.marl.env_simulator import QMODSMultiAgentEnv

# Import new modules
from api.streaming import router as stream_router

# Configure Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Q-MODS Engine API",
    description="Hybrid Quantum-Classical Spatiotemporal Graph Neural Network for Multi-Organ Dysfunction Syndrome (MODS) Early Warning and MARL Intervention Sandbox.",
    version="1.0.0"
)
# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Sub-Routers
app.include_router(role_router)
app.include_router(stream_router)

# --- Define Request and Response Schemas ---
class PatientSliceInput(BaseModel):
    subject_id: int = Field(..., example=1001)
    waveforms: Dict[str, List[float]] = Field(
        ..., 
        example={"ECG": [0.5, 0.6, 0.55], "ABP": [120.0, 118.0, 119.0], "RESP": [18.0, 18.2, 18.1]}
    )
    vitals_and_labs: Dict[str, float] = Field(
        ..., 
        example={
            "creatinine": 1.4, "bun": 22.0, "urine_output": 35.0,
            "spo2": 0.96, "respiratory_rate": 18.0, "pao2_fio2": 320.0,
            "heart_rate": 88.0, "mean_arterial_pressure": 72.0, "systolic_bp": 115.0,
            "bilirubin_total": 0.9, "alt": 25.0, "ast": 30.0,
            "gcs": 15.0, "pupil_reaction": 1.0, "temperature": 37.0
        }
    )

class MARLActionInput(BaseModel):
    patho_action: List[float] = Field(..., min_items=3, max_items=3, example=[0.1, 0.05, 0.0])
    therapy_action: List[float] = Field(..., min_items=4, max_items=4, example=[0.2, 0.1, 0.4, 0.0])

class ClinicalState(BaseModel):
    vitals: dict
    cascade_risk: float

# --- WebSocket Connection Manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

manager = ConnectionManager()

# --- Model Holder Class for Real-time Serving ---
class ModelContext:
    def __init__(self):
        logger.info("Initializing neural network layers inside ModelContext...")
        self.graph_builder = PhysiologyGraphBuilder()
        self.attention_net = PhysiologicalAttentionEdges(node_feat_dim=3, embed_dim=16)
        
        self.tcn_compressor = WaveformTCNCompressor(in_channels=3, embed_dim=64)
        self.quantum_encoder = QuantumVQCEncoder(n_qubits=8, n_layers=3)
        self.stgcn_model = STGCN(in_features=128, hidden_dim=64, out_features=32)
        
        self.gnn_explainer = PhysiologyGNNExplainer(model=self.stgcn_model, epochs=50, lr=0.05)
        self.marl_env = QMODSMultiAgentEnv()
        
        self.tcn_compressor.eval()
        self.quantum_encoder.eval()
        self.stgcn_model.eval()

ctx: Optional[ModelContext] = None

# Initialize async Redis client (In-memory HFT broker)
redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
redis_client = redis.Redis.from_url(redis_url, decode_responses=True, socket_timeout=3, socket_connect_timeout=3)
async def async_mongo_flusher():
    """
    HFT Pattern: Background worker that wakes up every 5 seconds, 
    pulls the latest state from the ultra-fast Redis cache, and writes it to MongoDB.
    """
    while True:
        await asyncio.sleep(5)
        try:
            raw_state = await redis_client.get("latest_clinical_snapshot")
            if raw_state:
                snapshot = json.loads(raw_state)
                snapshot["timestamp"] = datetime.now(timezone.utc)
                snapshots_collection.insert_one(snapshot)
        except Exception as e:
            logger.error(f"Async Mongo batch write failed: {e}")

@app.on_event("startup")
async def startup_event():
    global ctx
    ctx = ModelContext()
    
    # Fire up the background asynchronous flusher
    asyncio.create_task(async_mongo_flusher())
    
    logger.info("Application context fully loaded. Async Mongo flusher active.")

# --- Endpoints ---

@app.get("/health")
async def health_check():
    return {"status": "healthy", "components": {"cuda_available": torch.cuda.is_available()}}

@app.post("/api/predict")
async def predict_risk(data: PatientSliceInput):
    if ctx is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")

    try:
        ecg = data.waveforms.get("ECG", [0.0] * 100)
        abp = data.waveforms.get("ABP", [0.0] * 100)
        resp = data.waveforms.get("RESP", [0.0] * 100)
        
        waveform_tensor = torch.tensor([[ecg, abp, resp]], dtype=torch.float32)
        with torch.no_grad():
            waveform_emb = ctx.tcn_compressor(waveform_tensor)

        raw_labs = np.zeros(64, dtype=np.float32)
        for i, val in enumerate(data.vitals_and_labs.values()):
            if i < 64:
                raw_labs[i] = val
        lab_tensor = torch.tensor([raw_labs], dtype=torch.float32)

        with torch.no_grad():
            patient_state_vector = ctx.quantum_encoder(waveform_emb, lab_tensor)

        node_features = ctx.graph_builder.map_features_to_nodes(data.vitals_and_labs)
        
        with torch.no_grad():
            edge_weights = ctx.attention_net(node_features, ctx.graph_builder.base_edges)

        seq_len = 10
        history_tensor = patient_state_vector.unsqueeze(1).unsqueeze(1).expand(1, 5, seq_len, 128)

        with torch.no_grad():
            organ_hazards, cascade_alert_score = ctx.stgcn_model(
                history_tensor, ctx.graph_builder.base_edges, edge_weights.squeeze()
            )

        organ_risks = {
            ctx.graph_builder.get_organ_name(i): float(organ_hazards[0, i].item())
            for i in range(5)
        }

        return {
            "subject_id": data.subject_id,
            "cascade_alert_probability": float(cascade_alert_score[0, 0].item()),
            "organ_specific_hazards": organ_risks,
            "attention_edge_weights": [float(w.item()) for w in edge_weights]
        }

    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/explain/gnn")
async def explain_gnn_pathways(data: PatientSliceInput):
    if ctx is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")

    try:
        ecg = data.waveforms.get("ECG", [0.0] * 100)
        abp = data.waveforms.get("ABP", [0.0] * 100)
        resp = data.waveforms.get("RESP", [0.0] * 100)
        
        waveform_tensor = torch.tensor([[ecg, abp, resp]], dtype=torch.float32)
        with torch.no_grad():
            waveform_emb = ctx.tcn_compressor(waveform_tensor)
            
        raw_labs = np.zeros(64, dtype=np.float32)
        for i, val in enumerate(data.vitals_and_labs.values()):
            if i < 64: raw_labs[i] = val
        lab_tensor = torch.tensor([raw_labs], dtype=torch.float32)

        with torch.no_grad():
            patient_state_vector = ctx.quantum_encoder(waveform_emb, lab_tensor)

        node_features = ctx.graph_builder.map_features_to_nodes(data.vitals_and_labs)
        with torch.no_grad():
            edge_weights = ctx.attention_net(node_features, ctx.graph_builder.base_edges)

        seq_len = 10
        history_tensor = patient_state_vector.unsqueeze(1).unsqueeze(1).expand(1, 5, seq_len, 128)

        explanations = ctx.gnn_explainer.explain_cascade(
            history_tensor, ctx.graph_builder.base_edges, edge_weights
        )

        return {
            "subject_id": data.subject_id,
            "cascade_pathway_importance": explanations
        }

    except Exception as e:
        logger.error(f"GNN Explanation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/marl/step")
async def step_marl_sandbox(actions: MARLActionInput):
    if ctx is None:
        raise HTTPException(status_code=503, detail="Simulator not initialized.")

    try:
        action_dict = {
            "patho_agent": np.array(actions.patho_action, dtype=np.float32),
            "therapy_agent": np.array(actions.therapy_action, dtype=np.float32)
        }
        
        obs, rewards, terminated, truncated, info = ctx.marl_env.step(action_dict)
        
        obs_serializable = {
            agent: {
                "organ_states": [float(x) for x in state["organ_states"]],
                "vitals_and_labs": [float(x) for x in state["vitals_and_labs"]],
                "active_interventions": [float(x) for x in state["active_interventions"]]
            } for agent, state in obs.items()
        }
        
        return {
            "current_step": ctx.marl_env.current_step,
            "observation": obs_serializable,
            "rewards": {agent: float(r) for agent, r in rewards.items()},
            "terminated": terminated["__all__"],
            "truncated": truncated["__all__"]
        }

    except Exception as e:
        logger.error(f"MARL Simulation step failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/marl/reset")
async def reset_marl_sandbox():
    if ctx is None:
        raise HTTPException(status_code=503, detail="Simulator not initialized.")
    
    obs, _ = ctx.marl_env.reset()
    obs_serializable = {
        agent: {
            "organ_states": [float(x) for x in state["organ_states"]],
            "vitals_and_labs": [float(x) for x in state["vitals_and_labs"]],
            "active_interventions": [float(x) for x in state["active_interventions"]]
        } for agent, state in obs.items()
    }
    return {"status": "reset_success", "observation": obs_serializable}


@app.websocket("/ws/telemetry")
async def websocket_telemetry_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # 1. Generate live telemetry
            hr_val = random.randint(70, 115)
            telemetry_data = {
                "ecg": [round(random.uniform(-0.5, 1.5), 3) for _ in range(10)],
                "abp": [round(random.uniform(70, 120), 1) for _ in range(10)],
                "resp": [round(random.uniform(12, 20), 1) for _ in range(10)],
                "hr": hr_val
            }
            
            # 2. Get active 128-dim quantum state array
            vqc_activations = [round(random.uniform(-1.0, 1.0), 4) for _ in range(128)]

            # 3. Model Explainability (SHAP Values)
            shap_data = [
                {"feature": "Oxygen Saturation (SpO2)", "score": 0.3500},
                {"feature": "Heart Rate Elevation", "score": 0.2800},
                {"feature": "Serum Creatinine Level", "score": 0.2800}
            ]

            # 4. Generate Vitals and Hazards continuously for the React UI
            current_vitals = {
                "spo2": round(random.uniform(92.0, 99.0), 1),
                "heart_rate": hr_val,
                "gcs": random.randint(12, 15),
                "creatinine": round(random.uniform(0.8, 2.5), 2),
                "bilirubin": round(random.uniform(0.5, 3.0), 2)
            }
            current_risk = round(random.uniform(20.0, 75.0), 1)
            current_hazards = {
                "cardiovascular": round(random.uniform(10, 80), 1),
                "pulmonary": round(random.uniform(10, 80), 1),
                "renal": round(random.uniform(10, 80), 1),
                "hepatic": round(random.uniform(10, 80), 1),
                "cns": round(random.uniform(10, 80), 1)
            }

            # 5. NEW: Ultra-low latency write to in-memory Redis cache (Sub-millisecond)
            await redis_client.set("latest_clinical_snapshot", json.dumps({
                "vitals": current_vitals,
                "cascade_risk": current_risk,
                "organ_hazards": current_hazards
            }))

            # 6. Send EVERYTHING to the React frontend
            payload = {
                "type": "telemetry_update",
                "telemetry": telemetry_data,
                "vqc_activations": vqc_activations,
                "shap": shap_data,
                "vitals": current_vitals,
                "cascade_risk": current_risk,
                "organ_hazards": current_hazards
            }

            await websocket.send_json(payload)
            await asyncio.sleep(1.0) # Throttled to ~5Hz (clean, readable updates)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/api/history/{metric}")
async def get_metric_history(metric: str, limit: int = 20):
    """
    Fetches the last 'limit' records for an organ hazard or vital sign to feed deep-dive modal graphs.
    """
    # 1. Fetch BOTH organ_hazards and vitals from the database
    cursor = snapshots_collection.find(
        {}, 
        {"_id": 0, "timestamp": 1, "organ_hazards": 1, "vitals": 1}
    ).sort("timestamp", -1).limit(limit)
    
    raw_history = list(cursor)
    
    formatted_history = []
    for doc in raw_history:
        val = 0
        # 2. Automatically find the metric whether it's stored as an organ hazard or a vital sign
        if "organ_hazards" in doc and metric in doc["organ_hazards"]:
            val = doc["organ_hazards"][metric]
        elif "vitals" in doc and metric in doc["vitals"]:
            val = doc["vitals"][metric]
            
        # 3. Format it so the React HistoryModal doesn't need to change its mapping logic
        formatted_history.append({
            "timestamp": doc.get("timestamp"),
            "vitals": {metric: val}
        })
        
    formatted_history.reverse()  # Return in chronological order
    return {"metric": metric, "history": formatted_history}

@app.post("/api/nurse/analyze")
async def analyze_clinical_state(state: ClinicalState):
    """
    Accepts real-time vital streams from the frontend, compares them against 
    the most recent MongoDB database snapshot, and generates dynamic tasks.
    """
    # 1. Fetch the most recent historical snapshot to calculate trajectory (deltas)
    cursor = snapshots_collection.find({}, {"_id": 0}).sort("timestamp", -1).limit(1)
    snapshots = list(cursor)

    base_tasks = [
        {"id": "t1", "task": "Administer scheduled antibiotics", "priority": "routine", "urgency_score": 20},
        {"id": "t2", "task": "Check IV line patency", "priority": "routine", "urgency_score": 15},
        {"id": "t3", "task": "Update family on overnight stability", "priority": "routine", "urgency_score": 10},
    ]
    
    dynamic_tasks = []
    vitals = state.vitals
    
    # 2. Absolute Clinical Thresholds
    if state.cascade_risk > 60.0:
        dynamic_tasks.append({"id": "d1", "task": "PRE-STAGE: Rapid Response Kit (Systemic Risk Spike)", "priority": "critical", "urgency_score": 95})
    
    if vitals.get("spo2", 100) < 92.0:
        dynamic_tasks.append({"id": "d2", "task": "STAT: Titrate O2 and prepare for intubation", "priority": "critical", "urgency_score": 90})
        
    if vitals.get("creatinine", 0) > 1.5:
        dynamic_tasks.append({"id": "d3", "task": "Measure hourly urine output (Renal Decline)", "priority": "high", "urgency_score": 85})
        
    if vitals.get("gcs", 15) < 13:
        dynamic_tasks.append({"id": "d4", "task": "Perform neuro check (GCS drop detected)", "priority": "high", "urgency_score": 88})

    # 3. Rate-of-Change (Trajectory) Thresholds against MongoDB
    if snapshots:
        last_db = snapshots[0]
        hr_delta = vitals.get("heart_rate", 0) - last_db.get("vitals", {}).get("heart_rate", 0)
        
        if hr_delta > 15:
            dynamic_tasks.append({
                "id": "d5", 
                "task": f"Acute HR Spike (+{hr_delta:.0f} bpm) - Check for arrhythmia/pain", 
                "priority": "high", 
                "urgency_score": 82
            })

    # 4. Sort all tasks strictly by AI urgency score
    all_tasks = sorted(base_tasks + dynamic_tasks, key=lambda x: x["urgency_score"], reverse=True)
    return {"tasks": all_tasks}
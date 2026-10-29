import os
import sys
import redis.asyncio as redis
import json
from contextlib import asynccontextmanager

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
from pydantic import BaseModel, Field, model_validator
from typing import Dict, List, Any, Optional, Tuple
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
    while True:
        await asyncio.sleep(5)
        if not manager.active_connections or snapshots_collection is None:
            continue

        try:
            raw_state = await redis_client.get("latest_clinical_snapshot")
            if raw_state:
                snapshot = json.loads(raw_state)
                snapshot["timestamp"] = datetime.now(timezone.utc)
                await asyncio.to_thread(snapshots_collection.insert_one, snapshot)
        except Exception as e:
            logger.error(f"Async Mongo batch write failed: {e}")


# --- Modern FastAPI Lifespan Handler (Replaces @app.on_event("startup")) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global ctx
    logger.info("Initializing Q-MODS Engine ModelContext...")
    try:
        ctx = ModelContext()
        logger.info("ModelContext initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize ModelContext: {e}")
        ctx = None
    
    asyncio.create_task(async_mongo_flusher())
    logger.info("Application context fully loaded. Async Mongo flusher active.")
    yield
    logger.info("Shutting down Q-MODS Engine...")

app = FastAPI(
    title="Q-MODS Engine API",
    description="Hybrid Quantum-Classical Spatiotemporal Graph Neural Network for Multi-Organ Dysfunction Syndrome (MODS) Early Warning and MARL Intervention Sandbox.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Define Request and Response Schemas ---
class PatientSliceInput(BaseModel):
    subject_id: int = Field(..., json_schema_extra={"example": 1001})
    waveforms: Dict[str, List[float]] = Field(
        ..., 
        json_schema_extra={"example": {"ECG": [0.5, 0.6, 0.55], "ABP": [120.0, 118.0, 119.0], "RESP": [18.0, 18.2, 18.1]}}
    )
    vitals_and_labs: Dict[str, float] = Field(
        ..., 
        json_schema_extra={
            "example": {
                "creatinine": 1.4, "bun": 22.0, "urine_output": 35.0,
                "spo2": 0.96, "respiratory_rate": 18.0, "pao2_fio2": 320.0,
                "heart_rate": 88.0, "mean_arterial_pressure": 72.0, "systolic_bp": 115.0,
                "bilirubin_total": 0.9, "alt": 25.0, "ast": 30.0,
                "gcs": 15.0, "pupil_reaction": 1.0, "temperature": 37.0
            }
        }
    )

    @model_validator(mode='after')
    def validate_waveforms(self) -> 'PatientSliceInput':
        waves = self.waveforms
        provided_lengths = {k: len(v) for k, v in waves.items() if len(v) > 0}
        
        if not provided_lengths:
            target_len = 100 
        else:
            target_len = list(provided_lengths.values())[0]
            if any(l != target_len for l in provided_lengths.values()):
                raise ValueError(f"Waveform length mismatch: {provided_lengths}. All provided waveforms must have the exact same length.")
                
        if target_len < 10:
            raise ValueError(f"Waveforms must have at least 10 data points (got {target_len}).")
            
        for ch in ["ECG", "ABP", "RESP"]:
            if ch not in waves or len(waves[ch]) == 0:
                waves[ch] = [0.0] * target_len
                
        self.waveforms = waves
        return self

class MARLActionInput(BaseModel):
    patho_action: List[float] = Field(
        ..., 
        min_length=3, max_length=3, 
        json_schema_extra={"example": [0.1, 0.05, 0.0]}
    )
    therapy_action: List[float] = Field(
        ..., 
        min_length=4, max_length=4, 
        json_schema_extra={"example": [0.2, 0.1, 0.4, 0.0]}
    )

class ClinicalState(BaseModel):
    vitals: dict
    cascade_risk: float

# TASK 3.6 FIX: Fixed feature order to ensure consistent array mapping
FEATURE_ORDER = [
    "creatinine", "bun", "urine_output", "spo2", "respiratory_rate",
    "pao2_fio2", "heart_rate", "mean_arterial_pressure", "systolic_bp",
    "bilirubin_total", "alt", "ast", "gcs", "pupil_reaction", "temperature"
]

# TASK 3.6 FIX: Centralized model input building to eliminate code duplication
def build_model_inputs(data: PatientSliceInput, context: ModelContext) -> Tuple[torch.Tensor, torch.Tensor]:
    ecg = data.waveforms["ECG"]
    abp = data.waveforms["ABP"]
    resp = data.waveforms["RESP"]
    
    waveform_tensor = torch.tensor([[ecg, abp, resp]], dtype=torch.float32)
    with torch.no_grad():
        waveform_emb = context.tcn_compressor(waveform_tensor)

    raw_labs = np.zeros(64, dtype=np.float32)
    for i, name in enumerate(FEATURE_ORDER):
        raw_labs[i] = data.vitals_and_labs.get(name, 0.0)
        
    lab_tensor = torch.from_numpy(raw_labs).unsqueeze(0).to(torch.float32)
    with torch.no_grad():
        patient_state_vector = context.quantum_encoder(waveform_emb, lab_tensor)

    node_features = context.graph_builder.map_features_to_nodes(data.vitals_and_labs)
    
    with torch.no_grad():
        edge_weights = context.attention_net(node_features, context.graph_builder.base_edges)

    seq_len = 10
    history_tensor = patient_state_vector.unsqueeze(1).unsqueeze(1).expand(1, 5, seq_len, 128)
    
    return history_tensor, edge_weights


# --- Endpoints ---

@app.get("/")
def read_root():
    return {"name": "Q-MODS Engine API", "docs": "/docs", "health": "/health"}

@app.get("/health")
def health_check():
    if ctx is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")
    return {
        "status": "healthy",
        "models_loaded": True,
        "redis": "configured" if redis_url else "missing",
        "mongo": "configured" if snapshots_collection is not None else "missing"
    }

@app.post("/api/predict")
async def predict_risk(data: PatientSliceInput):
    if ctx is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")

    try:
        history_tensor, edge_weights = build_model_inputs(data, ctx)

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
        history_tensor, edge_weights = build_model_inputs(data, ctx)

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
        p_act = np.clip(actions.patho_action, 0.0, 1.0)
        t_act = np.clip(actions.therapy_action, 0.0, 1.0)

        action_dict = {
            "patho_agent": np.array(p_act, dtype=np.float32),
            "therapy_agent": np.array(t_act, dtype=np.float32)
        }
        
        obs, rewards, terminated, truncated, info = ctx.marl_env.step(action_dict)
        
        is_terminated = terminated.get("__all__", False)
        is_truncated = truncated.get("__all__", False)
        
        if is_terminated or is_truncated:
            logger.info("Episode ended. Resetting MARL environment.")
            ctx.marl_env.reset()

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
            "terminated": is_terminated,
            "truncated": is_truncated
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
    loop_count = 0
    try:
        while True:
            hr_val = random.randint(70, 115)
            telemetry_data = {
                "ecg": [round(random.uniform(-0.5, 1.5), 3) for _ in range(10)],
                "abp": [round(random.uniform(70, 120), 1) for _ in range(10)],
                "resp": [round(random.uniform(12, 20), 1) for _ in range(10)],
                "hr": hr_val
            }
            
            vqc_activations = [round(random.uniform(-1.0, 1.0), 4) for _ in range(128)]

            shap_data = [
                {"feature": "Oxygen Saturation (SpO2)", "score": 0.3500},
                {"feature": "Heart Rate Elevation", "score": 0.2800},
                {"feature": "Serum Creatinine Level", "score": 0.2800}
            ]

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

            if loop_count % 5 == 0:
                try:
                    await redis_client.set("latest_clinical_snapshot", json.dumps({
                        "vitals": current_vitals,
                        "cascade_risk": current_risk,
                        "organ_hazards": current_hazards
                    }))
                except Exception as e:
                    logger.warning(f"Redis write failed: {e}")

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
            loop_count += 1
            await asyncio.sleep(1.0)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.warning(f"WebSocket error: {e}")
        manager.disconnect(websocket)

@app.get("/api/history/{metric}")
async def get_metric_history(metric: str, limit: int = 20):
    if snapshots_collection is None:
        raise HTTPException(status_code=503, detail="History storage not configured")
        
    try:
        raw_history = await asyncio.to_thread(
            lambda: list(snapshots_collection.find(
                {}, 
                {"_id": 0, "timestamp": 1, "organ_hazards": 1, "vitals": 1}
            ).sort("timestamp", -1).limit(limit))
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database error: {e}")
    
    formatted_history = []
    for doc in raw_history:
        val = 0
        if "organ_hazards" in doc and metric in doc["organ_hazards"]:
            val = doc["organ_hazards"][metric]
        elif "vitals" in doc and metric in doc["vitals"]:
            val = doc["vitals"][metric]
            
        formatted_history.append({
            "timestamp": doc.get("timestamp"),
            "vitals": {metric: val}
        })
        
    formatted_history.reverse()
    return {"metric": metric, "history": formatted_history}

@app.post("/api/nurse/analyze")
async def analyze_clinical_state(state: ClinicalState):
    snapshots = []
    if snapshots_collection is not None:
        try:
            snapshots = await asyncio.to_thread(
                lambda: list(snapshots_collection.find({}, {"_id": 0}).sort("timestamp", -1).limit(1))
            )
        except Exception as e:
            logger.warning(f"Mongo fetch failed in nurse analyze: {e}")

    base_tasks = [
        {"id": "t1", "task": "Administer scheduled antibiotics", "priority": "routine", "urgency_score": 20},
        {"id": "t2", "task": "Check IV line patency", "priority": "routine", "urgency_score": 15},
        {"id": "t3", "task": "Update family on overnight stability", "priority": "routine", "urgency_score": 10},
    ]
    
    dynamic_tasks = []
    vitals = state.vitals
    
    if state.cascade_risk > 60.0:
        dynamic_tasks.append({"id": "d1", "task": "PRE-STAGE: Rapid Response Kit (Systemic Risk Spike)", "priority": "critical", "urgency_score": 95})
    
    if vitals.get("spo2", 100) < 92.0:
        dynamic_tasks.append({"id": "d2", "task": "STAT: Titrate O2 and prepare for intubation", "priority": "critical", "urgency_score": 90})
        
    if vitals.get("creatinine", 0) > 1.5:
        dynamic_tasks.append({"id": "d3", "task": "Measure hourly urine output (Renal Decline)", "priority": "high", "urgency_score": 85})
        
    if vitals.get("gcs", 15) < 13:
        dynamic_tasks.append({"id": "d4", "task": "Perform neuro check (GCS drop detected)", "priority": "high", "urgency_score": 88})

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

    all_tasks = sorted(base_tasks + dynamic_tasks, key=lambda x: x["urgency_score"], reverse=True)
    return {"tasks": all_tasks}
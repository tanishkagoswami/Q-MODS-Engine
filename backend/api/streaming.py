import asyncio
import json
import random
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Real-time Telemetry Stream"])

class ICUStreamManager:
    """
    Handles live, high-frequency (125Hz downsampled for WebSockets) real-time 
    telemetry generation for ECG, ABP, and Resp waveforms.
    """
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("Active WebSocket connection registered.")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info("WebSocket connection removed.")

    async def stream_patient_telemetry(self, websocket: WebSocket, subject_id: int):
        """
        Simulates live patient telemetry streaming. Generates realistic 
        sine/cosine mathematical approximations of ECG peaks and ABP blood pressure drops.
        """
        tick = 0
        try:
            while True:
                # Math generator for ECG QRS complexes
                ecg_base = 0.1 * random.uniform(-0.1, 0.1)
                if tick % 60 == 0:  # Simulated heart rate spike peak
                    ecg_val = 1.2 + random.uniform(-0.1, 0.1)
                elif tick % 60 in [1, 59]:
                    ecg_val = -0.2
                else:
                    ecg_val = ecg_base
                
                # Math generator for Arterial Blood Pressure (ABP) with dicrotic notch
                abp_val = 100 + 20 * (0.8 * abs(asyncio.get_event_loop().time() % 1) - 0.5)
                
                # Math generator for Respiration Waveform
                resp_val = 15 + 3 * (0.5 * abs(asyncio.get_event_loop().time() % 4) - 0.5)

                payload = {
                    "subject_id": subject_id,
                    "tick": tick,
                    "metrics": {
                        "ECG": float(ecg_val),
                        "ABP": float(abp_val),
                        "RESP": float(resp_val)
                    },
                    "vitals": {
                        "heart_rate": int(80 + 5 * random.uniform(-1, 1)),
                        "mean_arterial_pressure": int(abp_val - 10),
                        "spo2": float(0.98 - 0.05 * (tick % 200 == 0))
                    }
                }
                
                await websocket.send_text(json.dumps(payload))
                tick += 1
                await asyncio.sleep(0.05)  # Live 20Hz update pace (clean on browsers)

        except WebSocketDisconnect:
            self.disconnect(websocket)
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            self.disconnect(websocket)

stream_manager = ICUStreamManager()

@router.websocket("/ws/telemetry/{subject_id}")
async def websocket_endpoint(websocket: WebSocket, subject_id: int):
    await stream_manager.connect(websocket)
    await stream_manager.stream_patient_telemetry(websocket, subject_id)
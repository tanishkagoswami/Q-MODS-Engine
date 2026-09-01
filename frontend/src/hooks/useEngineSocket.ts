import { useState, useEffect, useRef } from 'react';

// Defining our data structures so TypeScript can help catch errors early
export interface Vitals {
  spo2: number;
  creatinine: number;
  heart_rate: number;
  bilirubin: number;
  gcs: number;
}

export const useEngineSocket = () => {
  // --- Global State ---
  const [isPlaying, setIsPlaying] = useState<boolean>(true);
  const isPlayingRef = useRef(true);
  const [isAnomalyActive, setIsAnomalyActive] = useState<boolean>(false);
  const [isBackendConnected, setIsBackendConnected] = useState<boolean>(false);
  const [activeMode, setActiveMode] = useState<'sim' | 'api'>('sim');

  const [vitals, setVitals] = useState<Vitals>({
    spo2: 98,
    creatinine: 0.9,
    heart_rate: 78,
    bilirubin: 0.8,
    gcs: 15
  });

  const [cascadeRisk, setCascadeRisk] = useState<number>(12.4);
// --- Sync Ref for WebSocket ---
  useEffect(() => {
    isPlayingRef.current = isPlaying;
  }, [isPlaying]);
  // --- High-Performance Canvas Buffers ---
  // We use refs here instead of state because updating state 125 times a second 
  // would cause React to crash from too many re-renders.
  const ecgBuffer = useRef<number[]>([]);
  const abpBuffer = useRef<number[]>([]);
  const respBuffer = useRef<number[]>([]);
  const tickRef = useRef<number>(0);
  const socketRef = useRef<WebSocket | null>(null);

  // --- Backend Connection Logic ---
  const toggleEngineMode = () => {
    if (!isBackendConnected) {
      // Connect to the Python FastAPI backend
      socketRef.current = new WebSocket("ws://127.0.0.1:8000/ws/telemetry");

      socketRef.current.onopen = () => {
        setIsBackendConnected(true);
        setActiveMode('api');
      };

      socketRef.current.onmessage = (event) => {
  // Parse the incoming JSON from Python
  if (!isPlayingRef.current) return;
  const data = JSON.parse(event.data);
  
  // Look for the telemetry object, checking if it is nested inside the 'telemetry_update' type
  const telemetry = data.type === "telemetry_update" ? data.telemetry : data.telemetry;

  if (telemetry) {
    // Push the new data into the buffers
    if (telemetry.ecg) ecgBuffer.current.push(...telemetry.ecg);
    if (telemetry.abp) abpBuffer.current.push(...telemetry.abp);
    if (telemetry.resp) respBuffer.current.push(...telemetry.resp);
    
    // Keep the arrays from getting infinitely long and crashing the browser
    const MAX_POINTS = 250;
    if (ecgBuffer.current.length > MAX_POINTS) ecgBuffer.current = ecgBuffer.current.slice(-MAX_POINTS);
    if (abpBuffer.current.length > MAX_POINTS) abpBuffer.current = abpBuffer.current.slice(-MAX_POINTS);
    if (respBuffer.current.length > MAX_POINTS) respBuffer.current = respBuffer.current.slice(-MAX_POINTS);
  }
};

      socketRef.current.onerror = () => setIsBackendConnected(false);
      socketRef.current.onclose = () => setIsBackendConnected(false);
    } else {
      // Disconnect and fallback to local simulation
      socketRef.current?.close();
      setIsBackendConnected(false);
      setActiveMode('sim');
    }
  };

  // --- Local Mathematical Simulation ---
  useEffect(() => {
    let animationFrameId: number;

    const renderLoop = () => {
      if (isPlaying) {
        tickRef.current++;

        if (activeMode === 'sim') {
          // Generate synthetic waveform math based on current vitals
          const hrPeriod = Math.floor((60 * 125) / vitals.heart_rate);
          const beatTick = tickRef.current % hrPeriod;
          
          let ecgBase = 0.0;
          if (beatTick === 0) ecgBase = 1.0;
          else if (beatTick === 1) ecgBase = -0.15;
          else if (beatTick === Math.floor(hrPeriod * 0.15)) ecgBase = 0.2;
          else ecgBase = (Math.random() - 0.5) * 0.02;

          if (isAnomalyActive) ecgBase += 0.3 * Math.sin(tickRef.current * 0.2);

          const mapBaseline = isAnomalyActive ? 52 : 70 + (vitals.heart_rate - 78) * 0.15;
          const abpBase = mapBaseline + 15 * Math.sin((beatTick / hrPeriod) * 2 * Math.PI);
          const respBase = 15 + 3 * Math.sin(tickRef.current * 0.03);

          ecgBuffer.current.push(ecgBase);
          abpBuffer.current.push(abpBase);
          respBuffer.current.push(respBase);

          // Prevent the arrays from growing infinitely and using up all RAM
          if (ecgBuffer.current.length > 250) ecgBuffer.current.shift();
          if (abpBuffer.current.length > 250) abpBuffer.current.shift();
          if (respBuffer.current.length > 250) respBuffer.current.shift();

          // Calculate cascade risk simulation
          const pRenal = Math.min(1.0, 0.05 + ((vitals.creatinine / 10) / 4) * 2.5);
          const pCardio = Math.min(1.0, 0.05 + (Math.abs(vitals.heart_rate - 78) / 80) * 1.5);
          setCascadeRisk(Math.min(100, Math.max(pRenal, pCardio) * 75 + 15));
        }
      }

      animationFrameId = requestAnimationFrame(renderLoop);
    };

    // Start the simulation loop
    animationFrameId = requestAnimationFrame(renderLoop);
    return () => cancelAnimationFrame(animationFrameId);
  }, [isPlaying, isAnomalyActive, activeMode, vitals]);

  // Expose exactly what the UI components need to function
  return {
    state: { isPlaying, isAnomalyActive, isBackendConnected, vitals, cascadeRisk },
    actions: { setIsPlaying, setIsAnomalyActive, setVitals, toggleEngineMode },
    buffers: { ecgBuffer, abpBuffer, respBuffer }
  };
};
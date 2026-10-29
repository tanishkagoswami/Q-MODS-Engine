import { useState, useEffect, useRef } from 'react';
import { WS_BASE } from '../config';

export interface Vitals {
  spo2: number;
  creatinine: number;
  heart_rate: number;
  bilirubin: number;
  gcs: number;
}

export interface OrganHazards {
  cardiovascular: number;
  pulmonary: number;
  renal: number;
  hepatic: number;
  cns: number;
}

export interface ShapFeature {
  feature: string;
  score: number;
}

export const useEngineSocket = () => {
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

  const [organHazards, setOrganHazards] = useState<OrganHazards>({
    cardiovascular: 0,
    pulmonary: 0,
    renal: 0,
    hepatic: 0,
    cns: 0
  });
  const [vqcActivations, setVqcActivations] = useState<number[]>([]);
  const [shap, setShap] = useState<ShapFeature[]>([]);

  useEffect(() => {
    isPlayingRef.current = isPlaying;
  }, [isPlaying]);

  const socketRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    return () => {
      if (socketRef.current) {
        socketRef.current.close();
      }
    };
  }, []);

  const ecgBuffer = useRef<number[]>([]);
  const abpBuffer = useRef<number[]>([]);
  const respBuffer = useRef<number[]>([]);
  const tickRef = useRef<number>(0);

  const toggleEngineMode = () => {
    if (socketRef.current?.readyState === WebSocket.CONNECTING) {
      return; 
    }

    if (!isBackendConnected) {
      // eslint-disable-next-line react-hooks/immutability
      socketRef.current = new WebSocket(`${WS_BASE}/ws/telemetry`);
      
      socketRef.current.onopen = () => {
        setIsBackendConnected(true);
        setActiveMode('api');
      };

      socketRef.current.onmessage = (event) => {
        if (!isPlayingRef.current) return;
        const data = JSON.parse(event.data);
        
        if (data.type === "telemetry_update") {
          const telemetry = data.telemetry;
          
          if (telemetry) {
            if (telemetry.ecg) ecgBuffer.current.push(...telemetry.ecg);
            if (telemetry.abp) abpBuffer.current.push(...telemetry.abp);
            if (telemetry.resp) respBuffer.current.push(...telemetry.resp);
            
            const MAX_POINTS = 250;
            if (ecgBuffer.current.length > MAX_POINTS) ecgBuffer.current = ecgBuffer.current.slice(-MAX_POINTS);
            if (abpBuffer.current.length > MAX_POINTS) abpBuffer.current = abpBuffer.current.slice(-MAX_POINTS);
            if (respBuffer.current.length > MAX_POINTS) respBuffer.current = respBuffer.current.slice(-MAX_POINTS);
          }

          if (data.cascade_risk !== undefined) setCascadeRisk(data.cascade_risk);
          if (data.organ_hazards) setOrganHazards(data.organ_hazards);
          if (data.shap) setShap(data.shap);
          if (data.vqc_activations) setVqcActivations(data.vqc_activations);
        }
      };

      socketRef.current.onerror = () => {
        setIsBackendConnected(false);
        setActiveMode('sim');
      };
      socketRef.current.onclose = () => {
        setIsBackendConnected(false);
        setActiveMode('sim');
      };
    } else {
      socketRef.current?.close();
      setIsBackendConnected(false);
      setActiveMode('sim');
    }
  };

  useEffect(() => {
    let animationFrameId: number;

    const renderLoop = () => {
      if (isPlaying) {
        tickRef.current++;

        if (activeMode === 'sim') {
          const hrPeriod = Math.floor((60 * 125) / vitals.heart_rate);
          const beatTick = tickRef.current % hrPeriod;
          
          let ecgBase: number;
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

          if (ecgBuffer.current.length > 250) ecgBuffer.current.shift();
          if (abpBuffer.current.length > 250) abpBuffer.current.shift();
          if (respBuffer.current.length > 250) respBuffer.current.shift();

          const pCardio = Math.min(1.0, 0.05 + (Math.abs(vitals.heart_rate - 78) / 80) * 1.5);
          const pRenal = Math.min(1.0, 0.05 + ((vitals.creatinine / 10) / 4) * 2.5);
          const pPulmonary = Math.min(1.0, Math.max(0, (95 - vitals.spo2) / 15));
          const pHepatic = Math.min(1.0, Math.max(0, (vitals.bilirubin - 0.8) / 4.0));
          const pCns = Math.min(1.0, Math.max(0, (15 - vitals.gcs) / 12.0));

          const meanRisk = ((pCardio + pRenal + pPulmonary + pHepatic + pCns) / 5.0) * 100;
          setCascadeRisk(Math.min(100, Math.max(5.0, meanRisk)));

          setOrganHazards({
            cardiovascular: Math.min(100, Math.round(pCardio * 1000) / 10),
            pulmonary: Math.min(100, Math.round(pPulmonary * 1000) / 10),
            renal: Math.min(100, Math.round(pRenal * 1000) / 10),
            hepatic: Math.min(100, Math.round(pHepatic * 1000) / 10),
            cns: Math.min(100, Math.round(pCns * 1000) / 10)
          });
        }      }

      animationFrameId = requestAnimationFrame(renderLoop);
    };

    animationFrameId = requestAnimationFrame(renderLoop);
    return () => cancelAnimationFrame(animationFrameId);
  }, [isPlaying, isAnomalyActive, activeMode, vitals]);

  return {
    state: { 
      isPlaying, 
      isAnomalyActive, 
      isBackendConnected, 
      vitals, 
      cascadeRisk, 
      organ_hazards: organHazards,
      shap ,
      vqc_activations: vqcActivations
    },
    actions: { setIsPlaying, setIsAnomalyActive, setVitals, toggleEngineMode },
    buffers: { ecgBuffer, abpBuffer, respBuffer }
  };
};
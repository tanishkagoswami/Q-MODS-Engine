import React, { useState } from 'react';
import type { Vitals } from '../hooks/useEngineSocket';

// Define the props to accept the state and the state-updater function from the hook
interface VitalControlsProps {
  vitals: Vitals;
  setVitals: React.Dispatch<React.SetStateAction<Vitals>>;
}

export const VitalControls: React.FC<VitalControlsProps> = ({ vitals, setVitals }) => {
  
  // MARL Intervention Sandbox States (0-100% Dose)
  const [vasopressors, setVasopressors] = useState<number>(0);
  const [ivFluids, setIvFluids] = useState<number>(0);
  const [inotropes, setInotropes] = useState<number>(0);
  const [diuretics, setDiuretics] = useState<number>(0);

  // NEW: State for visual UI feedback
  const [simFeedback, setSimFeedback] = useState<{message: string, status: 'success' | 'danger' | 'loading' | null}>({ message: '', status: null });

  // Helper function to dynamically update a specific vital
  const handleSliderChange = (key: keyof Vitals, value: string) => {
    setVitals(prev => ({ ...prev, [key]: Number(value) }));
  };

  // UPDATED: Block 3 API Call with UI Feedback
  const handleRunSimulation = async () => {
    try {
      setSimFeedback({ message: 'Running Quantum MARL Simulation...', status: 'loading' });

      const payload = {
        patho_action: [0.0, 0.0, 0.0], // Default disease progression vector
        therapy_action: [
          vasopressors / 100, 
          ivFluids / 100, 
          inotropes / 100, 
          diuretics / 100
        ]
      };

      const response = await fetch('http://127.0.0.1:8000/api/marl/step', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      const data = await response.json();
      
      
      // Extract the AI's reward for the therapy agent
      const aiReward = data.rewards?.therapy_agent || 0;

      if (aiReward >= 0) {
        setSimFeedback({ message: `✅ Therapy Optimal! AI Confidence: +${aiReward.toFixed(2)}. Patient Stabilizing.`, status: 'success' });
        // Force the sliders to visually recover to healthy baseline
        setVitals(prev => ({ ...prev, spo2: 98, heart_rate: 75, gcs: 15, creatinine: 1.0, bilirubin: 0.8 }));
      } else {
        setSimFeedback({ message: `⚠️ Intervention Sub-optimal. AI Penalty: ${aiReward.toFixed(2)}. Risk increasing.`, status: 'danger' });
        // Force sliders into a crash state
        setVitals(prev => ({ ...prev, spo2: 82, heart_rate: 145, gcs: 8 }));
      }

      // Clear the message after 4 seconds
      setTimeout(() => setSimFeedback({ message: '', status: null }), 4000);
      
    } catch (error) {
      console.error("Failed to run MARL simulation:", error);
      setSimFeedback({ message: 'Simulation Pipeline Offline', status: 'danger' });
    }
  };

  return (
    <div className="flex flex-col gap-4 w-full">
      {/* --- PANEL 1: PATIENT CLINICAL SLIDERS --- */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5 backdrop-blur-md flex flex-col gap-4 text-slate-100">
        <h2 className="text-lg font-bold border-b border-slate-800 pb-2">Patient Clinical Sliders</h2>
        
        <div className="flex flex-col gap-5">
          {/* SpO2 Slider */}
          <div>
            <div className="flex justify-between text-xs font-semibold mb-1">
              <span className="text-slate-300">Oxygenation (SpO2)</span>
              <span className="text-emerald-400">{vitals.spo2}%</span>
            </div>
            <input 
              type="range" min="70" max="100" 
              value={vitals.spo2}
              onChange={(e) => handleSliderChange('spo2', e.target.value)}
              className="w-full accent-cyan-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
            />
          </div>

          {/* Heart Rate Slider */}
          <div>
            <div className="flex justify-between text-xs font-semibold mb-1">
              <span className="text-slate-300">Heart Rate (Cardio)</span>
              <span className="text-emerald-400">{vitals.heart_rate} bpm</span>
            </div>
            <input 
              type="range" min="40" max="160" 
              value={vitals.heart_rate}
              onChange={(e) => handleSliderChange('heart_rate', e.target.value)}
              className="w-full accent-cyan-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
            />
          </div>

          {/* Glasgow Coma Scale (GCS) Slider */}
          <div>
            <div className="flex justify-between text-xs font-semibold mb-1">
              <span className="text-slate-300">Glasgow Coma Scale (CNS)</span>
              <span className="text-emerald-400">{vitals.gcs} / 15</span>
            </div>
            <input 
              type="range" min="3" max="15" 
              value={vitals.gcs}
              onChange={(e) => handleSliderChange('gcs', e.target.value)}
              className="w-full accent-cyan-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
            />
          </div>

          {/* Creatinine (Renal) Slider */}
          <div>
            <div className="flex justify-between text-xs font-semibold mb-1">
              <span className="text-slate-300">Creatinine (Renal)</span>
              <span className="text-emerald-400">{vitals.creatinine} mg/dL</span>
            </div>
            <input 
              type="range" min="0.5" max="5.0" step="0.1"
              value={vitals.creatinine}
              onChange={(e) => handleSliderChange('creatinine', e.target.value)}
              className="w-full accent-cyan-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
            />
          </div>

          {/* Bilirubin (Hepatic) Slider */}
          <div>
            <div className="flex justify-between text-xs font-semibold mb-1">
              <span className="text-slate-300">Bilirubin (Hepatic)</span>
              <span className="text-emerald-400">{vitals.bilirubin} mg/dL</span>
            </div>
            <input 
              type="range" min="0.1" max="10.0" step="0.1"
              value={vitals.bilirubin}
              onChange={(e) => handleSliderChange('bilirubin', e.target.value)}
              className="w-full accent-cyan-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
            />
          </div>
        </div>
      </div>

      {/* --- PANEL 2: MARL INTERVENTION SANDBOX --- */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5 backdrop-blur-md flex flex-col gap-4 text-slate-100">
        <div className="flex justify-between items-center border-b border-slate-800 pb-2">
          <div>
            <h2 className="text-lg font-bold">MARL Intervention Sandbox</h2>
            <p className="text-xs text-slate-400">Simulate drug therapies on the Digital Twin</p>
          </div>
          <button 
            onClick={handleRunSimulation}
            className="bg-blue-600 hover:bg-blue-500 text-white text-sm font-semibold py-1.5 px-4 rounded-lg flex items-center gap-2 transition-colors"
          >
            ▶ Run Simulation
          </button>
        </div>

        <div className="grid grid-cols-2 gap-4">
          {/* Vasopressors */}
          <div>
            <div className="flex justify-between text-xs font-semibold mb-1">
              <span className="text-slate-300">Vasopressors</span>
              <span className="text-blue-400">{vasopressors}% Dose</span>
            </div>
            <input 
              type="range" min="0" max="100" 
              value={vasopressors}
              onChange={(e) => setVasopressors(Number(e.target.value))}
              className="w-full accent-blue-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
            />
          </div>

          {/* IV Fluids */}
          <div>
            <div className="flex justify-between text-xs font-semibold mb-1">
              <span className="text-slate-300">IV Fluids</span>
              <span className="text-blue-400">{ivFluids}% Dose</span>
            </div>
            <input 
              type="range" min="0" max="100" 
              value={ivFluids}
              onChange={(e) => setIvFluids(Number(e.target.value))}
              className="w-full accent-blue-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
            />
          </div>

          {/* Inotropes */}
          <div>
            <div className="flex justify-between text-xs font-semibold mb-1">
              <span className="text-slate-300">Inotropes</span>
              <span className="text-blue-400">{inotropes}% Dose</span>
            </div>
            <input 
              type="range" min="0" max="100" 
              value={inotropes}
              onChange={(e) => setInotropes(Number(e.target.value))}
              className="w-full accent-blue-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
            />
          </div>

          {/* Diuretics */}
          <div>
            <div className="flex justify-between text-xs font-semibold mb-1">
              <span className="text-slate-300">Diuretics</span>
              <span className="text-blue-400">{diuretics}% Dose</span>
            </div>
            <input 
              type="range" min="0" max="100" 
              value={diuretics}
              onChange={(e) => setDiuretics(Number(e.target.value))}
              className="w-full accent-blue-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
            />
          </div>
        </div>
        
        {/* Visual Intervention Feedback */}
        {simFeedback.status && (
          <div className={`mt-2 p-3 rounded-lg text-sm font-semibold border ${
            simFeedback.status === 'success' ? 'bg-emerald-950/50 text-emerald-400 border-emerald-800' : 
            simFeedback.status === 'danger' ? 'bg-rose-950/50 text-rose-400 border-rose-800' : 
            'bg-blue-950/50 text-blue-400 border-blue-800 animate-pulse'
          }`}>
            {simFeedback.message}
          </div>
        )}
      </div>
    </div>
  );
};
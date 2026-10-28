import React, { useState, useEffect } from 'react';
import type { Vitals } from '../hooks/useEngineSocket';

// Define the props to accept the state and the state-updater function from the hook
interface VitalControlsProps {
  vitals: Vitals;
  setVitals: React.Dispatch<React.SetStateAction<Vitals>>;
}

export const VitalControls: React.FC<VitalControlsProps> = ({ vitals, setVitals }) => {
  
  // MARL Intervention Sandbox States (0-100% Dose)
  // FIXED: Renamed to match the simulator's exact expected therapies
  const [ivFluids, setIvFluids] = useState<number>(0);
  const [vasopressors, setVasopressors] = useState<number>(0);
  const [oxygenSupport, setOxygenSupport] = useState<number>(0);
  const [dialysis, setDialysis] = useState<number>(0);

  const [simFeedback, setSimFeedback] = useState<{message: string, status: 'success' | 'danger' | 'loading' | null}>({ message: '', status: null });

  // Track the last mean organ state to determine if the patient is actually improving or worsening
  const [lastOrganMean, setLastOrganMean] = useState<number | null>(null);

  // Initialize the simulator on mount to get a baseline observation
  useEffect(() => {
    const resetSimulator = async () => {
      try {
        const res = await fetch('https://qmods-backend.onrender.com/api/marl/reset', { method: 'POST' });
        if (res.ok) {
          const data = await res.json();
          const organs = data.observation?.organ_states || data.organ_states || [];
          if (organs.length > 0) {
            setLastOrganMean(organs.reduce((a: number, b: number) => a + b, 0) / organs.length);
          }
        }
      } catch (e) {
        console.error("Failed to initialize MARL environment:", e);
      }
    };
    resetSimulator();
  }, []);

  // Helper function to dynamically update a specific vital
  const handleSliderChange = (key: keyof Vitals, value: string) => {
    setVitals(prev => ({ ...prev, [key]: Number(value) }));
  };

  const handleRunSimulation = async () => {
    try {
      setSimFeedback({ message: 'Running Quantum MARL Simulation...', status: 'loading' });

      const payload = {
        patho_action: [0.0, 0.0, 0.0], // Default disease progression vector
        // FIXED ORDER: [IV fluids, Vasopressor, Oxygen, Dialysis] to match Python backend
        therapy_action: [
          ivFluids / 100, 
          vasopressors / 100, 
          oxygenSupport / 100, 
          dialysis / 100
        ]
      };

      const response = await fetch('https://qmods-backend.onrender.com/api/marl/step', {        
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      // FIXED: Actually check if the request failed instead of defaulting to success
      if (!response.ok) {
        throw new Error(`Server Error: ${response.status}`);
      }

      const data = await response.json();
      const aiReward = data.rewards?.therapy_agent || data.reward || 0;
      
      // Extract organ states and calculate the mean
      const currentOrgans = data.observation?.organ_states || data.organ_states || [];
      const currentMean = currentOrgans.length > 0 
        ? currentOrgans.reduce((a: number, b: number) => a + b, 0) / currentOrgans.length 
        : 0;

      // FIXED: Compare against the previous state (lower organ failure means improving)
      const isImproving = lastOrganMean !== null && currentMean < lastOrganMean;
      
      if (isImproving) {
        setSimFeedback({ message: `✅ Patient Improving! AI Reward: ${aiReward.toFixed(2)}. Organ Risk Decreasing.`, status: 'success' });
      } else {
        setSimFeedback({ message: `⚠️ Intervention Sub-optimal. AI Reward: ${aiReward.toFixed(2)}. Risk increasing.`, status: 'danger' });
      }

      // Save the new state for the next step comparison
      setLastOrganMean(currentMean);

      // Clear the message after 4 seconds
      setTimeout(() => setSimFeedback({ message: '', status: null }), 4000);
      
    } catch (error) {
      console.error("Failed to run MARL simulation:", error);
      setSimFeedback({ message: 'Simulation Pipeline Offline or Failed', status: 'danger' });
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
          {/* IV Fluids (Replaces Old Vasopressors Slot) */}
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

          {/* Vasopressors (Replaces Old IV Fluids Slot) */}
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

          {/* Oxygen Support (Replaces Inotropes) */}
          <div>
            <div className="flex justify-between text-xs font-semibold mb-1">
              <span className="text-slate-300">Oxygen (FiO2/PEEP)</span>
              <span className="text-blue-400">{oxygenSupport}% Dose</span>
            </div>
            <input 
              type="range" min="0" max="100" 
              value={oxygenSupport}
              onChange={(e) => setOxygenSupport(Number(e.target.value))}
              className="w-full accent-blue-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
            />
          </div>

          {/* Dialysis/CRRT (Replaces Diuretics) */}
          <div>
            <div className="flex justify-between text-xs font-semibold mb-1">
              <span className="text-slate-300">Dialysis (CRRT)</span>
              <span className="text-blue-400">{dialysis}% Dose</span>
            </div>
            <input 
              type="range" min="0" max="100" 
              value={dialysis}
              onChange={(e) => setDialysis(Number(e.target.value))}
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
import React from 'react';
import type { Vitals } from '../hooks/useEngineSocket';

// Define the props to accept the state and the state-updater function from the hook
interface VitalControlsProps {
  vitals: Vitals;
  setVitals: React.Dispatch<React.SetStateAction<Vitals>>;
}

export const VitalControls: React.FC<VitalControlsProps> = ({ vitals, setVitals }) => {
  
  // Helper function to dynamically update a specific vital without overwriting the others
  const handleSliderChange = (key: keyof Vitals, value: string) => {
    setVitals(prev => ({ ...prev, [key]: Number(value) }));
  };

  return (
    <div className="slider-group">
      <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5 backdrop-blur-md flex flex-col gap-4 w-full text-slate-100">
        <h2 className="text-lg font-bold border-b border-slate-800 pb-2">Patient Clinical Sliders</h2>
        
        <div className="flex flex-col gap-5">
          
          {/* SpO2 Slider */}
          <div>
            <div className="flex justify-between text-xs font-semibold mb-1">
              <span className="text-slate-300">Oxygenation (SpO2)</span>
              <span className="text-emerald-400">{vitals.spo2}%</span>
            </div>
            <input 
              type="range" 
              min="70" 
              max="100" 
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
              type="range" 
              min="40" 
              max="160" 
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
              type="range" 
              min="3" 
              max="15" 
              value={vitals.gcs}
              onChange={(e) => handleSliderChange('gcs', e.target.value)}
              className="w-full accent-cyan-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
            />
          </div>

          {/* NEW: Creatinine (Renal) Slider */}
          <div>
            <div className="flex justify-between text-xs font-semibold mb-1">
              <span className="text-slate-300">Creatinine (Renal)</span>
              <span className="text-emerald-400">{vitals.creatinine} mg/dL</span>
            </div>
            <input 
              type="range" 
              min="0.5" 
              max="5.0" 
              step="0.1"
              value={vitals.creatinine}
              onChange={(e) => handleSliderChange('creatinine', e.target.value)}
              className="w-full accent-cyan-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
            />
          </div>

          {/* NEW: Bilirubin (Hepatic) Slider */}
          <div>
            <div className="flex justify-between text-xs font-semibold mb-1">
              <span className="text-slate-300">Bilirubin (Hepatic)</span>
              <span className="text-emerald-400">{vitals.bilirubin} mg/dL</span>
            </div>
            <input 
              type="range" 
              min="0.1" 
              max="10.0" 
              step="0.1"
              value={vitals.bilirubin}
              onChange={(e) => handleSliderChange('bilirubin', e.target.value)}
              className="w-full accent-cyan-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
            />
          </div>

        </div>
      </div>
    </div>
  );
};
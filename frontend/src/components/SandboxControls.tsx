import { useState } from 'react';
import { Play, RotateCcw, CheckCircle2 } from 'lucide-react';

export default function SandboxControls() {
  // Array representing [Vasopressors, IV Fluids, Inotropes, Diuretics]
  const [therapy, setTherapy] = useState([0.0, 0.0, 0.0, 0.0]);
  const [isSimulating, setIsSimulating] = useState(false);
  const [simResult, setSimResult] = useState<{
    vitals_and_labs: number[];
    organ_states: number[];
    mode: string;
  } | null>(null);

  const handleSliderChange = (index: number, value: number) => {
    const newTherapy = [...therapy];
    newTherapy[index] = value;
    setTherapy(newTherapy);
  };

  const runSimulation = async () => {
    setIsSimulating(true);

    try {
      // 1. Try to query the live Python MARL engine
      const res = await fetch("http://127.0.0.1:8000/api/marl/step", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          patho_action: [0.1, 0.05, 0.0],
          therapy_action: therapy
        })
      });

      if (!res.ok) throw new Error("Backend offline or endpoint missing");

      const data = await res.json();
      setSimResult({
        ...data.observation.therapy_agent,
        mode: "Live MARL Engine"
      });
    } catch {
      // 2. Offline / Local fallback simulation math
      // Doses: [vaso, fluids, inotropes, diuretics]
      const vaso = therapy[0];
      const fluids = therapy[1];
      const inotropes = therapy[2];
      const diuretics = therapy[3];

      // Physiological response approximations
      const simHR = Math.max(50, Math.min(160, 78 + (inotropes * 35) + (vaso * 10) - (fluids * 8)));
      const simMAP = Math.max(45, Math.min(130, 70 + (vaso * 35) + (fluids * 18) - (diuretics * 12)));
      const simOrganScore = Math.max(0.1, Math.min(1.0, 0.85 - (vaso * 0.1) + (fluids * 0.12) - (inotropes * 0.05)));

      setSimResult({
        vitals_and_labs: [simHR, simMAP],
        organ_states: [simOrganScore],
        mode: "Local Simulation Engine"
      });
    } finally {
      setIsSimulating(false);
    }
  };

  const resetSimulation = async () => {
    setSimResult(null);
    setTherapy([0.0, 0.0, 0.0, 0.0]);
    try {
      await fetch("http://127.0.0.1:8000/api/marl/reset", { method: "POST" });
    } catch {
      // Silently catch if backend is offline
    }
  };

  const therapyLabels = ["Vasopressors", "IV Fluids", "Inotropes", "Diuretics"];

  return (
    <div className="bg-slate-900 p-6 rounded-2xl border border-slate-800 flex flex-col gap-5">
      <div className="flex justify-between items-center">
        <div>
          <h3 className="text-white text-lg font-bold">MARL Intervention Sandbox</h3>
          <p className="text-slate-400 text-xs">Simulate drug therapies on the Digital Twin</p>
        </div>
        <div className="flex items-center gap-2">
          <button 
            type="button"
            onClick={resetSimulation} 
            className="p-2.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-xl transition cursor-pointer"
            title="Reset Sandbox"
          >
            <RotateCcw size={16} />
          </button>
          <button 
            type="button"
            onClick={runSimulation} 
            disabled={isSimulating} 
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-xl font-bold text-sm transition disabled:opacity-50 cursor-pointer shadow-lg shadow-blue-600/20"
          >
            <Play size={15} fill="currentColor" />
            {isSimulating ? "Simulating..." : "Run Simulation"}
          </button>
        </div>
      </div>
      
      <div className="grid grid-cols-2 gap-x-6 gap-y-4">
        {therapyLabels.map((label, i) => (
          <div key={label} className="space-y-1.5">
            <div className="flex justify-between text-xs font-semibold">
              <span className="text-slate-300">{label}</span>
              <span className="text-blue-400 font-mono">{(therapy[i] * 100).toFixed(0)}% Dose</span>
            </div>
            <input 
              type="range" 
              min="0" 
              max="1" 
              step="0.05" 
              value={therapy[i]} 
              onChange={(e) => handleSliderChange(i, parseFloat(e.target.value))}
              className="w-full h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
            />
          </div>
        ))}
      </div>

      {/* Trajectory Outcome Card */}
      {simResult && (
        <div className="mt-2 p-4 bg-slate-800/80 rounded-xl border border-blue-500/30 flex flex-col gap-3">
          <div className="flex justify-between items-center">
            <h4 className="text-xs font-bold text-blue-400 tracking-wider uppercase flex items-center gap-1.5">
              <CheckCircle2 size={14} className="text-blue-400" />
              Predicted Trajectory Outcome
            </h4>
            <span className="text-[10px] bg-blue-500/10 text-blue-300 px-2 py-0.5 rounded-full font-mono">
              {simResult.mode}
            </span>
          </div>

          <div className="grid grid-cols-3 gap-3">
            <div className="bg-slate-900/90 p-3 rounded-lg border border-slate-700/60">
              <div className="text-slate-400 text-[10px] font-semibold uppercase">Simulated HR</div>
              <div className="text-white font-mono text-base font-bold mt-0.5">
                {simResult.vitals_and_labs[0].toFixed(1)} <span className="text-xs text-slate-400 font-normal">bpm</span>
              </div>
            </div>

            <div className="bg-slate-900/90 p-3 rounded-lg border border-slate-700/60">
              <div className="text-slate-400 text-[10px] font-semibold uppercase">Simulated MAP</div>
              <div className="text-white font-mono text-base font-bold mt-0.5">
                {simResult.vitals_and_labs[1].toFixed(1)} <span className="text-xs text-slate-400 font-normal">mmHg</span>
              </div>
            </div>

            <div className="bg-slate-900/90 p-3 rounded-lg border border-slate-700/60">
              <div className="text-slate-400 text-[10px] font-semibold uppercase">Organ Stability</div>
              <div className="text-emerald-400 font-mono text-base font-bold mt-0.5">
                {(simResult.organ_states[0] * 100).toFixed(0)}%
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
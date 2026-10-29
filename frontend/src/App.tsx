import { useState } from 'react';
import { RoleNavbar } from './components/RoleNavbar';
import type { ClinicalRole } from './components/RoleNavbar';
import { useEngineSocket } from './hooks/useEngineSocket';
import { WaveformCanvas } from './components/WaveformCanvas';
import { VitalControls } from './components/VitalControls';
import OrganGraph from './components/OrganGraph'; 

import NurseTaskQueue from './components/NurseTaskQueue';
import DigitalTwin from './components/DigitalTwin';
import ShapExplainer from './components/ShapExplainer';
import QuantumVisualizer from './components/QuantumVisualizer';
import PatientHub from './components/PatientHub';

export default function App() {
  const { state, actions, buffers } = useEngineSocket();
  const [currentRole, setCurrentRole] = useState<ClinicalRole>('physician');

  const roleSummary: Record<ClinicalRole, { title: string; description: string }> = {
    physician: {
      title: 'Physician Digital Twin Sandbox',
      description: 'Simulate drug interventions and evaluate cascade trajectories.'
    },
    nurse: {
      title: 'Nurse Dynamic Task Re-ranker',
      description: 'Prioritized clinical intervention queue and supply pre-staging alerts.'
    },
    patient: {
      title: 'Patient & Family Status Hub',
      description: 'Traffic light stability board and automated 4-hour plain-English summaries.'
    }
  };

  return (
    <div className="bg-[#0b0f19] text-slate-100 min-h-screen flex flex-col font-sans">
      <header className="w-full flex items-center justify-between p-4 bg-slate-800 border-b border-slate-700 sticky top-0 z-50">
        <div className="flex items-center space-x-3">
          <div className="h-10 w-10 rounded-lg bg-gradient-to-tr from-cyan-500 to-indigo-600 flex items-center justify-center shadow-lg text-xl">
            ⚡
          </div>
          <div>
            <h1 className="text-xl font-extrabold text-white">Q-MODS Engine</h1>
            <p className="text-xs text-slate-400 font-semibold uppercase">Quantum-Classical Spatiotemporal Graph</p>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <span className={`px-3 py-1 rounded-full text-xs font-semibold ${state.isBackendConnected ? 'bg-emerald-500/20 text-emerald-400' : 'bg-slate-500/20 text-slate-400'}`}>
            {state.isBackendConnected ? 'Python Engine: Online' : 'Python Engine: Offline (Local Sim)'}
          </span>
          <button
            onClick={actions.toggleEngineMode}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded text-white text-sm font-semibold transition-colors"
          >
            Toggle Live Engine
          </button>
        </div>
      </header>

      <div className="px-6 pt-6">
        <div className="flex flex-col gap-3 rounded-2xl border border-slate-800 bg-slate-900/60 p-4 backdrop-blur-md">
          <div className="flex items-center justify-between gap-4">
            <div>
              <p className="text-xs uppercase tracking-[0.2em] text-slate-400">Clinical view</p>
              <h2 className="text-lg font-bold text-white">{roleSummary[currentRole].title}</h2>
            </div>
            <RoleNavbar currentRole={currentRole} onSelectRole={setCurrentRole} />
          </div>
          <p className="text-sm text-slate-300">{roleSummary[currentRole].description}</p>
        </div>
      </div>

      <main className="flex-1 p-6 grid grid-cols-1 xl:grid-cols-12 gap-6 max-w-[1920px] mx-auto w-full">
        
        {/* COLUMN 1: Inputs and Controls */}
        <div className="xl:col-span-4 flex flex-col gap-6">
          <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5 backdrop-blur-md flex flex-col gap-4">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-bold">125Hz Telemetry Streams</h2>
              <span className="text-xs bg-emerald-500/10 text-emerald-400 px-2.5 py-1 rounded-full font-bold animate-pulse">
                Live Feed
              </span>
            </div>

            <WaveformCanvas buffers={buffers} heartRate={state.vitals.heart_rate} />

            <div className="flex items-center gap-3">
              <button
                onClick={() => actions.setIsPlaying(!state.isPlaying)}
                className="flex-1 bg-cyan-600 hover:bg-cyan-500 text-white font-bold text-sm py-2 px-4 rounded-xl transition"
              >
                {state.isPlaying ? 'Pause Feed' : 'Resume Feed'}
              </button>
              <button
                onClick={() => actions.setIsAnomalyActive(!state.isAnomalyActive)}
                className="bg-red-950/40 border border-red-800 hover:bg-red-900 text-red-400 font-bold text-sm py-2 px-4 rounded-xl transition"
              >
                {state.isAnomalyActive ? 'Clear Anomaly' : 'Trigger Anomaly'}
              </button>
            </div>
          </div>

          <VitalControls vitals={state.vitals} setVitals={actions.setVitals} />

          {currentRole === 'nurse' && <NurseTaskQueue vitals={state.vitals} cascadeRisk={state.cascadeRisk} />}
          {currentRole === 'patient' && <PatientHub vitals={state.vitals} cascadeRisk={state.cascadeRisk} />}
        </div>

        {/* COLUMN 2: Analytics, Risk, and AI Explainability */}
        <div className="xl:col-span-4 flex flex-col gap-6">
          
          <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5 backdrop-blur-md flex flex-col items-center">
            <h2 className="text-lg font-bold border-b border-slate-800 pb-2 w-full text-left mb-6">Systemic Cascade Warning</h2>
            <div className="flex flex-col items-center justify-center py-4">
              <span className={`text-7xl font-extrabold font-mono ${state.cascadeRisk > 60 ? 'text-rose-500' : state.cascadeRisk > 30 ? 'text-amber-500' : 'text-emerald-400'}`}>
                {state.cascadeRisk.toFixed(1)}%
              </span>
              <span className="text-sm text-slate-400 uppercase tracking-widest font-bold mt-2">Total Cascade Risk</span>
            </div>
          </div>

          <OrganGraph 
            hazards={state.organ_hazards || {
              cardiovascular: 0, pulmonary: 0, renal: 0, hepatic: 0, cns: 0
            }} 
          />

          <ShapExplainer shapData={state.shap || []} />

          <QuantumVisualizer activations={state.vqc_activations || []} />

        </div>

        {/* COLUMN 3: Digital Twin */}
        <div className="xl:col-span-4 flex flex-col">
          <DigitalTwin hazards={state.organ_hazards} />
        </div>

      </main>
    </div>
  );
}
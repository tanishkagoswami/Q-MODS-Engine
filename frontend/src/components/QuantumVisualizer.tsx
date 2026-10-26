import { Cpu } from 'lucide-react';

interface QuantumVisualizerProps {
  activations: number[];
}

export default function QuantumVisualizer({ activations = [] }: QuantumVisualizerProps) {
  // Fallback array if data hasn't streamed in yet
  const data = activations.length > 0 ? activations : Array(128).fill(0);

  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5 backdrop-blur-md w-full flex flex-col gap-4">
      <div className="flex justify-between items-center border-b border-slate-800 pb-3">
        <div className="flex items-center gap-2">
          <Cpu className="text-cyan-400 animate-pulse" size={20} />
          <div>
            <h3 className="text-white text-md font-bold">Quantum VQC State Matrix</h3>
            <p className="text-slate-400 text-xs font-mono">128-Qubit Latent Embedding Space</p>
          </div>
        </div>
        <span className="text-[10px] font-mono text-cyan-400 bg-cyan-500/10 px-2 py-0.5 rounded border border-cyan-500/20">
          ACTIVE STREAM
        </span>
      </div>

      {/* Dense Matrix Grid for the 128 dimensions */}
      <div className="grid grid-cols-16 gap-1.5 p-2 bg-slate-950/50 rounded-xl border border-slate-800/80">
        {data.map((val, idx) => {
          // Map value from [-1, 1] to opacity/color intensity
          const intensity = Math.abs(val);
          const isPositive = val >= 0;

          return (
            <div
              key={idx}
              title={`Dim ${idx}: ${val.toFixed(4)}`}
              className={`h-4 rounded-sm transition-all duration-75 flex items-center justify-center text-[8px] font-mono ${
                isPositive 
                  ? 'bg-cyan-400 shadow-[0_0_8px_rgba(34,211,238,0.4)]' 
                  : 'bg-indigo-600 shadow-[0_0_8px_rgba(79,70,229,0.4)]'
              }`}
              style={{ opacity: Math.max(0.2, intensity) }}
            />
          );
        })}
      </div>

      <div className="flex justify-between items-center text-[10px] text-slate-400 font-mono">
        <span>State Vector Tensor</span>
        <span className="text-cyan-400">Refresh: ~20Hz (Sub-50ms)</span>
      </div>
    </div>
  );
}
import { BrainCircuit } from 'lucide-react';

interface ShapFeature {
  feature: string;
  score: number;
}

interface ShapExplainerProps {
  shapData: ShapFeature[];
}

export default function ShapExplainer({ shapData = [] }: ShapExplainerProps) {
  // Find the highest score to accurately scale the bar chart widths
  const maxScore = Math.max(...shapData.map(d => d.score), 0.1);

  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5 backdrop-blur-md w-full flex flex-col gap-4">
      <div className="flex items-center gap-2 border-b border-slate-800 pb-3">
        <BrainCircuit className="text-purple-400" size={20} />
        <div>
          <h3 className="text-white text-md font-bold">AI Decision Explainability</h3>
          <p className="text-slate-400 text-xs">Live SHAP feature importance</p>
        </div>
      </div>
      
      <div className="flex flex-col gap-4">
        {shapData.length > 0 ? (
          shapData.map((item, idx) => (
            <div key={idx} className="flex flex-col gap-1.5">
              <div className="flex justify-between items-center text-xs font-semibold">
                <span className="text-slate-300">{item.feature}</span>
                <span className="font-mono text-purple-400 bg-purple-500/10 px-2 py-0.5 rounded">
                  +{item.score.toFixed(4)}
                </span>
              </div>
              <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                <div 
                  className="bg-gradient-to-r from-purple-600 to-indigo-400 h-1.5 rounded-full transition-all duration-300" 
                  style={{ width: `${(item.score / maxScore) * 100}%` }}
                ></div>
              </div>
            </div>
          ))
        ) : (
          <div className="text-xs text-slate-500 italic text-center py-4">
            Awaiting quantum tensor evaluations...
          </div>
        )}
      </div>
    </div>
  );
}
import { useState } from 'react';
import { Activity } from 'lucide-react';
import HistoryModal from './HistoryModal';
import type { OrganHazards } from '../hooks/useEngineSocket';
interface DigitalTwinProps {
  hazards: OrganHazards;
}

export default function DigitalTwin({ hazards }: DigitalTwinProps) {
  const [selectedOrgan, setSelectedOrgan] = useState<{id: string, name: string} | null>(null);

  const getOrganStyle = (risk: number) => {
    if (risk > 60) return { fill: 'rgba(244, 63, 94, 0.8)', stroke: '#f43f5e', animate: true }; 
    if (risk > 30) return { fill: 'rgba(245, 158, 11, 0.7)', stroke: '#f59e0b', animate: false }; 
    return { fill: 'rgba(45, 212, 191, 0.2)', stroke: '#2dd4bf', animate: false }; 
  };

  const organs = [
    { id: 'cns', name: 'Brain (CNS)', risk: hazards.cns, path: "M85 45 C85 25, 115 25, 115 45 C125 55, 105 75, 100 75 C95 75, 75 55, 85 45 Z" },
    { id: 'pulmonary', name: 'Lungs', risk: hazards.pulmonary, path: "M75 110 C65 140, 70 170, 90 170 C95 170, 95 140, 95 120 Z M125 110 C135 140, 130 170, 110 170 C105 170, 105 140, 105 120 Z" },
    { id: 'cardiovascular', name: 'Heart', risk: hazards.cardiovascular, path: "M95 140 C95 130, 115 130, 110 155 C105 170, 95 160, 95 140 Z" },
    { id: 'hepatic', name: 'Liver', risk: hazards.hepatic, path: "M70 180 C70 170, 110 170, 115 190 C120 210, 80 200, 70 180 Z" },
    { id: 'renal', name: 'Kidneys', risk: hazards.renal, path: "M75 210 C70 230, 85 230, 85 210 Z M125 210 C130 230, 115 230, 115 210 Z" }
  ];

  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5 backdrop-blur-md w-full h-full flex flex-col relative">
      <div className="flex items-center gap-2 border-b border-slate-800 pb-3 mb-4">
        <Activity className="text-emerald-400" size={20} />
        <div>
          <h3 className="text-white text-md font-bold">Interactive Digital Twin</h3>
          <p className="text-slate-400 text-xs">Real-time physiological hazard mapping</p>
        </div>
      </div>

      <div className="flex-1 flex items-center justify-center relative">
        <svg viewBox="0 0 200 350" className="w-full max-h-[600px] drop-shadow-2xl">
          <path 
            d="M60 90 C40 90, 30 110, 30 150 C30 200, 45 250, 50 350 L150 350 C155 250, 170 200, 170 150 C170 110, 160 90, 140 90 C120 90, 115 75, 115 75 C115 75, 120 60, 115 40 C110 15, 90 15, 85 40 C80 60, 85 75, 85 75 C85 75, 80 90, 60 90 Z" 
            fill="rgba(30, 41, 59, 0.4)" 
            stroke="#334155" 
            strokeWidth="2"
            strokeDasharray="4 4"
          />

          {organs.map((organ) => {
            const style = getOrganStyle(organ.risk);
            return (
              <g 
                key={organ.id} 
                className="transition-all duration-500 cursor-pointer hover:brightness-150"
                onClick={() => setSelectedOrgan({ id: organ.id, name: organ.name })}
              >
                <path 
                  d={organ.path} 
                  fill={style.fill} 
                  stroke={style.stroke} 
                  strokeWidth="2"
                  className={style.animate ? 'animate-pulse' : ''}
                />
              </g>
            );
          })}
        </svg>

        <div className="absolute bottom-4 left-4 flex flex-col gap-2">
          {organs.map(o => (
            <div key={o.id} className="flex items-center gap-2 text-xs font-mono cursor-pointer hover:text-white" onClick={() => setSelectedOrgan({ id: o.id, name: o.name })}>
              <div className="w-2 h-2 rounded-full" style={{ backgroundColor: getOrganStyle(o.risk).stroke }}></div>
              <span className="text-slate-300">{o.name}:</span>
              <span className={o.risk > 60 ? 'text-rose-400 font-bold' : o.risk > 30 ? 'text-amber-400' : 'text-emerald-400'}>
                {o.risk.toFixed(1)}%
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Render the Modal Overlay if an organ is clicked */}
      {selectedOrgan && (
        <HistoryModal 
          organId={selectedOrgan.id} 
          organName={selectedOrgan.name} 
          onClose={() => setSelectedOrgan(null)} 
        />
      )}
    </div>
  );
}
import React from 'react';

// We calculate basic hazards based on the vitals passed down
interface OrganGraphProps {
  vitals: {
    spo2: number;
    creatinine: number;
    heart_rate: number;
    bilirubin: number;
    gcs: number;
  };
}

export const OrganGraph: React.FC<OrganGraphProps> = ({ vitals }) => {
  // Simple mock calculations to make the bars react to the sliders
  const hazards = {
    Cardiovascular: Math.min(100, Math.abs(vitals.heart_rate - 78) * 1.2),
    Pulmonary: Math.min(100, (100 - vitals.spo2) * 3.5),
    Renal: Math.min(100, vitals.creatinine * 25),
    Hepatic: Math.min(100, vitals.bilirubin * 30),
    CNS: Math.min(100, (15 - vitals.gcs) * 8),
  };

  // Helper to determine the color of the bar based on the hazard percentage
  const getRiskColor = (val: number) => {
    if (val < 30) return 'bg-emerald-500';
    if (val < 70) return 'bg-amber-500';
    return 'bg-rose-500';
  };

  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5 backdrop-blur-md w-full">
      <h2 className="text-lg font-bold border-b border-slate-800 pb-2 text-slate-100 mb-4">
        Organ System Hazards
      </h2>
      
      <div className="flex flex-col gap-4">
        {Object.entries(hazards).map(([organ, value]) => (
          <div key={organ} className="flex flex-col gap-1">
            <div className="flex justify-between text-xs font-semibold text-slate-300">
              <span>{organ}</span>
              <span>{value.toFixed(1)}%</span>
            </div>
            {/* The progress bar background */}
            <div className="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
              {/* The animated fill bar */}
              <div 
                className={`h-full rounded-full transition-all duration-500 ease-out ${getRiskColor(value)}`}
                style={{ width: `${Math.max(2, value)}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
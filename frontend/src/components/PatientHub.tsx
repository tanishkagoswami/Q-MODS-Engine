import { Download, Heart, Wind } from 'lucide-react';
interface Vitals {
  spo2: number;
  creatinine: number;
  heart_rate: number;
  bilirubin: number;
  gcs: number;
}

interface PatientHubProps {
  vitals: Vitals;
  cascadeRisk: number;
}

export default function PatientHub({ vitals, cascadeRisk }: PatientHubProps) {
  // Translate system risk into a traffic light status
  const getStatus = () => {
    if (cascadeRisk < 30) return { label: 'Stable & Resting', color: 'text-emerald-400', bg: 'bg-emerald-500/10', border: 'border-emerald-500/30' };
    if (cascadeRisk < 60) return { label: 'Close Monitoring', color: 'text-amber-400', bg: 'bg-amber-500/10', border: 'border-amber-500/30' };
    return { label: 'Critical Care Active', color: 'text-rose-400', bg: 'bg-rose-500/10', border: 'border-rose-500/30' };
  };

  // Generate plain-English summary
  const getSummary = () => {
    const hrText = vitals.heart_rate >= 60 && vitals.heart_rate <= 100 
      ? "heart rate is in a normal, healthy range" 
      : "heart rate is being carefully managed by the care team";
      
    const o2Text = vitals.spo2 >= 95 
      ? "oxygen levels are excellent" 
      : "they are receiving support to maintain optimal oxygen levels";

    return `The patient is currently in a state of ${getStatus().label.toLowerCase()}. Their ${hrText}, and ${o2Text}. The overarching systemic stability is continuously tracked by our AI systems to ensure proactive care.`;
  };

  const downloadReport = () => {
    const reportText = `Q-MODS Engine | Family Shift Report\nDate: ${new Date().toLocaleString()}\n\nStatus: ${getStatus().label}\n\nSummary:\n${getSummary()}\n\nNote: This is an AI-generated summary to assist in understanding clinical stability. Always consult the attending physician for medical decisions.`;
    
    const blob = new Blob([reportText], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "Patient_Status_Report.txt";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const status = getStatus();

  return (
    <div className="bg-slate-900 p-6 rounded-2xl border border-slate-800 flex flex-col gap-5">
      <div className="flex justify-between items-center">
        <div>
          <h3 className="text-white text-lg font-bold">Patient & Family Hub</h3>
          <p className="text-slate-400 text-xs">Plain-English clinical translation</p>
        </div>
        <button 
          onClick={downloadReport}
          className="flex items-center gap-2 px-3 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-xl font-bold text-xs transition cursor-pointer"
        >
          <Download size={14} />
          Export Report
        </button>
      </div>

      {/* Traffic Light Status Banner */}
      <div className={`p-4 rounded-xl border ${status.bg} ${status.border} flex items-center justify-between`}>
        <div className="flex items-center gap-3">
          <div className="relative flex h-3 w-3">
            <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${status.bg.replace('/10', '')}`}></span>
            <span className={`relative inline-flex rounded-full h-3 w-3 ${status.bg.replace('/10', '')}`}></span>
          </div>
          <span className={`font-bold uppercase tracking-wider text-sm ${status.color}`}>
            Current Status: {status.label}
          </span>
        </div>
      </div>

      {/* Plain English Summary */}
      <div className="bg-slate-800/50 p-4 rounded-xl border border-slate-700/50 text-sm text-slate-300 leading-relaxed">
        {getSummary()}
      </div>

      {/* Simplified Vitals Readout */}
      <div className="grid grid-cols-2 gap-4 mt-2">
        <div className="flex items-center gap-3 bg-slate-900 border border-slate-800 p-3 rounded-xl">
          <div className="p-2 bg-blue-500/10 rounded-lg text-blue-400">
            <Heart size={18} />
          </div>
          <div>
            <div className="text-xs text-slate-400 font-semibold uppercase">Heart Rhythm</div>
            <div className="text-white font-bold">{vitals.heart_rate.toFixed(0)} <span className="text-xs text-slate-500 font-normal">bpm</span></div>
          </div>
        </div>

        <div className="flex items-center gap-3 bg-slate-900 border border-slate-800 p-3 rounded-xl">
          <div className="p-2 bg-emerald-500/10 rounded-lg text-emerald-400">
            <Wind size={18} />
          </div>
          <div>
            <div className="text-xs text-slate-400 font-semibold uppercase">Oxygen Level</div>
            <div className="text-white font-bold">{vitals.spo2.toFixed(1)} <span className="text-xs text-slate-500 font-normal">%</span></div>
          </div>
        </div>
      </div>
    </div>
  );
}
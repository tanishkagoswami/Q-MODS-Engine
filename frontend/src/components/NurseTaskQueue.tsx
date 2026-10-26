import { useState, useEffect } from 'react';
import { AlertTriangle, Clock, Activity, ArrowDownUp } from 'lucide-react';

interface Vitals {
  spo2: number;
  creatinine: number;
  heart_rate: number;
  bilirubin: number;
  gcs: number;
}

interface NurseTaskQueueProps {
  vitals: Vitals;
  cascadeRisk: number;
}

interface Task {
  id: string;
  task: string;
  priority: 'critical' | 'high' | 'routine';
  urgency_score: number;
}

export default function NurseTaskQueue({ vitals, cascadeRisk }: NurseTaskQueueProps) {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [isSyncing, setIsSyncing] = useState(false);

  // Send actual UI data to the Python Engine for AI analysis whenever vitals change
  useEffect(() => {
    const analyzeDataWithBackend = async () => {
      setIsSyncing(true);
      try {
        const res = await fetch("http://127.0.0.1:8000/api/nurse/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ 
            vitals: vitals, 
            cascade_risk: cascadeRisk 
          })
        });
        
        if (res.ok) {
          const data = await res.json();
          setTasks(data.tasks);
        }
      } catch (err) {
        console.error("Backend analysis failed. Engine offline.", err);
      } finally {
        // Small timeout for visual effect on the Live Sync indicator
        setTimeout(() => setIsSyncing(false), 300);
      }
    };

    analyzeDataWithBackend();
    
  }, [vitals, cascadeRisk]); // Dependency array: Re-runs instantly upon slider movement

  const getPriorityStyles = (priority: string) => {
    switch (priority) {
      case 'critical':
        return { bg: 'bg-rose-500/10', border: 'border-rose-500/40', text: 'text-rose-400', icon: <AlertTriangle size={18} /> };
      case 'high':
        return { bg: 'bg-amber-500/10', border: 'border-amber-500/40', text: 'text-amber-400', icon: <Activity size={18} /> };
      default:
        return { bg: 'bg-slate-800/50', border: 'border-slate-700/50', text: 'text-blue-400', icon: <Clock size={18} /> };
    }
  };

  return (
    <div className="bg-slate-900 p-6 rounded-2xl border border-slate-800 flex flex-col gap-5">
      <div className="flex justify-between items-center border-b border-slate-800 pb-4">
        <div>
          <h3 className="text-white text-lg font-bold flex items-center gap-2">
            <ArrowDownUp size={18} className="text-blue-500" />
            Dynamic Task Queue
          </h3>
          <p className="text-slate-400 text-xs mt-1">Backend AI-prioritized interventions</p>
        </div>
        <div className="flex items-center gap-2">
          <span className="relative flex h-2.5 w-2.5">
            {isSyncing && <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>}
            <span className={`relative inline-flex rounded-full h-2.5 w-2.5 ${isSyncing ? 'bg-emerald-400' : 'bg-emerald-600'}`}></span>
          </span>
          <span className="text-xs text-slate-400 font-mono">
            {isSyncing ? "Evaluating..." : "Live Sync"}
          </span>
        </div>
      </div>

      <div className="flex flex-col gap-3 min-h-[250px]">
        {tasks.length > 0 ? (
          tasks.map((task) => {
            const styles = getPriorityStyles(task.priority);
            return (
              <div 
                key={task.id} 
                className={`p-4 rounded-xl border ${styles.bg} ${styles.border} flex items-center justify-between transition-all duration-300 ease-in-out`}
              >
                <div className="flex items-center gap-4">
                  <div className={`p-2 rounded-lg bg-slate-900/50 ${styles.text}`}>
                    {styles.icon}
                  </div>
                  <div>
                    <h4 className="text-white font-semibold text-sm">{task.task}</h4>
                    <span className={`text-[10px] font-bold uppercase tracking-wider ${styles.text}`}>
                      {task.priority} Priority
                    </span>
                  </div>
                </div>
                <div className="flex flex-col items-end">
                  <span className="text-2xl font-black font-mono text-slate-300 opacity-50">
                    {task.urgency_score}
                  </span>
                  <span className="text-[9px] uppercase text-slate-500 font-bold">Urgency Score</span>
                </div>
              </div>
            );
          })
        ) : (
          <div className="flex-1 flex items-center justify-center text-slate-500 text-sm italic">
            Awaiting backend trajectory analysis...
          </div>
        )}
      </div>
    </div>
  );
}
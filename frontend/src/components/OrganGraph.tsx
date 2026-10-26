import { useState, useEffect } from 'react';import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import { X } from 'lucide-react';

// Define the shape of the incoming live hazard data
interface OrganGraphProps {
  hazards: {
    cardiovascular: number;
    pulmonary: number;
    renal: number;
    hepatic: number;
    cns: number;
  };
}

export default function OrganGraph({ hazards }: OrganGraphProps) {
  const [selectedOrgan, setSelectedOrgan] = useState<string | null>(null);
  const [historyData, setHistoryData] = useState<any[]>([]);

  // When an organ is clicked, fetch its corresponding historical vital from MongoDB
  useEffect(() => {
    if (!selectedOrgan) return;

    // Map the organ system to the specific vital tracked in the database
    const metricMap: Record<string, string> = {
      renal: 'creatinine',
      hepatic: 'bilirubin',
      cardiovascular: 'heart_rate',
      pulmonary: 'spo2',
      cns: 'gcs'
    };

    const metric = metricMap[selectedOrgan.toLowerCase()];
    
    if (metric) {
      fetch(`http://127.0.0.1:8000/api/history/${metric}`)
        .then(res => res.json())
        .then(data => {
          // Format the data for Recharts
          const formatted = data.history.map((d: any) => ({
            time: new Date(d.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
            value: d.vitals[metric]
          }));
          setHistoryData(formatted);
        })
        .catch(err => console.error("Failed to fetch history:", err));
    }
  }, [selectedOrgan]);

  // Helper to color the bars based on risk severity
  const getBarColor = (val: number) => {
    if (val > 75) return 'bg-red-500';
    if (val > 50) return 'bg-orange-500';
    return 'bg-emerald-400';
  };

  return (
    <div className="relative bg-slate-900 p-6 rounded-xl border border-slate-800">
      <h3 className="text-white text-lg font-semibold mb-6">Organ System Hazards</h3>
      
      <div className="space-y-5">
        {Object.entries(hazards).map(([organ, value]) => (
          <div 
            key={organ} 
            className="cursor-pointer group"
            onClick={() => setSelectedOrgan(organ)}
          >
            <div className="flex justify-between text-sm mb-1">
              <span className="text-slate-300 capitalize group-hover:text-blue-400 transition-colors">
                {organ}
              </span>
              <span className="text-slate-400">{value.toFixed(1)}%</span>
            </div>
            <div className="h-2 w-full bg-slate-800 rounded-full overflow-hidden">
              <div 
                className={`h-full ${getBarColor(value)} transition-all duration-500 ease-out`}
                style={{ width: `${value}%` }}
              />
            </div>
          </div>
        ))}
      </div>

      {/* Deep Dive Modal Overlay */}
      {selectedOrgan && (
        <div className="absolute inset-0 bg-slate-900/95 backdrop-blur-sm p-4 rounded-xl border border-blue-500/30 flex flex-col z-10">
          <div className="flex justify-between items-center mb-4">
            <h4 className="text-blue-400 font-medium capitalize">
              {selectedOrgan} History (Deep Dive)
            </h4>
            <button 
              onClick={() => setSelectedOrgan(null)}
              className="text-slate-400 hover:text-white transition-colors"
            >
              <X size={20} />
            </button>
          </div>
          
          <div className="flex-grow min-h-[150px]">
            {historyData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={historyData}>
                  <XAxis dataKey="time" stroke="#475569" fontSize={12} tickMargin={10} />
                  <YAxis stroke="#475569" fontSize={12} domain={['auto', 'auto']} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b' }}
                    itemStyle={{ color: '#38bdf8' }}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="value" 
                    stroke="#38bdf8" 
                    strokeWidth={2} 
                    dot={{ r: 4, fill: '#0ea5e9', strokeWidth: 0 }} 
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex items-center justify-center text-slate-500 text-sm">
                Loading snapshot history...
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
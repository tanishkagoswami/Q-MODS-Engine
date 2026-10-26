import { useState, useEffect } from 'react';
import { X, Activity } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';

interface HistoryModalProps {
  organId: string;
  organName: string;
  onClose: () => void;
}

export default function HistoryModal({ organId, organName, onClose }: HistoryModalProps) {
  const [data, setData] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // 1. Wrap our fetch logic in a function so we can call it repeatedly
    const fetchHistory = () => {
      fetch(`http://127.0.0.1:8000/api/history/${organId}`)
        .then(res => res.json())
        .then(fetchedData => {
          // Extract the array from the history key
          const historyArray = fetchedData.history || [];
          
          const formattedData = historyArray.map((entry: any, index: number) => {
            const vitals = entry.vitals || {};
            // Extract the specific organ hazard/vital from the backend payload
            const riskValue = vitals.risk || vitals.score || vitals[organId] || vitals.value || 0;
            
            return {
              time: `T-${historyArray.length - index}`,
              risk: riskValue
            };
          });
          
          setData(formattedData);
          setLoading(false);
        })
        .catch(err => {
          console.error("Failed to fetch history:", err);
          setLoading(false);
        });
    };

    // 2. Fetch immediately when the modal opens
    fetchHistory();

    // 3. Set up a polling interval to fetch new data every 5 seconds (matching Redis flusher)
    const intervalId = setInterval(fetchHistory, 5000);

    // 4. Cleanup function: stop fetching when the user closes the modal
    return () => clearInterval(intervalId);
    
  }, [organId]);

  return (
    <div className="absolute inset-0 z-50 flex items-center justify-center bg-slate-950/80 backdrop-blur-sm rounded-2xl p-4">
      <div className="bg-slate-900 border border-slate-700 rounded-xl p-5 w-full shadow-2xl flex flex-col gap-4">
        
        {/* Modal Header */}
        <div className="flex justify-between items-center border-b border-slate-800 pb-2">
          <div className="flex items-center gap-2">
            <Activity className="text-cyan-400" size={18} />
            <h3 className="text-white text-sm font-bold">{organName} Historical Trajectory</h3>
          </div>
          <button onClick={onClose} className="text-slate-400 hover:text-white transition-colors">
            <X size={18} />
          </button>
        </div>

        {/* Recharts Line Graph */}
        <div className="h-48 w-full">
          {loading ? (
            <div className="w-full h-full flex items-center justify-center text-slate-500 text-xs font-mono animate-pulse">
              Querying MongoDB Cold Storage...
            </div>
          ) : (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="time" stroke="#475569" fontSize={10} tickMargin={8} />
                <YAxis stroke="#475569" fontSize={10} domain={[0, 100]} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', fontSize: '12px', color: '#fff' }}
                  itemStyle={{ color: '#2dd4bf' }}
                />
                <Line 
                  type="monotone" 
                  dataKey="risk" 
                  stroke="#2dd4bf" 
                  strokeWidth={2}
                  dot={{ r: 3, fill: '#0f172a', stroke: '#2dd4bf', strokeWidth: 2 }}
                  activeDot={{ r: 5, fill: '#2dd4bf' }}
                />
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>
    </div>
  );
}
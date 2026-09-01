import React from 'react';

export type ClinicalRole = 'physician' | 'nurse' | 'patient';

interface RoleNavbarProps {
  currentRole: ClinicalRole;
  onSelectRole: (role: ClinicalRole) => void;
}

export const RoleNavbar: React.FC<RoleNavbarProps> = ({ currentRole, onSelectRole }) => {
  const roles: { id: ClinicalRole; label: string; badge: string; color: string }[] = [
    { id: 'physician', label: 'Physician Sandbox', badge: 'Digital Twin', color: 'from-blue-600 to-cyan-600' },
    { id: 'nurse', label: 'Nurse Station', badge: 'Task Re-ranker', color: 'from-emerald-600 to-teal-600' },
    { id: 'patient', label: 'Patient & Family', badge: 'Plain-English', color: 'from-purple-600 to-indigo-600' }
  ];

  return (
    <div className="flex items-center gap-2 bg-slate-900/80 p-1.5 rounded-xl border border-slate-800 backdrop-blur-md">
      {roles.map((r) => {
        const isActive = currentRole === r.id;
        return (
          <button
            key={r.id}
            onClick={() => onSelectRole(r.id)}
            className={`flex items-center gap-3 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
              isActive
                ? `bg-gradient-to-r ${r.color} text-white shadow-lg shadow-cyan-950/50`
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
            }`}
          >
            <span>{r.label}</span>
            <span
              className={`text-[10px] px-2 py-0.5 rounded-full font-semibold border ${
                isActive
                  ? 'bg-white/20 border-white/30 text-white'
                  : 'bg-slate-800 border-slate-700 text-slate-400'
              }`}
            >
              {r.badge}
            </span>
          </button>
        );
      })}
    </div>
  );
};
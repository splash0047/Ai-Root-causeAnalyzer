import { useState } from 'react';
import { api } from '../api';

const FAILURE_TYPES = [
  { value: 'noise', label: '📡 Feature Noise', desc: 'Shift distribution by N std devs' },
  { value: 'drop', label: '🚫 Feature Drop', desc: 'Zero out a feature' },
  { value: 'skew', label: '📈 Skew', desc: 'Push to extreme values' },
  { value: 'interaction', label: '🔗 Interaction', desc: 'Shift two features together' },
  { value: 'concept', label: '🔄 Concept Drift', desc: 'Flip target labels' },
  { value: 'missing', label: '❌ Missing Values', desc: 'Inject NaN values' },
];

const FEATURES = ['credit_score','income','loan_amount','debt_to_income','employment_years','age','num_credit_lines','has_mortgage','loan_purpose_encoded'];

export default function Simulator({ onViewRCA }) {
  const [failureType, setFailureType] = useState('noise');
  const [feature, setFeature] = useState('credit_score');
  const [feature2, setFeature2] = useState('income');
  const [nSamples, setNSamples] = useState(500);
  const [noiseFactor, setNoiseFactor] = useState(3.0);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  async function runSim() {
    setLoading(true); setError(null); setResult(null);
    try {
      const data = await api.runSimulation(failureType, feature, feature2, nSamples, noiseFactor, true);
      setResult(data);
    } catch (e) { setError(e.message); }
    setLoading(false);
  }

  return (
    <div className="animate-in">
      <div className="page-header">
        <h1 className="page-title">Failure Simulator</h1>
        <p className="page-subtitle">Inject controlled failures to validate the RCA engine</p>
      </div>
      <div className="sim-grid">
        <div className="glass-card sim-form">
          <h3 style={{marginBottom:20,fontSize:'1rem',fontWeight:700}}>⚙️ Config</h3>
          <div className="form-group">
            <label className="form-label">Failure Type</label>
            <select className="form-select" value={failureType} onChange={e=>setFailureType(e.target.value)}>
              {FAILURE_TYPES.map(t=><option key={t.value} value={t.value}>{t.label}</option>)}
            </select>
            <p style={{fontSize:'0.75rem',color:'var(--text-muted)',marginTop:6}}>{FAILURE_TYPES.find(t=>t.value===failureType)?.desc}</p>
          </div>
          <div className="form-group">
            <label className="form-label">Target Feature</label>
            <select className="form-select" value={feature} onChange={e=>setFeature(e.target.value)}>
              {FEATURES.map(f=><option key={f} value={f}>{f}</option>)}
            </select>
          </div>
          {failureType==='interaction' && (
            <div className="form-group">
              <label className="form-label">Second Feature</label>
              <select className="form-select" value={feature2} onChange={e=>setFeature2(e.target.value)}>
                {FEATURES.filter(f=>f!==feature).map(f=><option key={f} value={f}>{f}</option>)}
              </select>
            </div>
          )}
          <div className="form-group">
            <label className="form-label">Samples: {nSamples}</label>
            <input type="range" min="100" max="2000" step="100" value={nSamples} onChange={e=>setNSamples(Number(e.target.value))} style={{width:'100%'}} />
          </div>
          {failureType==='noise' && (
            <div className="form-group">
              <label className="form-label">Noise: {noiseFactor}σ</label>
              <input type="range" min="1" max="10" step="0.5" value={noiseFactor} onChange={e=>setNoiseFactor(Number(e.target.value))} style={{width:'100%'}} />
            </div>
          )}
          <button className="btn btn-primary" onClick={runSim} disabled={loading} style={{width:'100%',justifyContent:'center',marginTop:8}}>
            {loading ? <><div className="spinner"></div> Running...</> : '🚀 Run Simulation'}
          </button>
        </div>
        <div>
          {error && <div className="glass-card" style={{padding:24,borderColor:'rgba(244,63,94,0.3)'}}><p style={{color:'var(--accent-rose)'}}>❌ {error}</p></div>}
          {!result && !error && !loading && <div className="glass-card empty-state"><div className="empty-icon">🧪</div><p>Configure and run a simulation</p></div>}
          {loading && <div className="glass-card loading-overlay"><div className="spinner"></div><span>Running simulation + RCA...</span></div>}
          {result && (
            <div style={{display:'flex',flexDirection:'column',gap:16}}>
              <div className="glass-card" style={{padding:24}}>
                <h3 style={{marginBottom:16,fontWeight:700}}>📋 Simulation</h3>
                <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:12}}>
                  <div><span className="stat-label">Type</span><div style={{fontWeight:600}}>{result.simulation_type}</div></div>
                  <div><span className="stat-label">Target</span><div style={{fontWeight:600}}>{result.target_feature||'All'}</div></div>
                  <div><span className="stat-label">Samples</span><div style={{fontWeight:600}}>{result.n_samples}</div></div>
                  {result.drift_report && <div><span className="stat-label">Drift</span><span className={`badge badge-${result.drift_report.overall_drift_severity==='high'?'high':result.drift_report.overall_drift_severity==='medium'?'medium':'low'}`}>{result.drift_report.overall_drift_severity}</span></div>}
                </div>
              </div>
              {result.rca_result && (
                <div className="glass-card" style={{padding:24}}>
                  <h3 style={{marginBottom:16,fontWeight:700}}>🔍 RCA Diagnosis</h3>
                  <div style={{marginBottom:16,padding:16,background:'rgba(59,130,246,0.05)',borderRadius:'var(--radius-md)',border:'1px solid rgba(59,130,246,0.15)'}}>
                    <div style={{fontWeight:700,marginBottom:4}}>{result.rca_result.root_cause}</div>
                    <div style={{fontSize:'0.85rem',color:'var(--text-secondary)'}}>{result.rca_result.root_cause_detail}</div>
                  </div>
                  <div style={{display:'flex',gap:16,marginBottom:16,flexWrap:'wrap'}}>
                    <span className={`badge badge-${result.rca_result.severity}`}>{result.rca_result.severity}</span>
                    <span style={{fontFamily:'var(--font-mono)',fontSize:'0.85rem',color:'var(--accent-cyan)'}}>Confidence: {(result.rca_result.confidence_score*100).toFixed(1)}%</span>
                    {result.rca_result.is_uncertain && <span className="badge badge-medium">⚠ Uncertain</span>}
                  </div>
                  {result.rca_result.ranked_features?.length>0 && (
                    <div className="feature-rank">
                      {result.rca_result.ranked_features.slice(0,5).map((f,i)=>(
                        <div key={i} className="rank-item">
                          <div className="rank-num">{i+1}</div>
                          <div className="rank-bar-wrapper">
                            <div className="rank-name"><span>{f.feature}</span><span style={{display:'flex',gap:6,alignItems:'center'}}><span style={{fontFamily:'var(--font-mono)',fontSize:'0.75rem',color:'var(--text-muted)'}}>{f.impact.toFixed(4)}</span>{f.causality_confirmed && <span className="causal-badge">CAUSAL</span>}</span></div>
                            <div className="rank-bar"><div className={`rank-fill ${f.causality_confirmed?'causal':f.source==='interaction'?'interaction':'shap'}`} style={{width:`${Math.min(100,f.impact*200)}%`}}></div></div>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

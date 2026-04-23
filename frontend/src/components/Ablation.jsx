import { useState } from 'react';
import { api } from '../api';

export default function Ablation() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [nSamples, setNSamples] = useState(500);

  async function runStudy() {
    setLoading(true); setError(null); setResult(null);
    try {
      const res = await fetch('http://localhost:8000/ablation?n_samples=' + nSamples, { method: 'POST' });
      if (!res.ok) throw new Error('Ablation failed');
      setResult(await res.json());
    } catch (e) { setError(e.message); }
    setLoading(false);
  }

  return (
    <div className="animate-in">
      <div className="page-header">
        <h1 className="page-title">Ablation Study</h1>
        <p className="page-subtitle">Prove the incremental value of each RCA component</p>
      </div>

      {/* Controls */}
      <div className="glass-card" style={{padding:24,marginBottom:24,display:'flex',alignItems:'center',gap:24}}>
        <div>
          <span className="form-label">Samples per scenario</span>
          <select className="form-select" value={nSamples} onChange={e=>setNSamples(Number(e.target.value))} style={{width:140}}>
            <option value={200}>200</option>
            <option value={500}>500</option>
            <option value={1000}>1000</option>
          </select>
        </div>
        <button className="btn btn-primary" onClick={runStudy} disabled={loading} style={{marginTop:20}}>
          {loading ? <><div className="spinner"></div> Running 48 tests...</> : '🔬 Run Ablation Study'}
        </button>
        {loading && <span style={{fontSize:'0.8rem',color:'var(--text-muted)',marginTop:20}}>This takes 2-5 minutes...</span>}
      </div>

      {error && <div className="glass-card" style={{padding:24,borderColor:'rgba(244,63,94,0.3)'}}><p style={{color:'var(--accent-rose)'}}>❌ {error}</p></div>}

      {loading && (
        <div className="glass-card loading-overlay" style={{minHeight:300}}>
          <div className="spinner"></div>
          <span>Running 12 failure scenarios × 4 configurations...</span>
          <span style={{fontSize:'0.8rem',color:'var(--text-muted)'}}>Drift → +SHAP → +Counterfactuals → Full Pipeline</span>
        </div>
      )}

      {result && (
        <>
          {/* Accuracy Progression */}
          <div className="metrics-grid" style={{marginBottom:24}}>
            {result.configs?.map((config, i) => (
              <div key={i} className={`glass-card stat-card ${['blue','green','amber','purple'][i]}`}>
                <div className="stat-label">{config.name}</div>
                <div className={`stat-value ${['','green','amber',''][i]}`}>{(config.accuracy * 100).toFixed(0)}%</div>
                <div style={{marginTop:8,fontSize:'0.8rem',color:'var(--text-muted)'}}>
                  {config.correct}/{config.total} detected • {config.avg_time_ms}ms avg
                </div>
              </div>
            ))}
          </div>

          {/* Visual Bar Chart */}
          <div className="glass-card" style={{padding:24,marginBottom:24}}>
            <h3 style={{marginBottom:20,fontWeight:700}}>📊 Accuracy Progression</h3>
            <div style={{display:'flex',flexDirection:'column',gap:16}}>
              {result.configs?.map((config, i) => (
                <div key={i} style={{display:'flex',alignItems:'center',gap:16}}>
                  <div style={{width:140,fontSize:'0.85rem',fontWeight:600,textAlign:'right'}}>{config.name}</div>
                  <div style={{flex:1,height:28,background:'var(--bg-secondary)',borderRadius:'var(--radius-sm)',overflow:'hidden',position:'relative'}}>
                    <div style={{
                      height:'100%',
                      width:`${config.accuracy * 100}%`,
                      background:['var(--gradient-primary)','var(--gradient-success)','linear-gradient(135deg,#f59e0b,#8b5cf6)','linear-gradient(135deg,#ec4899,#8b5cf6)'][i],
                      borderRadius:'var(--radius-sm)',
                      transition:'width 1s ease',
                      display:'flex',alignItems:'center',justifyContent:'flex-end',paddingRight:8,
                    }}>
                      <span style={{fontSize:'0.75rem',fontWeight:700,color:'white'}}>{(config.accuracy*100).toFixed(0)}%</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Scenario Details */}
          <div className="glass-card" style={{overflow:'auto'}}>
            <table className="history-table">
              <thead>
                <tr>
                  <th>Scenario</th>
                  <th>Expected</th>
                  {result.configs?.map((c,i) => <th key={i}>{c.name}</th>)}
                </tr>
              </thead>
              <tbody>
                {result.configs?.[0]?.scenarios?.map((s, si) => (
                  <tr key={si}>
                    <td style={{fontWeight:500}}>{s.scenario}</td>
                    <td style={{fontFamily:'var(--font-mono)',fontSize:'0.8rem'}}>{s.expected}</td>
                    {result.configs.map((config, ci) => {
                      const sc = config.scenarios[si];
                      return (
                        <td key={ci}>
                          {sc?.error ? (
                            <span style={{color:'var(--accent-rose)',fontSize:'0.8rem'}}>ERR</span>
                          ) : sc?.hit ? (
                            <span style={{color:'var(--accent-emerald)',fontWeight:700}}>✓</span>
                          ) : (
                            <span style={{color:'var(--accent-rose)'}}>✗</span>
                          )}
                          {sc && !sc.error && (
                            <span style={{fontSize:'0.7rem',color:'var(--text-muted)',marginLeft:6}}>{sc.top_feature}</span>
                          )}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {!result && !loading && !error && (
        <div className="glass-card empty-state" style={{minHeight:300}}>
          <div className="empty-icon">🔬</div>
          <p>Run the ablation study to see how each component improves detection accuracy across 12 failure scenarios.</p>
        </div>
      )}
    </div>
  );
}

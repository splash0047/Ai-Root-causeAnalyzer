import { useState } from 'react';
import { api } from '../api';

export default function RCADetail({ data, onBack }) {
  const [feedbackSent, setFeedbackSent] = useState(false);
  const [feedbackLoading, setFeedbackLoading] = useState(false);
  const [notes, setNotes] = useState('');

  async function sendFeedback(type) {
    setFeedbackLoading(true);
    try {
      await api.submitFeedback(data.id, type, notes || null);
      setFeedbackSent(true);
    } catch (e) { console.error(e); }
    setFeedbackLoading(false);
  }

  const chain = data.reasoning_chain || [];
  const features = data.ranked_features || [];

  return (
    <div className="animate-in">
      <div className="detail-header">
        <button className="back-btn" onClick={onBack}>← Back</button>
        <div>
          <h1 className="page-title">RCA #{data.id}</h1>
          <p className="page-subtitle">{data.timestamp ? new Date(data.timestamp).toLocaleString() : ''}</p>
        </div>
        <span className={`badge badge-${data.severity}`} style={{marginLeft:'auto'}}>{data.severity}</span>
      </div>

      {/* Top Insights Summary */}
      <div className="glass-card" style={{padding:24,marginBottom:24, borderLeft: '4px solid var(--accent-cyan)'}}>
        <h3 style={{marginBottom: 16, color: 'var(--accent-cyan)'}}>✨ Top Insights</h3>
        <ul style={{listStyleType: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column', gap: '12px'}}>
          <li style={{display: 'flex', alignItems: 'flex-start', gap: '12px'}}>
            <span style={{fontSize: '1.2rem'}}>🔴</span>
            <div>
              <strong style={{display: 'block', color: 'var(--text-primary)'}}>Main Cause</strong>
              <span style={{color: 'var(--text-secondary)'}}>{data.root_cause}</span>
            </div>
          </li>
          {features.length > 0 && (
          <li style={{display: 'flex', alignItems: 'flex-start', gap: '12px'}}>
            <span style={{fontSize: '1.2rem'}}>👥</span>
            <div>
              <strong style={{display: 'block', color: 'var(--text-primary)'}}>Top Affected Feature</strong>
              <span style={{color: 'var(--text-secondary)'}}>{features[0].feature}</span>
            </div>
          </li>
          )}
          <li style={{display: 'flex', alignItems: 'flex-start', gap: '12px'}}>
            <span style={{fontSize: '1.2rem'}}>🔧</span>
            <div>
              <strong style={{display: 'block', color: 'var(--text-primary)'}}>Suggested Fix</strong>
              <span style={{color: 'var(--text-secondary)'}}>{data.suggested_fix || data.explanation || "No fix suggested"}</span>
            </div>
          </li>
        </ul>
        <div style={{display:'flex',gap:16,marginTop:20,flexWrap:'wrap', paddingTop: 16, borderTop: '1px solid rgba(255,255,255,0.1)'}}>
          <span style={{fontFamily:'var(--font-mono)',fontSize:'0.85rem',color:'var(--accent-cyan)'}}>
            Confidence: {(data.confidence_score*100).toFixed(1)}%
          </span>
          <span style={{fontSize:'0.85rem',color:'var(--text-muted)'}}>Mode: {data.rca_mode}</span>
          {data.is_uncertain && <span className="badge badge-medium">⚠ Uncertain</span>}
        </div>
      </div>

      {/* Confidence Breakdown */}
      {data.confidence_components && (
        <div className="glass-card" style={{padding:24,marginBottom:24}}>
          <h3 style={{marginBottom: 16}}>🎯 Confidence Breakdown</h3>
          <div style={{display: 'flex', flexDirection: 'column', gap: '12px'}}>
            {Object.entries(data.confidence_components).map(([key, value]) => (
              <div key={key}>
                <div style={{display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '4px', color: 'var(--text-secondary)'}}>
                  <span>{key.replace('_', ' ').toUpperCase()}</span>
                  <span>{(value * 100).toFixed(1)}% weight</span>
                </div>
                <div style={{width: '100%', height: '6px', backgroundColor: 'rgba(255,255,255,0.1)', borderRadius: '3px', overflow: 'hidden'}}>
                  <div style={{width: `${value * 100}%`, height: '100%', backgroundColor: 'var(--accent-cyan)', borderRadius: '3px'}}></div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Latency Breakdown */}
      {data.latency_breakdown && (
        <div className="glass-card" style={{padding:24,marginBottom:24}}>
          <h3 style={{marginBottom: 16}}>⏱️ Latency Breakdown</h3>
          <div style={{display: 'flex', gap: '4px', width: '100%', height: '24px', borderRadius: '4px', overflow: 'hidden', marginBottom: '16px'}}>
            {Object.entries(data.latency_breakdown).map(([stage, time], i) => {
              const total = Object.values(data.latency_breakdown).reduce((a, b) => a + b, 0);
              const percentage = (time / total) * 100;
              const colors = ['#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981', '#6366f1', '#14b8a6'];
              return (
                <div 
                  key={stage} 
                  style={{width: `${percentage}%`, height: '100%', backgroundColor: colors[i % colors.length]}}
                  title={`${stage}: ${time}ms`}
                />
              );
            })}
          </div>
          <div style={{display: 'flex', flexWrap: 'wrap', gap: '12px', fontSize: '0.8rem', color: 'var(--text-secondary)'}}>
            {Object.entries(data.latency_breakdown).map(([stage, time], i) => {
              const colors = ['#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981', '#6366f1', '#14b8a6'];
              return (
                <div key={stage} style={{display: 'flex', alignItems: 'center', gap: '4px'}}>
                  <div style={{width: '8px', height: '8px', borderRadius: '50%', backgroundColor: colors[i % colors.length]}}></div>
                  <span>{stage}: <strong>{time}ms</strong></span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      <div className="detail-grid">
        {/* Reasoning Chain */}
        <div className="glass-card detail-section">
          <h3>🧠 Reasoning Chain</h3>
          {chain.length > 0 ? (
            <div className="reasoning-chain">
              {chain.map((step, i) => (
                <div key={i} className="chain-step">
                  <div className="chain-dot">{i+1}</div>
                  <div className="chain-info">
                    <div className="chain-title">{step.step}</div>
                    <div className="chain-result">{step.result}</div>
                    {step.score !== undefined && <div className="chain-score">Score: {step.score}</div>}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p style={{color:'var(--text-muted)',fontSize:'0.85rem'}}>No reasoning chain available</p>
          )}
        </div>

        {/* Feature Ranking */}
        <div className="glass-card detail-section">
          <h3>📊 Ranked Features</h3>
          {features.length > 0 ? (
            <div className="feature-rank">
              {features.slice(0,8).map((f, i) => (
                <div key={i} className="rank-item">
                  <div className="rank-num">{i+1}</div>
                  <div className="rank-bar-wrapper">
                    <div className="rank-name">
                      <span>{f.feature}</span>
                      <span style={{display:'flex',gap:6,alignItems:'center'}}>
                        <span style={{fontFamily:'var(--font-mono)',fontSize:'0.75rem',color:'var(--text-muted)'}}>{typeof f.impact === 'number' ? f.impact.toFixed(4) : f.impact}</span>
                        {f.causality_confirmed && <span className="causal-badge">CAUSAL</span>}
                      </span>
                    </div>
                    <div className="rank-bar">
                      <div className={`rank-fill ${f.causality_confirmed?'causal':f.source==='interaction'?'interaction':'shap'}`}
                           style={{width:`${Math.min(100,(typeof f.impact==='number'?f.impact:0)*200)}%`}}></div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p style={{color:'var(--text-muted)',fontSize:'0.85rem'}}>No features ranked</p>
          )}
        </div>
      </div>

      {/* Feedback */}
      <div className="glass-card" style={{marginTop:24}}>
        <div className="feedback-section">
          {feedbackSent ? (
            <span style={{color:'var(--accent-emerald)',fontWeight:600}}>✓ Feedback submitted</span>
          ) : (
            <>
              <span className="feedback-label">Was this diagnosis accurate?</span>
              <input type="text" placeholder="Optional notes..." value={notes} onChange={e=>setNotes(e.target.value)}
                     className="form-input" style={{flex:1,maxWidth:300}} />
              <button className="btn btn-success" onClick={()=>sendFeedback('accurate')} disabled={feedbackLoading}>
                ✓ Accurate
              </button>
              <button className="btn btn-danger" onClick={()=>sendFeedback('rejected')} disabled={feedbackLoading}>
                ✗ Reject
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

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

      {/* Root Cause Summary */}
      <div className="glass-card" style={{padding:24,marginBottom:24}}>
        <div style={{fontWeight:800,fontSize:'1.1rem',marginBottom:8}}>{data.root_cause}</div>
        <div style={{fontSize:'0.9rem',color:'var(--text-secondary)',lineHeight:1.8}}>{data.explanation}</div>
        <div style={{display:'flex',gap:16,marginTop:16,flexWrap:'wrap'}}>
          <span style={{fontFamily:'var(--font-mono)',fontSize:'0.85rem',color:'var(--accent-cyan)'}}>
            Confidence: {(data.confidence_score*100).toFixed(1)}%
          </span>
          <span style={{fontSize:'0.85rem',color:'var(--text-muted)'}}>Mode: {data.rca_mode}</span>
          {data.is_uncertain && <span className="badge badge-medium">⚠ Uncertain</span>}
        </div>
      </div>

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

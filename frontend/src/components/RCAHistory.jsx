import { useState, useEffect } from 'react';
import { api } from '../api';

export default function RCAHistory({ onViewDetail }) {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [severity, setSeverity] = useState('');
  const [limit, setLimit] = useState(20);
  const [viewMode, setViewMode] = useState('all'); // 'all' | 'failures'

  useEffect(() => {
    loadHistory();
  }, [severity, limit]);

  async function loadHistory() {
    setLoading(true);
    try {
      const data = await api.getRCAHistory(limit, severity || null);
      setHistory(data.results || []);
    } catch (e) {
      console.error('History load error:', e);
    }
    setLoading(false);
  }

  return (
    <div className="animate-in">
      <div className="page-header">
        <h1 className="page-title">RCA History</h1>
        <p className="page-subtitle">Historical root cause analysis results</p>
      </div>

      {/* ── Filters ── */}
      <div className="history-controls">
        <div className="tab-group" style={{ display: 'flex', gap: '8px', marginRight: 'auto' }}>
          <button className={`btn ${viewMode === 'all' ? 'btn-primary' : 'btn-secondary'}`} onClick={() => setViewMode('all')}>All History</button>
          <button className={`btn ${viewMode === 'failures' ? 'btn-primary' : 'btn-secondary'}`} onClick={() => setViewMode('failures')}>Failure Explorer</button>
        </div>

        <select className="form-select" value={severity} onChange={e => setSeverity(e.target.value)}
                style={{ width: 160 }}>
          <option value="">All Severities</option>
          <option value="high">🔴 High</option>
          <option value="medium">🟡 Medium</option>
          <option value="low">🟢 Low</option>
        </select>

        <select className="form-select" value={limit} onChange={e => setLimit(Number(e.target.value))}
                style={{ width: 120 }}>
          <option value={10}>Last 10</option>
          <option value={20}>Last 20</option>
          <option value={50}>Last 50</option>
        </select>

        <button className="btn btn-secondary" onClick={loadHistory}>↻ Refresh</button>
      </div>

      {/* ── Table ── */}
      <div className="glass-card" style={{ overflow: 'auto' }}>
        {loading ? (
          <div className="loading-overlay"><div className="spinner"></div><span>Loading...</span></div>
        ) : history.length === 0 ? (
          <div className="empty-state">
            <div className="empty-icon">📋</div>
            <p>No RCA records found. Run a simulation to generate some.</p>
          </div>
        ) : viewMode === 'failures' ? (
          <div className="failures-explorer" style={{ padding: '20px' }}>
            <h3 style={{ marginBottom: '20px' }}>System Diagnostic Failures</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              {history.filter(item => item.user_feedback === 'rejected' || item.confidence_score < 0.4 || (item.ranked_features && item.ranked_features.length > 3)).map(item => {
                let category = 'Ambiguous Signals';
                let icon = '❓';
                if (item.user_feedback === 'rejected') { category = 'Incorrect RCA'; icon = '❌'; }
                else if (item.confidence_score < 0.4) { category = 'Low Confidence'; icon = '📉'; }
                else if (item.ranked_features && item.ranked_features.length > 3) { category = 'Multi-Cause Conflict'; icon = '⚔️'; }
                
                return (
                  <div key={item.id} className="glass-card" style={{ padding: '16px', display: 'flex', gap: '16px', alignItems: 'center' }}>
                    <div style={{ fontSize: '2rem' }}>{icon}</div>
                    <div style={{ flex: 1 }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <strong>{category}</strong>
                        <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>#{item.id} - {(item.confidence_score * 100).toFixed(0)}% Conf</span>
                      </div>
                      <p style={{ margin: '4px 0 0 0', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>{item.root_cause}</p>
                    </div>
                    <button className="btn btn-secondary btn-sm" onClick={() => onViewDetail(item)}>Investigate</button>
                  </div>
                );
              })}
              {history.filter(item => item.user_feedback === 'rejected' || item.confidence_score < 0.4 || (item.ranked_features && item.ranked_features.length > 3)).length === 0 && (
                <p style={{ color: 'var(--text-muted)' }}>No system diagnostic failures found.</p>
              )}
            </div>
          </div>
        ) : (
          <table className="history-table">
            <thead>
              <tr>
                <th>ID</th>
                <th>Timestamp</th>
                <th>Root Cause</th>
                <th>Severity</th>
                <th>Confidence</th>
                <th>Mode</th>
                <th>Feedback</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {history.map(item => (
                <tr key={item.id}>
                  <td style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem' }}>#{item.id}</td>
                  <td style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem', color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>
                    {item.timestamp ? new Date(item.timestamp).toLocaleString() : '—'}
                  </td>
                  <td className="root-cause-cell">{item.root_cause}</td>
                  <td><span className={`badge badge-${item.severity}`}>{item.severity}</span></td>
                  <td>
                    <div className="confidence-bar-cell">
                      <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem' }}>
                        {(item.confidence_score * 100).toFixed(0)}%
                      </span>
                      <div className="confidence-mini-bar">
                        <div className="confidence-mini-fill" style={{ width: `${item.confidence_score * 100}%` }}></div>
                      </div>
                    </div>
                  </td>
                  <td style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{item.rca_mode}</td>
                  <td>
                    {item.user_feedback ? (
                      <span className={`badge ${item.user_feedback === 'accurate' ? 'badge-low' : 'badge-high'}`}>
                        {item.user_feedback}
                      </span>
                    ) : (
                      <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>—</span>
                    )}
                  </td>
                  <td>
                    <button className="view-btn" onClick={() => onViewDetail(item)}>View</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

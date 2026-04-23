import { useState, useEffect } from 'react';
import { api } from '../api';

export default function Dashboard({ onViewRCA }) {
  const [metrics, setMetrics] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [window, setWindow] = useState(24);

  useEffect(() => {
    loadData();
  }, [window]);

  async function loadData() {
    setLoading(true);
    try {
      const [m, h] = await Promise.all([
        api.getMetrics(window),
        api.getRCAHistory(5),
      ]);
      setMetrics(m);
      setHistory(h.results || []);
    } catch (e) {
      console.error('Dashboard load error:', e);
    }
    setLoading(false);
  }

  if (loading) {
    return (
      <div className="loading-overlay">
        <div className="spinner"></div>
        <span>Loading dashboard...</span>
      </div>
    );
  }

  return (
    <div className="animate-in">
      <div className="page-header">
        <h1 className="page-title">Dashboard</h1>
        <p className="page-subtitle">Real-time model performance & RCA insights</p>
      </div>

      {/* ── Metrics Cards ── */}
      <div className="metrics-grid">
        <div className="glass-card stat-card blue">
          <div className="stat-label">Accuracy</div>
          <div className="stat-value">{metrics?.accuracy ? `${(metrics.accuracy * 100).toFixed(1)}%` : '—'}</div>
          <div style={{ marginTop: 8, fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            Baseline: {metrics?.baseline_accuracy ? `${(metrics.baseline_accuracy * 100).toFixed(1)}%` : '—'}
          </div>
        </div>

        <div className="glass-card stat-card green">
          <div className="stat-label">F1 Score</div>
          <div className="stat-value green">{metrics?.f1_score?.toFixed(4) || '—'}</div>
          <div style={{ marginTop: 8, fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            {metrics?.total_predictions || 0} predictions
          </div>
        </div>

        <div className="glass-card stat-card amber">
          <div className="stat-label">Accuracy Drop</div>
          <div className="stat-value amber">{metrics?.accuracy_drop ? `${(metrics.accuracy_drop * 100).toFixed(1)}%` : '0%'}</div>
          <div style={{ marginTop: 8, fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            vs training baseline
          </div>
        </div>

        <div className="glass-card stat-card purple">
          <div className="stat-label">High Anomalies</div>
          <div className="stat-value">{metrics?.high_anomaly_count ?? 0}</div>
          <div style={{ marginTop: 8, fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            Avg score: {metrics?.avg_anomaly_score?.toFixed(4) || '—'}
          </div>
        </div>
      </div>

      {/* ── Window Selector ── */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 24 }}>
        {[1, 6, 12, 24, 72].map(w => (
          <button key={w} className={`btn ${window === w ? 'btn-primary' : 'btn-secondary'}`}
                  onClick={() => setWindow(w)} style={{ padding: '6px 14px', fontSize: '0.8rem' }}>
            {w}h
          </button>
        ))}
        <button className="btn btn-secondary" onClick={loadData}
                style={{ marginLeft: 'auto', padding: '6px 14px', fontSize: '0.8rem' }}>
          ↻ Refresh
        </button>
      </div>

      {/* ── Recent RCA ── */}
      <div className="glass-card" style={{ padding: 24 }}>
        <h2 className="section-title">
          <span className="icon">🔍</span> Recent RCA Diagnoses
        </h2>

        {history.length === 0 ? (
          <div className="empty-state">
            <div className="empty-icon">📭</div>
            <p>No RCA results yet. Run a simulation or ingest data to get started.</p>
          </div>
        ) : (
          <table className="history-table">
            <thead>
              <tr>
                <th>Time</th>
                <th>Root Cause</th>
                <th>Severity</th>
                <th>Confidence</th>
                <th>Mode</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {history.map(item => (
                <tr key={item.id}>
                  <td style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
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
                    <button className="view-btn" onClick={() => onViewRCA(item)}>View</button>
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

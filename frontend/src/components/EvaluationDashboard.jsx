import { useEffect, useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { api } from '../api';

export default function EvaluationDashboard() {
  const [data, setData] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    api.getEvalMetrics().then(setData).catch(e => setError(e.message));
  }, []);

  if (error) return <div className="cyber-card">{error}</div>;
  if (!data) return <div className="cyber-card">Loading evaluation data…</div>;
  const { metrics, calibration_curve: curve } = data;
  const percentage = value => value == null ? 'Unavailable' : `${(value * 100).toFixed(1)}%`;

  return (
    <div>
      <h1 className="page-title">Evaluation</h1>
      <p className="page-subtitle">{data.scope}</p>
      <div className="metrics-grid">
        {[
          ['Feedback agreement', percentage(metrics.reviewed_agreement_rate)],
          ['Reviewed cases', metrics.reviewed_count],
          ['Average score', percentage(metrics.avg_confidence)],
          ['RCA runs', metrics.total_rca_runs],
          ['Measured latency', 'Unavailable'],
          ['False positive rate', 'Unavailable'],
        ].map(([label, value]) => (
          <div className="cyber-card stat-card" key={label}>
            <div className="stat-label">{label}</div>
            <div className="stat-value">{value}</div>
          </div>
        ))}
      </div>
      <div className="cyber-card" style={{ padding: 24, marginTop: 24 }}>
        <h2>Confidence score versus explicit reviewer feedback</h2>
        <p className="page-subtitle">Only reviewed cases appear here. Agreement is subjective and does not calibrate the score against independent ground truth.</p>
        {curve.length > 0 ? (
          <div style={{ width: '100%', height: 300 }}>
            <ResponsiveContainer>
              <LineChart data={curve}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="expected_accuracy" domain={[0, 1]} type="number" />
                <YAxis domain={[0, 1]} />
                <Tooltip />
                <Line type="monotone" dataKey="actual_accuracy" stroke="#8B5CF6" name="Reviewer agreement" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        ) : <p>No reviewed diagnoses yet.</p>}
      </div>
    </div>
  );
}

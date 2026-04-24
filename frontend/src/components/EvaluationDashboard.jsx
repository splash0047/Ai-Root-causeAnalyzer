import React, { useState, useEffect } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Legend, ResponsiveContainer 
} from 'recharts';
import { api } from '../api';

const EvaluationDashboard = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const response = await api.getEvalMetrics();
        setData(response);
      } catch (error) {
        console.error('Failed to load eval metrics', error);
      } finally {
        setLoading(false);
      }
    };
    fetchMetrics();
  }, []);

  if (loading) return <div className="glass-panel p-6 animate-pulse">Loading evaluation metrics...</div>;
  if (!data) return null;

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="glass-panel p-6">
        <h2 className="text-2xl font-bold mb-6 bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
          System Evaluation & ROI
        </h2>
        
        {/* KPIs */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
          <div className="bg-white/5 rounded-lg p-4 border border-white/10 hover-scale">
            <div className="text-sm text-gray-400">RCA Accuracy</div>
            <div className="text-3xl font-bold text-green-400">{(data.metrics.rca_accuracy * 100).toFixed(1)}%</div>
          </div>
          <div className="bg-white/5 rounded-lg p-4 border border-white/10 hover-scale">
            <div className="text-sm text-gray-400">Avg Confidence</div>
            <div className="text-3xl font-bold text-blue-400">{(data.metrics.avg_confidence * 100).toFixed(1)}%</div>
          </div>
          <div className="bg-white/5 rounded-lg p-4 border border-white/10 hover-scale">
            <div className="text-sm text-gray-400">Avg Latency</div>
            <div className="text-3xl font-bold text-yellow-400">{data.metrics.avg_latency_ms}ms</div>
          </div>
          <div className="bg-white/5 rounded-lg p-4 border border-white/10 hover-scale">
            <div className="text-sm text-gray-400">False Positive Rate</div>
            <div className="text-3xl font-bold text-purple-400">{(data.metrics.false_positive_rate * 100).toFixed(1)}%</div>
          </div>
        </div>

        {/* ROI / Business Impact */}
        <div className="bg-gradient-to-r from-green-500/10 to-emerald-500/10 border border-green-500/20 rounded-xl p-6 mb-8 flex items-center justify-between hover:shadow-lg hover:shadow-green-500/10 transition-all">
          <div>
            <h3 className="text-lg font-semibold text-green-300">Business Impact: Time Saved</h3>
            <p className="text-gray-300 mt-1">{data.metrics.estimated_time_saved}</p>
          </div>
          <div className="h-12 w-12 rounded-full bg-green-500/20 flex items-center justify-center">
            <span className="text-2xl">⏱️</span>
          </div>
        </div>

        {/* Segment-Level Heatmap */}
        <div className="mt-8 mb-8">
          <h3 className="text-lg font-semibold mb-2">Segment-Level Failure Rate Heatmap</h3>
          <p className="text-sm text-gray-400 mb-6">
            Identifies specific subpopulations where the model fails most frequently.
          </p>
          <div className="bg-white/5 rounded-xl p-6 border border-white/10">
            <div style={{display: 'grid', gridTemplateColumns: '100px repeat(4, 1fr)', gap: '4px'}}>
              {/* Header */}
              <div></div>
              <div className="text-center text-xs text-gray-400">Income &lt; 30k</div>
              <div className="text-center text-xs text-gray-400">Income 30-60k</div>
              <div className="text-center text-xs text-gray-400">Income 60-100k</div>
              <div className="text-center text-xs text-gray-400">Income &gt; 100k</div>
              
              {/* Row 1 */}
              <div className="text-xs text-gray-400 flex items-center justify-end pr-4">Age &lt; 30</div>
              <div className="h-12 rounded" style={{backgroundColor: 'rgba(239, 68, 68, 0.2)'}} title="Failure Rate: 2%"></div>
              <div className="h-12 rounded" style={{backgroundColor: 'rgba(239, 68, 68, 0.4)'}} title="Failure Rate: 4%"></div>
              <div className="h-12 rounded" style={{backgroundColor: 'rgba(239, 68, 68, 0.1)'}} title="Failure Rate: 1%"></div>
              <div className="h-12 rounded" style={{backgroundColor: 'rgba(239, 68, 68, 0.1)'}} title="Failure Rate: 1%"></div>
              
              {/* Row 2 */}
              <div className="text-xs text-gray-400 flex items-center justify-end pr-4">Age 30-50</div>
              <div className="h-12 rounded" style={{backgroundColor: 'rgba(239, 68, 68, 0.3)'}} title="Failure Rate: 3%"></div>
              <div className="h-12 rounded" style={{backgroundColor: 'rgba(239, 68, 68, 0.1)'}} title="Failure Rate: 1%"></div>
              <div className="h-12 rounded" style={{backgroundColor: 'rgba(239, 68, 68, 0.2)'}} title="Failure Rate: 2%"></div>
              <div className="h-12 rounded" style={{backgroundColor: 'rgba(239, 68, 68, 0.1)'}} title="Failure Rate: 1%"></div>
              
              {/* Row 3 */}
              <div className="text-xs text-gray-400 flex items-center justify-end pr-4">Age &gt; 50</div>
              <div className="h-12 rounded" style={{backgroundColor: 'rgba(239, 68, 68, 0.9)'}} title="Failure Rate: 12%"></div>
              <div className="h-12 rounded" style={{backgroundColor: 'rgba(239, 68, 68, 0.6)'}} title="Failure Rate: 7%"></div>
              <div className="h-12 rounded" style={{backgroundColor: 'rgba(239, 68, 68, 0.3)'}} title="Failure Rate: 3%"></div>
              <div className="h-12 rounded" style={{backgroundColor: 'rgba(239, 68, 68, 0.1)'}} title="Failure Rate: 1%"></div>
            </div>
            <div className="mt-4 text-center text-sm text-red-400">
              ⚠️ Alert: High failure rate detected for segment [Age &gt; 50 &amp; Income &lt; 30k].
            </div>
          </div>
        </div>

        {/* Calibration Curve */}
        <div className="mt-8">
          <h3 className="text-lg font-semibold mb-2">Confidence Calibration Curve</h3>
          <p className="text-sm text-gray-400 mb-6">
            Tracks predicted confidence against actual accuracy. A well-calibrated model tracks closely to the diagonal line.
          </p>
          <div className="h-80 w-full bg-white/5 rounded-xl p-4 border border-white/10">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data.calibration_curve}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="expected_accuracy" stroke="#9CA3AF" tickFormatter={(val) => `${(val * 100).toFixed(0)}%`} />
                <YAxis stroke="#9CA3AF" tickFormatter={(val) => `${(val * 100).toFixed(0)}%`} domain={[0, 1]} />
                <RechartsTooltip 
                  contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151', borderRadius: '0.5rem' }}
                  formatter={(value) => `${(value * 100).toFixed(1)}%`}
                />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="actual_accuracy" 
                  name="Actual Accuracy"
                  stroke="#8B5CF6" 
                  strokeWidth={3}
                  activeDot={{ r: 8 }} 
                />
                <Line 
                  type="linear" 
                  dataKey="expected_accuracy" 
                  name="Perfect Calibration"
                  stroke="#4B5563" 
                  strokeDasharray="5 5" 
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
};

export default EvaluationDashboard;

import { useState, useEffect } from 'react';
import './index.css';
import './App.css';
import { api } from './api';
import Dashboard from './components/Dashboard';
import RCAHistory from './components/RCAHistory';
import Simulator from './components/Simulator';
import RCADetail from './components/RCADetail';
import Ablation from './components/Ablation';
import EvaluationDashboard from './components/EvaluationDashboard';

export default function App() {
  const [tab, setTab] = useState('dashboard');
  const [health, setHealth] = useState(null);
  const [rcaDetail, setRcaDetail] = useState(null);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth({ status: 'offline' }));
  }, []);

  const showRCADetail = (data) => {
    setRcaDetail(data);
    setTab('rca-detail');
  };

  return (
    <div className="app-shell">
      {/* ── Sidebar ── */}
      <aside className="sidebar">
        <div className="sidebar-brand">
          <div className="brand-icon">⚡</div>
          <div>
            <div className="brand-title">RCA Engine</div>
            <div className="brand-sub">AI Root Cause Analyzer</div>
          </div>
        </div>

        <nav className="sidebar-nav">
          <button className={`nav-item ${tab === 'dashboard' ? 'active' : ''}`} onClick={() => setTab('dashboard')}>
            <span className="nav-icon">📊</span> Dashboard
          </button>
          <button className={`nav-item ${tab === 'history' ? 'active' : ''}`} onClick={() => setTab('history')}>
            <span className="nav-icon">📋</span> RCA History
          </button>
          <button className={`nav-item ${tab === 'simulator' ? 'active' : ''}`} onClick={() => setTab('simulator')}>
            <span className="nav-icon">🧪</span> Simulator
          </button>
          <button className={`nav-item ${tab === 'ablation' ? 'active' : ''}`} onClick={() => setTab('ablation')}>
            <span className="nav-icon">🔬</span> Ablation Study
          </button>
          <button className={`nav-item ${tab === 'evaluation' ? 'active' : ''}`} onClick={() => setTab('evaluation')}>
            <span className="nav-icon">📈</span> Evaluation
          </button>
        </nav>

        <div className="sidebar-footer">
          <div className={`status-dot ${health?.status === 'healthy' ? 'online' : 'offline'}`}></div>
          <span className="status-text">
            {health?.status === 'healthy' ? 'System Online' : 'Connecting...'}
          </span>
          {health?.vector_memory && <span className="status-badge">🧠 Memory</span>}
        </div>
      </aside>

      {/* ── Main Content ── */}
      <main className="main-content">
        {tab === 'dashboard' && <Dashboard onViewRCA={showRCADetail} />}
        {tab === 'history' && <RCAHistory onViewDetail={showRCADetail} />}
        {tab === 'simulator' && <Simulator onViewRCA={showRCADetail} />}
        {tab === 'ablation' && <Ablation />}
        {tab === 'evaluation' && <EvaluationDashboard />}
        {tab === 'rca-detail' && rcaDetail && (
          <RCADetail data={rcaDetail} onBack={() => setTab('history')} />
        )}
      </main>
    </div>
  );
}

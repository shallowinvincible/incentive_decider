import { useState, useEffect } from 'react';
import './App.css';
import Dashboard from './pages/Dashboard';
import Predict from './pages/Predict';
import BatchPredict from './pages/BatchPredict';
import ModelInsights from './pages/ModelInsights';
import { healthCheck } from './api';

const NAV_ITEMS = [
  { id: 'dashboard', label: 'Dashboard', icon: '📊' },
  { id: 'predict', label: 'Predict', icon: '🎯' },
  { id: 'batch', label: 'Batch Predict', icon: '📦' },
  { id: 'insights', label: 'Model Insights', icon: '🧠' },
];

function App() {
  const [activePage, setActivePage] = useState('dashboard');
  const [serverStatus, setServerStatus] = useState('checking');
  const [sidebarOpen, setSidebarOpen] = useState(false);

  useEffect(() => {
    healthCheck()
      .then((data) => setServerStatus(data.model_loaded ? 'ready' : 'loading'))
      .catch(() => setServerStatus('offline'));

    const interval = setInterval(() => {
      healthCheck()
        .then((data) => setServerStatus(data.model_loaded ? 'ready' : 'loading'))
        .catch(() => setServerStatus('offline'));
    }, 10000);
    return () => clearInterval(interval);
  }, []);

  const renderPage = () => {
    switch (activePage) {
      case 'dashboard': return <Dashboard />;
      case 'predict': return <Predict />;
      case 'batch': return <BatchPredict />;
      case 'insights': return <ModelInsights />;
      default: return <Dashboard />;
    }
  };

  return (
    <div className="app-container">
      {/* Mobile menu overlay */}
      {sidebarOpen && (
        <div className="sidebar-overlay" onClick={() => setSidebarOpen(false)} />
      )}

      {/* Sidebar */}
      <aside className={`sidebar ${sidebarOpen ? 'open' : ''}`}>
        <div className="sidebar-brand">
          <div className="sidebar-brand-icon">⚡</div>
          <div>
            <div className="sidebar-brand-text">IncentiveAI</div>
            <div className="sidebar-brand-sub">Optimization Engine</div>
          </div>
        </div>

        <nav className="sidebar-nav">
          {NAV_ITEMS.map((item) => (
            <div
              key={item.id}
              className={`nav-item ${activePage === item.id ? 'active' : ''}`}
              onClick={() => { setActivePage(item.id); setSidebarOpen(false); }}
            >
              <span className="nav-icon">{item.icon}</span>
              <span>{item.label}</span>
            </div>
          ))}
        </nav>

        <div className="sidebar-footer">
          <div className="server-status">
            <span
              className={`status-dot ${serverStatus}`}
            />
            <span className="status-text">
              {serverStatus === 'ready' ? 'API Connected' :
                serverStatus === 'loading' ? 'Model Loading...' :
                  serverStatus === 'checking' ? 'Connecting...' : 'API Offline'}
            </span>
          </div>
          <div className="sidebar-footer-text">
            Dynamic Incentive Optimizer v1.0
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="main-content">
        {/* Mobile hamburger */}
        <button className="mobile-menu-btn" onClick={() => setSidebarOpen(!sidebarOpen)}>
          ☰
        </button>
        {renderPage()}
      </main>
    </div>
  );
}

export default App;

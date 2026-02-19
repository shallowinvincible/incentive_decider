import { useState, useEffect } from 'react';
import { fetchModelInfo, retrainModel } from '../api';

export default function ModelInsights() {
    const [info, setInfo] = useState(null);
    const [loading, setLoading] = useState(true);
    const [retraining, setRetraining] = useState(false);
    const [error, setError] = useState(null);

    const loadInfo = () => {
        setLoading(true);
        fetchModelInfo()
            .then(setInfo)
            .catch((e) => setError(e.message))
            .finally(() => setLoading(false));
    };

    useEffect(() => { loadInfo(); }, []);

    const handleRetrain = async () => {
        setRetraining(true);
        try {
            await retrainModel();
            loadInfo();
        } catch (e) {
            setError(e.message);
        } finally {
            setRetraining(false);
        }
    };

    if (loading) {
        return (
            <div className="loading-state">
                <div className="spinner spinner-lg" />
                <span>Loading model info...</span>
            </div>
        );
    }

    if (error) {
        return (
            <div className="empty-state">
                <div className="empty-state-icon">⚠️</div>
                <div className="empty-state-title">Error Loading Model Info</div>
                <div className="empty-state-text">{error}</div>
            </div>
        );
    }

    const { metrics, feature_importances, config } = info;

    // Prepare feature importance data
    const featureEntries = Object.entries(feature_importances || {})
        .sort(([, a], [, b]) => b - a)
        .slice(0, 15);
    const maxImportance = featureEntries.length > 0 ? featureEntries[0][1] : 1;

    return (
        <div>
            <div className="page-header">
                <h1 className="page-title">🧠 Model Insights</h1>
                <p className="page-subtitle">
                    Performance metrics, feature importances, and platform configuration
                </p>
            </div>

            {/* Model Metrics */}
            <div className="card" style={{ marginBottom: '24px' }}>
                <div className="card-header">
                    <div>
                        <div className="card-title">Model Performance</div>
                        <div className="card-subtitle">{info.model_type} — {metrics.n_features} features</div>
                    </div>
                    <button className="btn btn-secondary" onClick={handleRetrain} disabled={retraining}>
                        {retraining ? <><div className="spinner" /> Retraining...</> : '🔄 Retrain Model'}
                    </button>
                </div>

                <div className="metrics-row">
                    <div className="metric-card">
                        <div className="metric-label">Accuracy</div>
                        <div className="metric-value" style={{ color: 'var(--accent-emerald)' }}>{(metrics.accuracy * 100).toFixed(1)}%</div>
                    </div>
                    <div className="metric-card">
                        <div className="metric-label">ROC-AUC</div>
                        <div className="metric-value" style={{ color: 'var(--accent-violet-light)' }}>{(metrics.roc_auc * 100).toFixed(1)}%</div>
                    </div>
                    <div className="metric-card">
                        <div className="metric-label">Precision</div>
                        <div className="metric-value" style={{ color: 'var(--accent-cyan)' }}>{(metrics.precision * 100).toFixed(1)}%</div>
                    </div>
                    <div className="metric-card">
                        <div className="metric-label">Recall</div>
                        <div className="metric-value" style={{ color: 'var(--accent-amber)' }}>{(metrics.recall * 100).toFixed(1)}%</div>
                    </div>
                    <div className="metric-card">
                        <div className="metric-label">F1 Score</div>
                        <div className="metric-value" style={{ color: 'var(--accent-emerald)' }}>{(metrics.f1_score * 100).toFixed(1)}%</div>
                    </div>
                    <div className="metric-card">
                        <div className="metric-label">Train Size</div>
                        <div className="metric-value" style={{ color: 'var(--text-secondary)' }}>{metrics.train_size.toLocaleString()}</div>
                    </div>
                    <div className="metric-card">
                        <div className="metric-label">Test Size</div>
                        <div className="metric-value" style={{ color: 'var(--text-secondary)' }}>{metrics.test_size.toLocaleString()}</div>
                    </div>
                </div>

                {/* Accuracy visual bar */}
                <div style={{ marginTop: '16px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', color: 'var(--text-muted)', marginBottom: '6px' }}>
                        <span>Model Accuracy</span>
                        <span>{(metrics.accuracy * 100).toFixed(1)}%</span>
                    </div>
                    <div style={{ height: '8px', background: 'var(--bg-input)', borderRadius: '4px', overflow: 'hidden' }}>
                        <div style={{
                            height: '100%',
                            width: `${metrics.accuracy * 100}%`,
                            background: 'var(--gradient-profit)',
                            borderRadius: '4px',
                            transition: 'width 1s cubic-bezier(0.34, 1.56, 0.64, 1)',
                        }} />
                    </div>
                </div>
                <div style={{ marginTop: '12px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', color: 'var(--text-muted)', marginBottom: '6px' }}>
                        <span>ROC-AUC Score</span>
                        <span>{(metrics.roc_auc * 100).toFixed(1)}%</span>
                    </div>
                    <div style={{ height: '8px', background: 'var(--bg-input)', borderRadius: '4px', overflow: 'hidden' }}>
                        <div style={{
                            height: '100%',
                            width: `${metrics.roc_auc * 100}%`,
                            background: 'var(--gradient-primary)',
                            borderRadius: '4px',
                            transition: 'width 1s cubic-bezier(0.34, 1.56, 0.64, 1)',
                        }} />
                    </div>
                </div>
            </div>

            <div className="charts-grid">
                {/* Feature Importance */}
                <div className="chart-container">
                    <div className="chart-title">🏆 Top Feature Importances</div>
                    <div className="feature-list">
                        {featureEntries.map(([name, value], i) => (
                            <div className="feature-row" key={name}>
                                <div className="feature-rank">{i + 1}</div>
                                <div className="feature-name" title={name}>{name}</div>
                                <div className="feature-bar-track">
                                    <div
                                        className="feature-bar-fill"
                                        style={{ width: `${(value / maxImportance) * 100}%` }}
                                    />
                                </div>
                                <div className="feature-value">{(value * 100).toFixed(2)}%</div>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Platform Config */}
                <div className="chart-container">
                    <div className="chart-title">⚙️ Platform Configuration</div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                        <ConfigRow label="Platform Name" value={config.platform_name} />
                        <ConfigRow label="Commission Rate" value={`${(config.commission_rate * 100)}%`} />
                        <ConfigRow label="Delay Penalty" value={`₹${config.delay_penalty_per_minute}/min`} />
                        <ConfigRow label="Max Incentive" value={`₹${config.max_incentive}`} />
                        <ConfigRow label="Min Incentive" value={`₹${config.min_incentive}`} />
                        <ConfigRow label="Incentive Step" value={`₹${config.incentive_step}`} />
                        <ConfigRow label="Acceptance Threshold" value={`${(config.required_acceptance_threshold * 100)}%`} />
                        <ConfigRow label="Currency" value={config.currency} />
                    </div>

                    <div style={{ marginTop: '24px', padding: '16px', background: 'rgba(139, 92, 246, 0.06)', borderRadius: 'var(--radius-md)', border: '1px solid rgba(139, 92, 246, 0.15)' }}>
                        <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--accent-violet-light)', marginBottom: '8px' }}>
                            ℹ️ How the Optimizer Works
                        </div>
                        <div style={{ fontSize: '12px', color: 'var(--text-secondary)', lineHeight: '1.7' }}>
                            For each order, the system sweeps incentives from ₹{config.min_incentive} to ₹{config.max_incentive} in
                            steps of ₹{config.incentive_step}. It predicts acceptance probability using the trained model, computes
                            expected profit, and selects the incentive that maximizes profit while keeping acceptance ≥{' '}
                            {(config.required_acceptance_threshold * 100)}%.
                        </div>
                    </div>
                </div>

                {/* Pipeline Info */}
                <div className="chart-container" style={{ gridColumn: '1 / -1' }}>
                    <div className="chart-title">🔧 ML Pipeline Architecture</div>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: '12px', marginTop: '8px' }}>
                        {[
                            { step: '1', name: 'Clean', icon: '🧹', desc: 'Dedup, impute, fix types' },
                            { step: '2', name: 'Encode', icon: '🔤', desc: 'One-hot categoricals' },
                            { step: '3', name: 'Scale', icon: '📏', desc: 'StandardScaler numerics' },
                            { step: '4', name: 'Polynomial', icon: '🧮', desc: 'Degree-2 interactions' },
                            { step: '5', name: 'LASSO', icon: '✂️', desc: `→ ${info.n_selected_features} features` },
                            { step: '6', name: 'GBClassify', icon: '🎯', desc: 'P(accept | features)' },
                        ].map((s) => (
                            <div key={s.step} style={{
                                background: 'var(--bg-glass)',
                                border: '1px solid var(--border-color)',
                                borderRadius: 'var(--radius-md)',
                                padding: '16px 12px',
                                textAlign: 'center',
                                position: 'relative',
                            }}>
                                <div style={{ fontSize: '24px', marginBottom: '8px' }}>{s.icon}</div>
                                <div style={{ fontSize: '13px', fontWeight: 600, marginBottom: '4px' }}>{s.name}</div>
                                <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>{s.desc}</div>
                                <div style={{
                                    position: 'absolute', top: '8px', right: '8px',
                                    fontSize: '10px', fontWeight: 700, color: 'var(--accent-violet-light)',
                                    background: 'rgba(139, 92, 246, 0.1)',
                                    padding: '2px 6px', borderRadius: '4px',
                                }}>
                                    {s.step}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
}

function ConfigRow({ label, value }) {
    return (
        <div style={{
            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
            padding: '8px 12px',
            background: 'var(--bg-glass)',
            borderRadius: 'var(--radius-sm)',
            border: '1px solid var(--border-color)',
        }}>
            <span style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>{label}</span>
            <span style={{ fontSize: '13px', fontWeight: 600, fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>{value}</span>
        </div>
    );
}

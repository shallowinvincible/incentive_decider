import { useState } from 'react';
import { fetchSampleOrders, predictBatch } from '../api';

export default function BatchPredict() {
    const [orders, setOrders] = useState(null);
    const [results, setResults] = useState(null);
    const [summary, setSummary] = useState(null);
    const [loading, setLoading] = useState(false);
    const [loadingOrders, setLoadingOrders] = useState(false);
    const [error, setError] = useState(null);
    const [count, setCount] = useState(20);

    const loadSample = async () => {
        setLoadingOrders(true);
        setError(null);
        setResults(null);
        setSummary(null);
        try {
            const data = await fetchSampleOrders(count);
            setOrders(data.orders);
        } catch (e) {
            setError(e.message);
        } finally {
            setLoadingOrders(false);
        }
    };

    const runBatchPrediction = async () => {
        if (!orders || orders.length === 0) return;
        setLoading(true);
        setError(null);
        try {
            const data = await predictBatch(orders);
            setResults(data.results);
            setSummary(data.summary);
        } catch (e) {
            setError(e.message);
        } finally {
            setLoading(false);
        }
    };

    const downloadCSV = () => {
        if (!results) return;
        const header = 'order_id,recommended_incentive,predicted_acceptance_probability,expected_profit,delivery_revenue,threshold_met\n';
        const rows = results.map(r =>
            `${r.order_id},${r.recommended_incentive},${r.predicted_acceptance_probability},${r.expected_profit},${r.delivery_revenue},${r.threshold_met}`
        ).join('\n');
        const blob = new Blob([header + rows], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'incentive_predictions.csv';
        a.click();
        URL.revokeObjectURL(url);
    };

    return (
        <div>
            <div className="page-header">
                <h1 className="page-title">📦 Batch Prediction</h1>
                <p className="page-subtitle">Optimize incentives for multiple orders at once</p>
            </div>

            {/* Controls */}
            <div className="card" style={{ marginBottom: '24px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '16px', flexWrap: 'wrap' }}>
                    <div className="form-group" style={{ width: '140px' }}>
                        <label className="form-label">Sample Size</label>
                        <input
                            className="form-input"
                            type="number"
                            min="5"
                            max="50"
                            value={count}
                            onChange={(e) => setCount(parseInt(e.target.value) || 10)}
                        />
                    </div>
                    <button className="btn btn-secondary" onClick={loadSample} disabled={loadingOrders} style={{ marginTop: '18px' }}>
                        {loadingOrders ? <><div className="spinner" /> Loading...</> : '📥 Load Sample Orders'}
                    </button>
                    {orders && (
                        <button className="btn btn-primary" onClick={runBatchPrediction} disabled={loading} style={{ marginTop: '18px' }}>
                            {loading ? <><div className="spinner" /> Optimizing {orders.length} orders...</> : `⚡ Optimize ${orders.length} Orders`}
                        </button>
                    )}
                    {results && (
                        <button className="download-btn" onClick={downloadCSV} style={{ marginTop: '18px' }}>
                            📄 Download CSV
                        </button>
                    )}
                </div>
            </div>

            {error && (
                <div className="card" style={{ borderColor: 'rgba(244, 63, 94, 0.3)', marginBottom: '24px' }}>
                    <div style={{ color: 'var(--accent-rose)' }}>⚠️ {error}</div>
                </div>
            )}

            {/* Summary */}
            {summary && (
                <div className="batch-summary" style={{ animation: 'slideUp 0.4s ease' }}>
                    <div>
                        <div className="kpi-label">Total Orders</div>
                        <div className="kpi-value violet">{summary.total_orders}</div>
                    </div>
                    <div>
                        <div className="kpi-label">Avg Incentive</div>
                        <div className="kpi-value amber">₹{summary.avg_incentive}</div>
                    </div>
                    <div>
                        <div className="kpi-label">Total Incentive Cost</div>
                        <div className="kpi-value rose">₹{summary.total_incentive_cost.toLocaleString()}</div>
                    </div>
                    <div>
                        <div className="kpi-label">Avg Profit</div>
                        <div className="kpi-value green">₹{summary.avg_profit}</div>
                    </div>
                    <div>
                        <div className="kpi-label">Total Profit</div>
                        <div className="kpi-value green">₹{summary.total_profit.toLocaleString()}</div>
                    </div>
                    <div>
                        <div className="kpi-label">Avg Acceptance</div>
                        <div className="kpi-value violet">{(summary.avg_acceptance * 100).toFixed(1)}%</div>
                    </div>
                    <div>
                        <div className="kpi-label">Above 90% Threshold</div>
                        <div className="kpi-value" style={{ color: summary.pct_above_threshold >= 90 ? 'var(--accent-emerald)' : 'var(--accent-rose)' }}>
                            {summary.pct_above_threshold}%
                        </div>
                    </div>
                </div>
            )}

            {/* Results Table */}
            {results && (
                <div className="card">
                    <div className="card-header">
                        <div className="card-title">Optimization Results</div>
                        <div className="card-subtitle">{results.length} orders processed</div>
                    </div>
                    <div className="table-container">
                        <table className="data-table">
                            <thead>
                                <tr>
                                    <th>Order ID</th>
                                    <th>Incentive</th>
                                    <th>Acceptance</th>
                                    <th>Expected Profit</th>
                                    <th>Revenue</th>
                                    <th>Status</th>
                                </tr>
                            </thead>
                            <tbody>
                                {results.map((r) => (
                                    <tr key={r.order_id}>
                                        <td>{r.order_id}</td>
                                        <td style={{ color: 'var(--accent-violet-light)' }}>₹{r.recommended_incentive}</td>
                                        <td>{(r.predicted_acceptance_probability * 100).toFixed(1)}%</td>
                                        <td style={{ color: r.expected_profit >= 0 ? 'var(--accent-emerald)' : 'var(--accent-rose)' }}>
                                            ₹{r.expected_profit}
                                        </td>
                                        <td>₹{r.delivery_revenue}</td>
                                        <td>
                                            <span className={`badge ${r.threshold_met ? 'badge-success' : 'badge-warning'}`}>
                                                {r.threshold_met ? '✓ OK' : '⚠ Low'}
                                            </span>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}

            {/* Empty State */}
            {!orders && !results && (
                <div className="card">
                    <div className="empty-state">
                        <div className="empty-state-icon">📦</div>
                        <div className="empty-state-title">No Orders Loaded</div>
                        <div className="empty-state-text">
                            Click "Load Sample Orders" to fetch orders from the synthetic dataset,<br />
                            then click "Optimize" to run batch incentive optimization.
                        </div>
                    </div>
                </div>
            )}

            {/* Orders Preview */}
            {orders && !results && (
                <div className="card">
                    <div className="card-header">
                        <div className="card-title">Loaded Orders Preview</div>
                        <div className="card-subtitle">{orders.length} orders ready for optimization</div>
                    </div>
                    <div className="table-container">
                        <table className="data-table">
                            <thead>
                                <tr>
                                    <th>Order ID</th>
                                    <th>City</th>
                                    <th>Distance</th>
                                    <th>Weather</th>
                                    <th>Traffic</th>
                                    <th>Hour</th>
                                    <th>Order Value</th>
                                    <th>Revenue</th>
                                </tr>
                            </thead>
                            <tbody>
                                {orders.map((o) => (
                                    <tr key={o.order_id}>
                                        <td>{o.order_id}</td>
                                        <td style={{ fontFamily: 'var(--font-sans)' }}>{o.city}</td>
                                        <td>{o.distance_km} km</td>
                                        <td style={{ fontFamily: 'var(--font-sans)' }}>{o.weather}</td>
                                        <td>{o.traffic_level}</td>
                                        <td>{o.hour_of_day}:00</td>
                                        <td>₹{o.order_value}</td>
                                        <td>₹{o.delivery_revenue}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}
        </div>
    );
}

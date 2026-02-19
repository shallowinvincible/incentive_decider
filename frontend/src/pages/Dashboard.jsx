import { useState, useEffect } from 'react';
import { fetchDashboardStats } from '../api';

export default function Dashboard() {
    const [stats, setStats] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        fetchDashboardStats()
            .then(setStats)
            .catch((e) => setError(e.message))
            .finally(() => setLoading(false));
    }, []);

    if (loading) {
        return (
            <div className="loading-state">
                <div className="spinner spinner-lg" />
                <span>Loading dashboard data...</span>
            </div>
        );
    }

    if (error) {
        return (
            <div className="empty-state">
                <div className="empty-state-icon">⚠️</div>
                <div className="empty-state-title">Connection Error</div>
                <div className="empty-state-text">
                    Make sure the backend server is running on port 5000.
                    <br />Error: {error}
                </div>
            </div>
        );
    }

    const { kpis, weather_stats, hour_stats, city_stats, traffic_stats, distance_stats, incentive_histogram } = stats;

    // Prepare hour chart data
    const hourData = Object.entries(hour_stats || {})
        .sort(([a], [b]) => parseInt(a) - parseInt(b))
        .map(([hour, data]) => ({
            label: `${hour}h`,
            incentive: data.incentive_given,
            acceptance: data.order_accepted,
        }));

    const maxIncentiveByHour = Math.max(...hourData.map(d => d.incentive), 1);

    return (
        <div>
            <div className="page-header">
                <h1 className="page-title">📊 Dashboard</h1>
                <p className="page-subtitle">
                    Overview of historical delivery data and incentive patterns across {kpis.total_orders.toLocaleString()} orders
                </p>
            </div>

            {/* KPI Cards */}
            <div className="kpi-grid">
                <div className="kpi-card violet">
                    <div className="kpi-label">Total Orders</div>
                    <div className="kpi-value violet">{kpis.total_orders.toLocaleString()}</div>
                    <div className="kpi-sub">Historical training records</div>
                </div>
                <div className="kpi-card green">
                    <div className="kpi-label">Avg Profit / Order</div>
                    <div className="kpi-value green">₹{kpis.avg_profit}</div>
                    <div className="kpi-sub">Revenue − Incentive</div>
                </div>
                <div className="kpi-card amber">
                    <div className="kpi-label">Avg Incentive</div>
                    <div className="kpi-value amber">₹{kpis.avg_incentive}</div>
                    <div className="kpi-sub">Per delivery order</div>
                </div>
                <div className="kpi-card violet">
                    <div className="kpi-label">Acceptance Rate</div>
                    <div className="kpi-value violet">{kpis.acceptance_rate}%</div>
                    <div className="kpi-sub">Overall rider acceptance</div>
                </div>
                <div className="kpi-card green">
                    <div className="kpi-label">Avg Revenue</div>
                    <div className="kpi-value green">₹{kpis.avg_revenue}</div>
                    <div className="kpi-sub">Platform delivery revenue</div>
                </div>
                <div className="kpi-card amber">
                    <div className="kpi-label">Avg Distance</div>
                    <div className="kpi-value amber">{kpis.avg_distance} km</div>
                    <div className="kpi-sub">Average delivery distance</div>
                </div>
                <div className="kpi-card rose">
                    <div className="kpi-label">Total Profit</div>
                    <div className="kpi-value rose">₹{kpis.total_profit.toLocaleString()}</div>
                    <div className="kpi-sub">Across all historical orders</div>
                </div>
                <div className="kpi-card violet">
                    <div className="kpi-label">Avg Order Value</div>
                    <div className="kpi-value violet">₹{kpis.avg_order_value}</div>
                    <div className="kpi-sub">Customer order amount</div>
                </div>
            </div>

            {/* Charts */}
            <div className="charts-grid">
                {/* Incentive by Weather */}
                <div className="chart-container">
                    <div className="chart-title">📍 Avg Incentive by Weather Condition</div>
                    <div className="bar-chart">
                        {Object.entries(weather_stats || {})
                            .sort(([, a], [, b]) => b.incentive_given - a.incentive_given)
                            .map(([weather, data]) => {
                                const maxIncentive = Math.max(...Object.values(weather_stats).map(d => d.incentive_given));
                                const pct = (data.incentive_given / maxIncentive) * 100;
                                return (
                                    <div className="bar-row" key={weather}>
                                        <div className="bar-label">{weather}</div>
                                        <div className="bar-track">
                                            <div
                                                className="bar-fill violet"
                                                style={{ width: `${pct}%` }}
                                            >
                                                ₹{data.incentive_given.toFixed(0)}
                                            </div>
                                        </div>
                                    </div>
                                );
                            })}
                    </div>
                </div>

                {/* Incentive by Traffic */}
                <div className="chart-container">
                    <div className="chart-title">🚦 Avg Incentive by Traffic Level</div>
                    <div className="bar-chart">
                        {Object.entries(traffic_stats || {})
                            .sort(([, a], [, b]) => b.incentive_given - a.incentive_given)
                            .map(([level, data]) => {
                                const maxIncentive = Math.max(...Object.values(traffic_stats).map(d => d.incentive_given));
                                const pct = (data.incentive_given / maxIncentive) * 100;
                                const colors = ['green', 'blue', 'amber', 'rose'];
                                const idx = ['Low', 'Medium', 'High', 'Very High'].indexOf(level);
                                return (
                                    <div className="bar-row" key={level}>
                                        <div className="bar-label">{level}</div>
                                        <div className="bar-track">
                                            <div
                                                className={`bar-fill ${colors[idx] || 'violet'}`}
                                                style={{ width: `${pct}%` }}
                                            >
                                                ₹{data.incentive_given.toFixed(0)}
                                            </div>
                                        </div>
                                    </div>
                                );
                            })}
                    </div>
                </div>

                {/* Incentive by Distance */}
                <div className="chart-container">
                    <div className="chart-title">📏 Avg Incentive by Distance Range</div>
                    <div className="bar-chart">
                        {Object.entries(distance_stats || {})
                            .map(([range, data]) => {
                                const maxIncentive = Math.max(...Object.values(distance_stats).map(d => d.incentive_given));
                                const pct = (data.incentive_given / maxIncentive) * 100;
                                return (
                                    <div className="bar-row" key={range}>
                                        <div className="bar-label">{range}</div>
                                        <div className="bar-track">
                                            <div
                                                className="bar-fill blue"
                                                style={{ width: `${pct}%` }}
                                            >
                                                ₹{data.incentive_given.toFixed(0)}
                                            </div>
                                        </div>
                                    </div>
                                );
                            })}
                    </div>
                </div>

                {/* Incentive Distribution Histogram */}
                <div className="chart-container">
                    <div className="chart-title">📊 Incentive Distribution (Histogram)</div>
                    <div className="curve-chart-wrapper">
                        {(incentive_histogram || []).map((bin, i) => {
                            const maxCount = Math.max(...incentive_histogram.map(b => b.count));
                            const pct = (bin.count / maxCount) * 100;
                            return (
                                <div
                                    key={i}
                                    className="curve-bar-item"
                                    style={{
                                        height: `${pct}%`,
                                        background: `linear-gradient(180deg, var(--accent-violet) 0%, var(--accent-blue) 100%)`,
                                        opacity: 0.7 + (pct / 100) * 0.3,
                                    }}
                                    title={`${bin.range}: ${bin.count} orders`}
                                />
                            );
                        })}
                    </div>
                    <div className="curve-axis">
                        <span>₹0</span>
                        <span>₹50</span>
                        <span>₹100</span>
                        <span>₹150</span>
                        <span>₹200</span>
                    </div>
                </div>

                {/* Hourly Incentive Pattern */}
                <div className="chart-container" style={{ gridColumn: '1 / -1' }}>
                    <div className="chart-title">🕐 Incentive Pattern by Hour of Day</div>
                    <div className="curve-chart-wrapper" style={{ height: '160px' }}>
                        {hourData.map((d, i) => {
                            const pct = (d.incentive / maxIncentiveByHour) * 100;
                            const isPeak = [11, 12, 13, 19, 20, 21, 22].includes(parseInt(d.label));
                            return (
                                <div
                                    key={i}
                                    className="curve-bar-item"
                                    style={{
                                        height: `${pct}%`,
                                        background: isPeak
                                            ? 'linear-gradient(180deg, var(--accent-rose) 0%, var(--accent-orange) 100%)'
                                            : 'linear-gradient(180deg, var(--accent-emerald) 0%, var(--accent-cyan) 100%)',
                                        opacity: 0.6 + (pct / 100) * 0.4,
                                    }}
                                    title={`${d.label}: ₹${d.incentive.toFixed(0)} avg incentive, ${(d.acceptance * 100).toFixed(1)}% acceptance`}
                                />
                            );
                        })}
                    </div>
                    <div className="curve-axis">
                        <span>0h</span>
                        <span>4h</span>
                        <span>8h</span>
                        <span>12h</span>
                        <span>16h</span>
                        <span>20h</span>
                        <span>23h</span>
                    </div>
                    <div className="curve-legend">
                        <div className="curve-legend-item">
                            <div className="curve-legend-dot" style={{ background: 'var(--accent-emerald)' }} />
                            Off-Peak Hours
                        </div>
                        <div className="curve-legend-item">
                            <div className="curve-legend-dot" style={{ background: 'var(--accent-rose)' }} />
                            Peak Hours (11-14, 19-22)
                        </div>
                    </div>
                </div>

                {/* City Stats */}
                <div className="chart-container" style={{ gridColumn: '1 / -1' }}>
                    <div className="chart-title">🏙️ Performance by City</div>
                    <div className="table-container">
                        <table className="data-table">
                            <thead>
                                <tr>
                                    <th>City</th>
                                    <th>Avg Incentive</th>
                                    <th>Acceptance Rate</th>
                                    <th>Avg Distance</th>
                                    <th>Avg Revenue</th>
                                </tr>
                            </thead>
                            <tbody>
                                {Object.entries(city_stats || {})
                                    .sort(([, a], [, b]) => b.delivery_revenue - a.delivery_revenue)
                                    .map(([city, data]) => (
                                        <tr key={city}>
                                            <td style={{ fontFamily: 'var(--font-sans)', fontWeight: 500 }}>{city}</td>
                                            <td>₹{data.incentive_given.toFixed(0)}</td>
                                            <td>
                                                <span className={`badge ${data.order_accepted >= 0.7 ? 'badge-success' : data.order_accepted >= 0.5 ? 'badge-warning' : 'badge-danger'}`}>
                                                    {(data.order_accepted * 100).toFixed(1)}%
                                                </span>
                                            </td>
                                            <td>{data.distance_km.toFixed(1)} km</td>
                                            <td>₹{data.delivery_revenue.toFixed(0)}</td>
                                        </tr>
                                    ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    );
}

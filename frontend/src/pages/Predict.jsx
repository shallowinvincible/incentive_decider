import { useState } from 'react';
import { predictSingleOrder } from '../api';

const INITIAL_FORM = {
    order_id: 99001,
    city: 'Mumbai',
    zone: 'Central',
    distance_km: 8.5,
    estimated_time_min: 32,
    order_value: 450,
    base_delivery_fee: 45,
    surge_multiplier: 1.5,
    weather: 'Light Rain',
    rain_intensity: 0.35,
    temperature: 28,
    humidity: 75,
    wind_speed: 12,
    traffic_level: 'High',
    hour_of_day: 20,
    day_of_week: 'Friday',
    is_weekend: 0,
    festival_flag: 0,
    restaurant_prep_time: 18,
    historical_rider_speed: 20,
    historical_acceptance_rate_zone: 0.78,
    historical_cancel_rate_zone: 0.08,
    delivery_revenue: 165,
};

const CITIES = ['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 'Pune', 'Kolkata', 'Jaipur'];
const ZONES = ['Central', 'North', 'South', 'East', 'West', 'Suburban', 'Airport', 'Industrial'];
const WEATHER = ['Clear', 'Cloudy', 'Light Rain', 'Heavy Rain', 'Fog', 'Storm'];
const TRAFFIC = ['Low', 'Medium', 'High', 'Very High'];
const DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];

export default function Predict() {
    const [form, setForm] = useState(INITIAL_FORM);
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleChange = (field, value) => {
        const numericFields = [
            'order_id', 'distance_km', 'estimated_time_min', 'order_value', 'base_delivery_fee',
            'surge_multiplier', 'rain_intensity', 'temperature', 'humidity', 'wind_speed',
            'hour_of_day', 'is_weekend', 'festival_flag', 'restaurant_prep_time',
            'historical_rider_speed', 'historical_acceptance_rate_zone', 'historical_cancel_rate_zone',
            'delivery_revenue'
        ];
        if (numericFields.includes(field)) {
            value = parseFloat(value) || 0;
        }
        setForm(prev => ({ ...prev, [field]: value }));
    };

    const handleSubmit = async () => {
        setLoading(true);
        setError(null);
        setResult(null);
        try {
            const data = await predictSingleOrder(form);
            setResult(data);
        } catch (e) {
            setError(e.message);
        } finally {
            setLoading(false);
        }
    };

    const handleReset = () => {
        setForm(INITIAL_FORM);
        setResult(null);
        setError(null);
    };

    // Generate quick presets
    const presets = [
        { name: '☀️ Easy Nearby', values: { distance_km: 2.5, weather: 'Clear', traffic_level: 'Low', hour_of_day: 14, rain_intensity: 0, order_value: 300, delivery_revenue: 95 } },
        { name: '🌧️ Rain + Far', values: { distance_km: 18, weather: 'Heavy Rain', traffic_level: 'High', hour_of_day: 21, rain_intensity: 0.7, order_value: 800, delivery_revenue: 220 } },
        { name: '🌙 Night Storm', values: { distance_km: 12, weather: 'Storm', traffic_level: 'Very High', hour_of_day: 23, rain_intensity: 0.9, order_value: 600, delivery_revenue: 185 } },
        { name: '🎉 Festival Rush', values: { distance_km: 10, weather: 'Cloudy', traffic_level: 'Very High', hour_of_day: 20, festival_flag: 1, surge_multiplier: 2.2, order_value: 1200, delivery_revenue: 340 } },
    ];

    return (
        <div>
            <div className="page-header">
                <h1 className="page-title">🎯 Single Order Prediction</h1>
                <p className="page-subtitle">Enter order details to get the optimal incentive recommendation</p>
            </div>

            {/* Presets */}
            <div style={{ display: 'flex', gap: '8px', marginBottom: '24px', flexWrap: 'wrap' }}>
                <span style={{ fontSize: '13px', color: 'var(--text-muted)', alignSelf: 'center', marginRight: '4px' }}>Quick presets:</span>
                {presets.map((preset) => (
                    <button
                        key={preset.name}
                        className="btn btn-secondary"
                        style={{ padding: '6px 14px', fontSize: '12px' }}
                        onClick={() => setForm(prev => ({ ...prev, ...preset.values }))}
                    >
                        {preset.name}
                    </button>
                ))}
            </div>

            <div className="predict-layout">
                {/* Form */}
                <div className="card">
                    <div className="predict-form-section">
                        <div className="form-section-title">📍 Location & Order</div>
                        <div className="form-grid">
                            <div className="form-group">
                                <label className="form-label">City</label>
                                <select className="form-select" value={form.city} onChange={e => handleChange('city', e.target.value)}>
                                    {CITIES.map(c => <option key={c} value={c}>{c}</option>)}
                                </select>
                            </div>
                            <div className="form-group">
                                <label className="form-label">Zone</label>
                                <select className="form-select" value={form.zone} onChange={e => handleChange('zone', e.target.value)}>
                                    {ZONES.map(z => <option key={z} value={z}>{z}</option>)}
                                </select>
                            </div>
                            <div className="form-group">
                                <label className="form-label">Distance (km)</label>
                                <input className="form-input" type="number" step="0.5" value={form.distance_km} onChange={e => handleChange('distance_km', e.target.value)} />
                            </div>
                            <div className="form-group">
                                <label className="form-label">Est. Time (min)</label>
                                <input className="form-input" type="number" value={form.estimated_time_min} onChange={e => handleChange('estimated_time_min', e.target.value)} />
                            </div>
                            <div className="form-group">
                                <label className="form-label">Order Value (₹)</label>
                                <input className="form-input" type="number" value={form.order_value} onChange={e => handleChange('order_value', e.target.value)} />
                            </div>
                            <div className="form-group">
                                <label className="form-label">Delivery Revenue (₹)</label>
                                <input className="form-input" type="number" value={form.delivery_revenue} onChange={e => handleChange('delivery_revenue', e.target.value)} />
                            </div>
                            <div className="form-group">
                                <label className="form-label">Base Fee (₹)</label>
                                <input className="form-input" type="number" value={form.base_delivery_fee} onChange={e => handleChange('base_delivery_fee', e.target.value)} />
                            </div>
                            <div className="form-group">
                                <label className="form-label">Surge Multiplier</label>
                                <input className="form-input" type="number" step="0.1" value={form.surge_multiplier} onChange={e => handleChange('surge_multiplier', e.target.value)} />
                            </div>
                        </div>

                        <div className="form-section-title">🌦 Weather & Environment</div>
                        <div className="form-grid">
                            <div className="form-group">
                                <label className="form-label">Weather</label>
                                <select className="form-select" value={form.weather} onChange={e => handleChange('weather', e.target.value)}>
                                    {WEATHER.map(w => <option key={w} value={w}>{w}</option>)}
                                </select>
                            </div>
                            <div className="form-group">
                                <label className="form-label">Rain Intensity (0-1)</label>
                                <input className="form-input" type="number" step="0.05" min="0" max="1" value={form.rain_intensity} onChange={e => handleChange('rain_intensity', e.target.value)} />
                            </div>
                            <div className="form-group">
                                <label className="form-label">Temperature (°C)</label>
                                <input className="form-input" type="number" value={form.temperature} onChange={e => handleChange('temperature', e.target.value)} />
                            </div>
                            <div className="form-group">
                                <label className="form-label">Humidity (%)</label>
                                <input className="form-input" type="number" value={form.humidity} onChange={e => handleChange('humidity', e.target.value)} />
                            </div>
                            <div className="form-group">
                                <label className="form-label">Wind Speed (km/h)</label>
                                <input className="form-input" type="number" value={form.wind_speed} onChange={e => handleChange('wind_speed', e.target.value)} />
                            </div>
                            <div className="form-group">
                                <label className="form-label">Traffic Level</label>
                                <select className="form-select" value={form.traffic_level} onChange={e => handleChange('traffic_level', e.target.value)}>
                                    {TRAFFIC.map(t => <option key={t} value={t}>{t}</option>)}
                                </select>
                            </div>
                        </div>

                        <div className="form-section-title">🕐 Time & Flags</div>
                        <div className="form-grid">
                            <div className="form-group">
                                <label className="form-label">Hour of Day (0-23)</label>
                                <input className="form-input" type="number" min="0" max="23" value={form.hour_of_day} onChange={e => handleChange('hour_of_day', e.target.value)} />
                            </div>
                            <div className="form-group">
                                <label className="form-label">Day of Week</label>
                                <select className="form-select" value={form.day_of_week} onChange={e => handleChange('day_of_week', e.target.value)}>
                                    {DAYS.map(d => <option key={d} value={d}>{d}</option>)}
                                </select>
                            </div>
                            <div className="form-group">
                                <label className="form-label">Weekend?</label>
                                <select className="form-select" value={form.is_weekend} onChange={e => handleChange('is_weekend', e.target.value)}>
                                    <option value={0}>No</option>
                                    <option value={1}>Yes</option>
                                </select>
                            </div>
                            <div className="form-group">
                                <label className="form-label">Festival?</label>
                                <select className="form-select" value={form.festival_flag} onChange={e => handleChange('festival_flag', e.target.value)}>
                                    <option value={0}>No</option>
                                    <option value={1}>Yes</option>
                                </select>
                            </div>
                        </div>

                        <div className="form-section-title">🏍️ Rider & Restaurant</div>
                        <div className="form-grid">
                            <div className="form-group">
                                <label className="form-label">Prep Time (min)</label>
                                <input className="form-input" type="number" value={form.restaurant_prep_time} onChange={e => handleChange('restaurant_prep_time', e.target.value)} />
                            </div>
                            <div className="form-group">
                                <label className="form-label">Rider Speed (km/h)</label>
                                <input className="form-input" type="number" value={form.historical_rider_speed} onChange={e => handleChange('historical_rider_speed', e.target.value)} />
                            </div>
                            <div className="form-group">
                                <label className="form-label">Zone Accept Rate</label>
                                <input className="form-input" type="number" step="0.01" min="0" max="1" value={form.historical_acceptance_rate_zone} onChange={e => handleChange('historical_acceptance_rate_zone', e.target.value)} />
                            </div>
                            <div className="form-group">
                                <label className="form-label">Zone Cancel Rate</label>
                                <input className="form-input" type="number" step="0.01" min="0" max="1" value={form.historical_cancel_rate_zone} onChange={e => handleChange('historical_cancel_rate_zone', e.target.value)} />
                            </div>
                        </div>

                        <div style={{ display: 'flex', gap: '12px', marginTop: '8px' }}>
                            <button className="btn btn-primary btn-lg" onClick={handleSubmit} disabled={loading} style={{ flex: 1 }}>
                                {loading ? (
                                    <><div className="spinner" /> Optimizing...</>
                                ) : (
                                    <>⚡ Optimize Incentive</>
                                )}
                            </button>
                            <button className="btn btn-secondary btn-lg" onClick={handleReset}>
                                ↺ Reset
                            </button>
                        </div>
                    </div>
                </div>

                {/* Results Panel */}
                <div>
                    {!result && !error && (
                        <div className="card" style={{ minHeight: '300px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                            <div className="empty-state">
                                <div className="empty-state-icon">🎯</div>
                                <div className="empty-state-title">Ready to Optimize</div>
                                <div className="empty-state-text">
                                    Fill the order details and click "Optimize Incentive" to get the ML-powered recommendation.
                                </div>
                            </div>
                        </div>
                    )}

                    {error && (
                        <div className="card" style={{ borderColor: 'rgba(244, 63, 94, 0.3)' }}>
                            <div className="empty-state">
                                <div className="empty-state-icon">❌</div>
                                <div className="empty-state-title">Prediction Failed</div>
                                <div className="empty-state-text">{error}</div>
                            </div>
                        </div>
                    )}

                    {result && (
                        <>
                            <div className="result-card">
                                <div style={{ textAlign: 'center', marginBottom: '24px' }}>
                                    <span style={{ fontSize: '12px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em', fontWeight: 600 }}>
                                        ML Optimization Result
                                    </span>
                                </div>

                                <div className="result-grid">
                                    <div className="result-item">
                                        <div className="result-icon">💰</div>
                                        <div className="result-label">Recommended Incentive</div>
                                        <div className="result-value incentive">₹{result.recommended_incentive}</div>
                                        <div className="result-sub">Optimal amount</div>
                                    </div>
                                    <div className="result-item">
                                        <div className="result-icon">✅</div>
                                        <div className="result-label">Acceptance Probability</div>
                                        <div className="result-value acceptance">{(result.predicted_acceptance_probability * 100).toFixed(1)}%</div>
                                        <div className="result-sub">
                                            <span className={`badge ${result.threshold_met ? 'badge-success' : 'badge-warning'}`}>
                                                {result.threshold_met ? '✓ Meets 90% threshold' : '⚠ Below threshold'}
                                            </span>
                                        </div>
                                    </div>
                                    <div className="result-item">
                                        <div className="result-icon">📈</div>
                                        <div className="result-label">Expected Profit</div>
                                        <div className="result-value profit">₹{result.expected_profit}</div>
                                        <div className="result-sub">Profit-maximized</div>
                                    </div>
                                    <div className="result-item">
                                        <div className="result-icon">🏷️</div>
                                        <div className="result-label">Delivery Revenue</div>
                                        <div className="result-value revenue">₹{result.delivery_revenue}</div>
                                        <div className="result-sub">Platform earnings</div>
                                    </div>
                                </div>
                            </div>

                            {/* Incentive Curve */}
                            {result.incentive_curve && (
                                <div className="card" style={{ marginTop: '16px' }}>
                                    <div className="chart-title">📉 Incentive vs Acceptance/Profit Curve</div>
                                    <div className="curve-chart-wrapper" style={{ height: '160px' }}>
                                        {result.incentive_curve.map((point, i) => {
                                            const maxProfit = Math.max(...result.incentive_curve.map(p => Math.abs(p.expected_profit)), 1);
                                            const pct = (Math.max(0, point.expected_profit) / maxProfit) * 100;
                                            const isSelected = point.incentive === result.recommended_incentive;
                                            const meetsThreshold = point.acceptance_probability >= 0.9;
                                            return (
                                                <div
                                                    key={i}
                                                    className={`curve-bar-item ${isSelected ? 'selected' : ''}`}
                                                    style={{
                                                        height: `${Math.max(pct, 2)}%`,
                                                        background: isSelected
                                                            ? 'linear-gradient(180deg, var(--accent-violet-light), var(--accent-violet))'
                                                            : meetsThreshold
                                                                ? 'linear-gradient(180deg, var(--accent-emerald), var(--accent-cyan))'
                                                                : 'linear-gradient(180deg, var(--accent-rose), rgba(244, 63, 94, 0.3))',
                                                        opacity: isSelected ? 1 : 0.6,
                                                    }}
                                                    title={`₹${point.incentive}: Profit ₹${point.expected_profit}, Accept ${(point.acceptance_probability * 100).toFixed(1)}%`}
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
                                    <div className="curve-legend">
                                        <div className="curve-legend-item">
                                            <div className="curve-legend-dot" style={{ background: 'var(--accent-emerald)' }} />
                                            Meets ≥90% threshold
                                        </div>
                                        <div className="curve-legend-item">
                                            <div className="curve-legend-dot" style={{ background: 'var(--accent-rose)' }} />
                                            Below threshold
                                        </div>
                                        <div className="curve-legend-item">
                                            <div className="curve-legend-dot" style={{ background: 'var(--accent-violet)' }} />
                                            Optimal choice
                                        </div>
                                    </div>
                                </div>
                            )}
                        </>
                    )}
                </div>
            </div>
        </div>
    );
}

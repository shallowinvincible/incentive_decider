const API_BASE = 'http://localhost:5001/api';

export async function fetchDashboardStats() {
    const res = await fetch(`${API_BASE}/dashboard-stats`);
    if (!res.ok) throw new Error('Failed to fetch dashboard stats');
    return res.json();
}

export async function fetchModelInfo() {
    const res = await fetch(`${API_BASE}/model-info`);
    if (!res.ok) throw new Error('Failed to fetch model info');
    return res.json();
}

export async function predictSingleOrder(orderData) {
    const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(orderData),
    });
    if (!res.ok) throw new Error('Prediction failed');
    return res.json();
}

export async function predictBatch(ordersArray) {
    const res = await fetch(`${API_BASE}/batch-predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(ordersArray),
    });
    if (!res.ok) throw new Error('Batch prediction failed');
    return res.json();
}

export async function fetchSampleOrders(n = 10) {
    const res = await fetch(`${API_BASE}/sample-orders?n=${n}`);
    if (!res.ok) throw new Error('Failed to fetch sample orders');
    return res.json();
}

export async function retrainModel() {
    const res = await fetch(`${API_BASE}/retrain`, { method: 'POST' });
    if (!res.ok) throw new Error('Retrain failed');
    return res.json();
}

export async function healthCheck() {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error('Health check failed');
    return res.json();
}

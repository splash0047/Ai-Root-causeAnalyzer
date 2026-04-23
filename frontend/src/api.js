/**
 * AI Root Cause Analyzer — API Client
 * Centralized API communication with the FastAPI backend.
 */

const API_BASE = 'http://localhost:8000';

async function request(path, options = {}) {
  const url = `${API_BASE}${path}`;
  const res = await fetch(url, {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `API Error: ${res.status}`);
  }

  return res.json();
}

export const api = {
  // Health
  health: () => request('/health'),

  // Metrics
  getMetrics: (windowHours = 24) => request(`/metrics?window_hours=${windowHours}`),

  // RCA
  runRCA: (records, actuals = null, mode = 'deep') =>
    request('/rca', {
      method: 'POST',
      body: JSON.stringify({ records, actuals, mode }),
    }),

  getRCAHistory: (limit = 20, severity = null) => {
    let path = `/rca/history?limit=${limit}`;
    if (severity) path += `&severity=${severity}`;
    return request(path);
  },

  // Feedback
  submitFeedback: (rcaId, feedback, notes = null) =>
    request('/feedback', {
      method: 'POST',
      body: JSON.stringify({ rca_id: rcaId, feedback, notes }),
    }),

  // Simulation
  runSimulation: (failureType, feature = null, feature2 = null, nSamples = 500, noiseFactor = 3.0, runRca = true) =>
    request('/simulate', {
      method: 'POST',
      body: JSON.stringify({
        failure_type: failureType,
        feature,
        feature2,
        n_samples: nSamples,
        noise_factor: noiseFactor,
        run_rca: runRca,
      }),
    }),

  // Ingest
  ingestData: (records, actuals = null, batchId = null) =>
    request('/ingest', {
      method: 'POST',
      body: JSON.stringify({ records, actuals, batch_id: batchId }),
    }),
};

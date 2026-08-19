
const API_BASE_URL = (import.meta.env.VITE_BBB_API_URL || "https://lgbm-bbb-backend.onrender.com").replace(/\/+$/, '');
const confidenceScore = (label) => {
  switch (String(label).toLowerCase()) {
    case 'good':
    case 'high':
      return 0.9;
    case 'moderate':
    case 'medium':
      return 0.7;
    case 'poor/unreliable':
    case 'low':
      return 0.4;
    default:
      return 0;
  }
};

const normalizeProbability = (value) => {
  const numericValue = Number(value);
  if (!Number.isFinite(numericValue)) return 0;
  return numericValue > 1 ? numericValue / 100 : numericValue;
};

const normalizePrediction = (raw = {}) => {
  const confidenceLabel =
    raw.confidence_label ?? raw.Confidence ?? raw.confidence ?? 'Unknown';
  const rawConfidence = Number(raw.confidence_score ?? raw.confidence);
  const confidence = Number.isFinite(rawConfidence)
    ? rawConfidence > 1
      ? rawConfidence / 100
      : rawConfidence
    : confidenceScore(confidenceLabel);
  const positiveProbability = normalizeProbability(
    raw.probability_bbb_positive ??
    raw.probability_bbb_positive_percent ??
    raw.BBB_plus_Probability_Percent ??
    0,
  );
  const status = raw.status ?? raw.Status ?? 'Failed';
  const isSuccess = status === 'Success';

  return {
    name: raw.name ?? raw.Name ?? 'Unknown',
    smiles: raw.smiles ?? raw.SMILES ?? '',
    prediction: raw.prediction ?? raw.Prediction ?? null,
    confidence,
    confidence_label: confidenceLabel,
    probability_bbb_positive: positiveProbability,
    probability_bbb_negative: normalizeProbability(
      raw.probability_bbb_negative ??
      raw.probability_bbb_negative_percent ??
      (isSuccess ? 1 - positiveProbability : 0),
    ),
    interpretation:
      raw.interpretation ??
      (isSuccess
        ? `${confidenceLabel} confidence prediction`
        : raw.status_reason ?? 'Prediction failed'),
    curation_status: raw.curation_status ?? raw.Curation_Status ?? 'N/A',
    generation_status:
      raw.generation_status ?? raw['3D_Generation_Status'] ?? 'N/A',
    status,
    success: isSuccess,
    error: isSuccess ? undefined : raw.error ?? raw.status_reason ?? status,
  };
};

const normalizeBatchResponse = (raw = {}) => ({
  ...raw,
  predictions: Array.isArray(raw.predictions)
    ? raw.predictions.map(normalizePrediction)
    : [],
  summary: raw.summary ?? {},
});

class BBBApiClient {
  constructor() {
    this.baseURL = API_BASE_URL;
  }

  async request(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`;
    
    // Set default headers, but don't override Content-Type if it's explicitly set to null (for FormData)
    const headers = {};
    if (options.headers?.['Content-Type'] !== null) {
      headers['Content-Type'] = 'application/json';
    }
    
    const config = {
      headers: {
        ...headers,
        ...options.headers,
      },
      ...options,
    };

    // Remove null headers (like Content-Type: null for FormData)
    Object.keys(config.headers).forEach(key => {
      if (config.headers[key] === null) {
        delete config.headers[key];
      }
    });

    try {
      const response = await fetch(url, config);
      
      // Check if response is JSON based on content-type header
      const contentType = response.headers.get('content-type');
      const isJson = contentType && contentType.includes('application/json');
      
      let data;
      if (isJson) {
        try {
          data = await response.json();
        } catch (parseError) {
          // If JSON parsing fails, fall back to text
          console.warn('Failed to parse JSON response, falling back to text:', parseError);
          data = await response.text();
        }
      } else {
        // For non-JSON responses (like health checks), return as text
        data = await response.text();
      }
      
      if (!response.ok) {
        const errorMessage =
          (isJson && (data?.message || data?.detail)) ||
          data ||
          `HTTP error! status: ${response.status}`;
        throw new Error(errorMessage);
      }
      
      // For non-JSON responses that are successful, wrap in an object for consistency
      return isJson ? data : { data, status: 'success' };
    } catch (error) {
      // Handle network errors or other fetch failures
      if (error.name === 'TypeError' && error.message.includes('fetch')) {
        console.error('Network error:', error);
        throw new Error('Network connection failed. Please check your internet connection.');
      }
      console.error('API request failed:', error);
      throw error;
    }
  }

  // Single molecule prediction
  async predictSingle(smiles, name = '', threshold = 0.5228) {
    const result = await this.request('/predict/single', {
      method: 'POST',
      body: JSON.stringify({
        smiles,
        name,
        threshold,
      }),
    });
    return normalizePrediction(result);
  }

  // Batch prediction
  async predictBatch(molecules, threshold = 0.5228) {
    const result = await this.request('/predict/batch', {
      method: 'POST',
      body: JSON.stringify({
        molecules,
        threshold,
      }),
    });
    return normalizeBatchResponse(result);
  }

  // File upload prediction
  async predictFile(formData, threshold = 0.5228, smilesColumn = null, nameColumn = null) {
    let endpoint = `/predict/file?threshold=${threshold}`;
    if (smilesColumn) endpoint += `&smiles_column=${encodeURIComponent(smilesColumn)}`;
    if (nameColumn) endpoint += `&name_column=${encodeURIComponent(nameColumn)}`;

    const result = await this.request(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': null, // This signals to not set Content-Type, let browser handle it for FormData
      },
      body: formData,
    });
    // FileUpload.jsx already consumes the historical { data: ... } envelope.
    return { data: normalizeBatchResponse(result) };
  }

  // Model information
  async getModelInfo() {
    return this.request('/model/info');
  }

  // Health check
  async getHealth() {
    return this.request('/health');
  }

  // SMILES validation
  async validateSmiles(smiles) {
    return this.request('/validate/smiles', {
      method: 'POST',
      body: JSON.stringify({ smiles }),
    });
  }
}

export const apiClient = new BBBApiClient();
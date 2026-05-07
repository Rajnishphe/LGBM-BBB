// API client for BBB permeability prediction service
const API_BASE_URL = 'https://lgbm-bbb.onrender.com';

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
        const errorMessage = (isJson && data?.message) || data || `HTTP error! status: ${response.status}`;
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
    return this.request('/predict/single', {
      method: 'POST',
      body: JSON.stringify({
        smiles,
        name,
        threshold,
      }),
    });
  }

  // Batch prediction
  async predictBatch(molecules, threshold = 0.5228) {
    return this.request('/predict/batch', {
      method: 'POST',
      body: JSON.stringify({
        molecules,
        threshold,
      }),
    });
  }

  // File upload prediction
  async predictFile(formData, threshold = 0.5228, smilesColumn = null, nameColumn = null) {
    let endpoint = `/predict/file?threshold=${threshold}`;
    if (smilesColumn) endpoint += `&smiles_column=${encodeURIComponent(smilesColumn)}`;
    if (nameColumn) endpoint += `&name_column=${encodeURIComponent(nameColumn)}`;

    return this.request(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': null, // This signals to not set Content-Type, let browser handle it for FormData
      },
      body: formData,
    });
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
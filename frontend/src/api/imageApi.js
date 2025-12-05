import axios from 'axios'

const API_BASE_URL = import.meta.env.PROD 
  ? '/api' 
  : 'http://localhost:8000/api'

export async function processImage(file, endpoint, params = {}) {
  const formData = new FormData()
  formData.append('file', file)

  // Build query string from params
  const queryParams = new URLSearchParams()
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null) {
      queryParams.append(key, value)
    }
  })

  const queryString = queryParams.toString()
  const url = `${API_BASE_URL}/${endpoint}${queryString ? `?${queryString}` : ''}`

  const response = await axios.post(url, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    responseType: 'blob',
  })

  // Convert blob to data URL
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result)
    reader.onerror = reject
    reader.readAsDataURL(response.data)
  })
}

export async function getOperations() {
  const response = await axios.get(`${API_BASE_URL.replace('/api', '')}/operations`)
  return response.data
}

export async function checkHealth() {
  try {
    const response = await axios.get(API_BASE_URL.replace('/api', ''))
    return response.data
  } catch (error) {
    throw new Error('Backend server is not running')
  }
}

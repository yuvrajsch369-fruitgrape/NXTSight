// Thin client for backend/main.py -- one FastAPI service exposing
// NXTSight's real inference engine. Default matches that file's own
// documented run command (`uvicorn backend.main:app --port 8000`);
// override via VITE_API_BASE_URL for anything else (e.g. a later prod deploy).
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

export class ApiError extends Error {
  status?: number

  constructor(message: string, status?: number) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function extractErrorMessage(response: Response): Promise<string> {
  // FastAPI's HTTPException serializes as {"detail": "..."} -- surface
  // that human-readable message, not the raw JSON envelope around it.
  try {
    const body = await response.json()
    if (typeof body?.detail === 'string') return body.detail
  } catch {
    // not JSON -- fall through to plain text
  }
  return response.statusText
}

async function handle<T>(promise: Promise<Response>): Promise<T> {
  let response: Response
  try {
    response = await promise
  } catch {
    // A network-level failure (backend not running, CORS rejection, etc.)
    // -- fetch() itself throws a generic, unhelpful TypeError for all of
    // these, so we re-frame it as the specific, actionable case it almost
    // always is in dev.
    throw new ApiError(
      `Could not reach the NXTSight backend at ${API_BASE_URL}. Is it running? ` +
        '(uvicorn backend.main:app --port 8000)',
    )
  }
  if (!response.ok) {
    throw new ApiError(await extractErrorMessage(response), response.status)
  }
  return response.json() as Promise<T>
}

function getJSON<T>(path: string): Promise<T> {
  return handle<T>(fetch(`${API_BASE_URL}${path}`))
}

function postJSON<T>(path: string, body: unknown): Promise<T> {
  return handle<T>(
    fetch(`${API_BASE_URL}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  )
}

// No Content-Type set here on purpose -- the browser sets
// multipart/form-data with the correct boundary itself; setting it
// manually strips that boundary and the backend can't parse the upload.
function postFile<T>(path: string, file: File): Promise<T> {
  const form = new FormData()
  form.append('file', file)
  return handle<T>(fetch(`${API_BASE_URL}${path}`, { method: 'POST', body: form }))
}

// ---------------------------------------------------------------------
// Status
// ---------------------------------------------------------------------

export interface StatusResponse {
  execution_path: string
  network_isolated: boolean
  network_isolation_note: string
  ai_hub_models: {
    ocr: { active: boolean; status: string }
    minilm_v2_text_encoder: { active: boolean; status: string }
    whisper_encoder: { active: boolean; status: string }
  }
  call_shield_speech_to_text: {
    tier_1_snapdragon_ai_hub: { active: boolean; status: string }
    tier_2_whisper_cpp: { active: boolean; status: string }
  }
  llm_second_opinion: { scope: string; available: boolean; status: string }
}

export function getStatus(): Promise<StatusResponse> {
  return getJSON<StatusResponse>('/status')
}

// ---------------------------------------------------------------------
// Scam Shield
// ---------------------------------------------------------------------

// Shared shape: classify_scam() / analyze_call() both return exactly this.
export interface ScamVerdict {
  is_scam: boolean
  confidence: number
  reason: string
}

export interface ScamScreenshotVerdict extends ScamVerdict {
  extracted_text: string
}

export function scamShieldText(text: string): Promise<ScamVerdict> {
  return postJSON<ScamVerdict>('/scam-shield/text', { text })
}

export function scamShieldScreenshot(file: File): Promise<ScamScreenshotVerdict> {
  return postFile<ScamScreenshotVerdict>('/scam-shield/screenshot', file)
}

// ---------------------------------------------------------------------
// Money Insight
// ---------------------------------------------------------------------

export interface CategorizedTransaction {
  text: string
  category: string
  confidence: number
  amount: number | null
  direction: 'debit' | 'credit' | null
  source: string
}

export interface MoneyInsightResponse {
  categorized: CategorizedTransaction[]
  insight: string
}

export function moneyInsight(transactions: string[]): Promise<MoneyInsightResponse> {
  return postJSON<MoneyInsightResponse>('/money-insight/transactions', { transactions })
}

// ---------------------------------------------------------------------
// Receipt Scanner
// ---------------------------------------------------------------------

export interface ReceiptResult {
  ok: true
  transaction_text: string
  merchant: string | null
  amount: number | null
  date: string | null
  raw_text: string
}

export function spendScannerScreenshot(file: File): Promise<ReceiptResult> {
  return postFile<ReceiptResult>('/spend-scanner/screenshot', file)
}

// ---------------------------------------------------------------------
// Call Shield
// ---------------------------------------------------------------------

export function callShieldTranscript(transcript: string): Promise<ScamVerdict> {
  return postJSON<ScamVerdict>('/call-shield/transcript', { transcript })
}

export interface CallRecordingVerdict extends ScamVerdict {
  transcript: string
}

export function callShieldRecording(file: File): Promise<CallRecordingVerdict> {
  return postFile<CallRecordingVerdict>('/call-shield/recording', file)
}

// ---------------------------------------------------------------------
// Payment Pause
// ---------------------------------------------------------------------

export interface PaymentFlag {
  source: string
  reason: string
  confidence: number
  age: string
}

export interface PaymentPauseLog {
  window_minutes: number
  total_flags: number
  flags: PaymentFlag[]
}

export function paymentPauseLog(): Promise<PaymentPauseLog> {
  return getJSON<PaymentPauseLog>('/payment-pause/log')
}

export type PaymentPauseConfirmResponse =
  | { paused: true; window_minutes: number; flags: PaymentFlag[] }
  | { paused: false; window_minutes: number; message: string }

export function paymentPauseConfirm(): Promise<PaymentPauseConfirmResponse> {
  return postJSON<PaymentPauseConfirmResponse>('/payment-pause/confirm', {})
}

export function paymentPauseReset(): Promise<{ ok: boolean; message: string }> {
  return postJSON<{ ok: boolean; message: string }>('/payment-pause/reset', {})
}

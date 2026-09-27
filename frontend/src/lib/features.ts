// Mirrors app.py's tab order/icons exactly (scam_tab, spend_tab, receipt_tab,
// call_tab, payment_tab) -- this is real NXTSight content, not placeholder copy.
export interface Feature {
  symbol: string
  name: string
  description: string
  accent: 'violet' | 'cyan' | 'green' | 'amber'
  path: string
}

export const FEATURES: Feature[] = [
  {
    symbol: '◈',
    name: 'Scam Shield',
    description: 'Paste a message or drop a screenshot — get a scam/legit verdict with the exact phrases that triggered it, on-device.',
    accent: 'violet',
    path: '/scam-shield',
  },
  {
    symbol: '¤',
    name: 'Money Insight',
    description: 'Turn a wall of bank SMS into a plain-language picture of where your money actually goes.',
    accent: 'cyan',
    path: '/money-insight',
  },
  {
    symbol: '▤',
    name: 'Receipt Scanner',
    description: 'Photograph a receipt — OCR reads the merchant, amount, and date straight into Money Insight.',
    accent: 'green',
    path: '/receipt-scanner',
  },
  {
    symbol: '☎',
    name: 'Call Shield',
    description: 'A live call recording, transcribed and screened for scam patterns as it happens.',
    accent: 'amber',
    path: '/call-shield',
  },
  {
    symbol: '⏸',
    name: 'Payment Pause',
    description: 'A recent scam flag stays visible right up until money would actually move.',
    accent: 'violet',
    path: '/payment-pause',
  },
]

import { motion } from 'motion/react'
import Layout from '@/components/Layout'
import FileDropzone from '@/components/FileDropzone'
import ErrorBanner from '@/components/ErrorBanner'
import { useAsyncAction } from '@/hooks/useAsyncAction'
import { spendScannerScreenshot } from '@/lib/api'

export default function ReceiptScannerPage() {
  const [state, run] = useAsyncAction(spendScannerScreenshot)

  return (
    <Layout icon="▤" title="Receipt Scanner" subtitle="Photograph a receipt — OCR reads the merchant, amount, and date straight into a transaction entry.">
      <FileDropzone accept="image/*" label="Upload a receipt or payment-confirmation screenshot" onSelect={(file) => run(file)} />

      {state.status === 'loading' && <p className="text-sm text-muted-foreground">Reading the receipt…</p>}
      {state.status === 'error' && <ErrorBanner message={state.message} />}

      {state.status === 'success' && (
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
          className="grid grid-cols-1 gap-4 sm:grid-cols-3"
        >
          {[
            { label: 'Merchant', value: state.data.merchant ?? '—' },
            { label: 'Amount', value: state.data.amount != null ? `₹${state.data.amount.toLocaleString('en-IN')}` : '—' },
            { label: 'Date', value: state.data.date ?? '—' },
          ].map((stat) => (
            <div key={stat.label} className="rounded-2xl border border-border bg-card p-4">
              <div className="text-xs uppercase tracking-wide text-muted-foreground">{stat.label}</div>
              <div className="mt-1 font-display text-lg font-semibold">{stat.value}</div>
            </div>
          ))}
          <div className="rounded-2xl border border-border bg-card p-4 text-sm sm:col-span-3">
            <div className="mb-1 text-xs uppercase tracking-wide text-muted-foreground">Built transaction text</div>
            {state.data.transaction_text}
          </div>
        </motion.div>
      )}
    </Layout>
  )
}

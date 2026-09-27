import { useEffect } from 'react'
import { motion, AnimatePresence } from 'motion/react'
import Layout from '@/components/Layout'
import Button from '@/components/Button'
import ErrorBanner from '@/components/ErrorBanner'
import { useAsyncAction } from '@/hooks/useAsyncAction'
import { paymentPauseLog, paymentPauseConfirm, paymentPauseReset } from '@/lib/api'

export default function PaymentPausePage() {
  const [logState, runLog] = useAsyncAction(paymentPauseLog)
  const [confirmState, runConfirm] = useAsyncAction(paymentPauseConfirm)
  const [resetState, runReset] = useAsyncAction(paymentPauseReset)

  useEffect(() => {
    runLog()
  }, [runLog])

  async function handleReset() {
    await runReset()
    runLog()
  }

  return (
    <Layout icon="⏸" title="Payment Pause" subtitle="A recent scam flag stays visible right up until money would actually move.">
      <div className="flex flex-wrap gap-3">
        <Button onClick={() => runConfirm()} disabled={confirmState.status === 'loading'}>
          {confirmState.status === 'loading' ? 'Simulating…' : 'Simulate payment confirmation'}
        </Button>
        <Button variant="secondary" onClick={handleReset} disabled={resetState.status === 'loading'}>
          {resetState.status === 'loading' ? 'Clearing…' : 'Reset log'}
        </Button>
      </div>

      {confirmState.status === 'error' && <ErrorBanner message={confirmState.message} />}
      <AnimatePresence>
        {confirmState.status === 'success' && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.3 }}
            className="overflow-hidden"
          >
            <div
              className="rounded-2xl border p-4 text-sm"
              style={{
                borderColor: confirmState.data.paused ? 'rgba(239, 68, 68, 0.4)' : 'rgba(34, 197, 94, 0.4)',
                background: confirmState.data.paused ? 'rgba(239, 68, 68, 0.08)' : 'rgba(34, 197, 94, 0.08)',
              }}
            >
              {confirmState.data.paused
                ? `⏸ Payment paused — a flag was raised in the last ${confirmState.data.window_minutes} minutes.`
                : `✓ ${confirmState.data.message}`}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {logState.status === 'loading' && <p className="text-sm text-muted-foreground">Loading flag log…</p>}
      {logState.status === 'error' && <ErrorBanner message={logState.message} />}

      {logState.status === 'success' && (
        <div className="flex flex-col gap-3">
          <p className="text-xs uppercase tracking-wide text-muted-foreground">
            {logState.data.total_flags} flag{logState.data.total_flags === 1 ? '' : 's'} logged · {logState.data.window_minutes}-minute pause window
          </p>
          {logState.data.flags.length === 0 && (
            <p className="rounded-xl border border-border bg-card p-4 text-sm text-muted-foreground">
              No flags yet — a scam verdict from Scam Shield or Call Shield logs here automatically.
            </p>
          )}
          {logState.data.flags.map((flag, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, x: -8 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.05 }}
              className="flex items-center justify-between gap-4 rounded-xl border p-4 text-sm"
              style={{ borderColor: 'rgba(239, 68, 68, 0.3)', background: 'rgba(239, 68, 68, 0.06)' }}
            >
              <div className="min-w-0 flex-1">
                <div className="font-medium">{flag.source}</div>
                <div className="truncate text-muted-foreground">{flag.reason}</div>
              </div>
              <div className="shrink-0 text-right text-xs text-muted-foreground">
                <div>{(flag.confidence * 100).toFixed(0)}%</div>
                <div>{flag.age}</div>
              </div>
            </motion.div>
          ))}
        </div>
      )}
    </Layout>
  )
}

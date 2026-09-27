import { motion } from 'motion/react'
import type { ReactNode } from 'react'
import ConfidenceMeter from './ConfidenceMeter'
import type { ScamVerdict } from '@/lib/api'

// Shared result display for Scam Shield and Call Shield -- both return
// the identical {is_scam, confidence, reason} shape from the same
// classify_scam()-based engine call.
export default function VerdictCard({ verdict, extra }: { verdict: ScamVerdict; extra?: ReactNode }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
      className="flex flex-col gap-4 rounded-2xl border p-6"
      style={{
        borderColor: verdict.is_scam ? 'rgba(239, 68, 68, 0.4)' : 'rgba(34, 197, 94, 0.4)',
        background: verdict.is_scam ? 'rgba(239, 68, 68, 0.08)' : 'rgba(34, 197, 94, 0.08)',
      }}
    >
      <div className="flex items-center gap-3">
        <span
          className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full text-lg font-bold"
          style={{
            background: verdict.is_scam ? 'var(--nxt-red)' : 'var(--nxt-green)',
            color: '#0a0e17',
          }}
        >
          {verdict.is_scam ? '!' : '✓'}
        </span>
        <h3 className="font-display text-xl font-semibold">{verdict.is_scam ? 'Likely a scam' : 'Looks legitimate'}</h3>
      </div>
      <p className="text-sm leading-relaxed text-muted-foreground">{verdict.reason}</p>
      <ConfidenceMeter confidence={verdict.confidence} danger={verdict.is_scam} />
      {extra}
    </motion.div>
  )
}

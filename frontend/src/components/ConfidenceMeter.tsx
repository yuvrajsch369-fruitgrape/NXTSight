import { motion } from 'motion/react'

// Direct port of assets/theme.css's .nxt-meter-* rules and src/ui/theme.py's
// meter() helper -- same visual language, now driven by real API confidence
// values instead of a Streamlit call.
export default function ConfidenceMeter({ confidence, danger }: { confidence: number; danger: boolean }) {
  const pct = Math.max(0, Math.min(1, confidence)) * 100
  return (
    <div>
      <div className="mb-1 flex justify-between text-xs uppercase tracking-wide text-muted-foreground">
        <span>Confidence</span>
        <span>{pct.toFixed(0)}%</span>
      </div>
      <div className="h-2 overflow-hidden rounded-full border border-border bg-white/5">
        <motion.div
          className="h-full rounded-full"
          style={{
            background: danger
              ? 'linear-gradient(90deg, #7f1d1d, var(--nxt-red))'
              : 'linear-gradient(90deg, #14532d, var(--nxt-green))',
          }}
          initial={{ width: '0%' }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
        />
      </div>
    </div>
  )
}

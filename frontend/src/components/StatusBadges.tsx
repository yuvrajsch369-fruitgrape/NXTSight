import { motion } from 'motion/react'
import { useStatus } from '@/hooks/useStatus'

export interface StatusBadge {
  label: string
  variant: 'npu' | 'cpu' | 'accel' | 'safe'
}

const VARIANT_STYLES: Record<StatusBadge['variant'], string> = {
  npu: 'border-[#22c55e66] bg-[#22c55e1a] text-[#6ee7a3]',
  cpu: 'border-[#60a5fa66] bg-[#60a5fa1a] text-[#93c5fd]',
  accel: 'border-[#f59e0b66] bg-[#f59e0b1a] text-[#fcd34d]',
  safe: 'border-[#22c55e66] bg-[#22c55e1a] text-[#6ee7a3]',
}

// Lightweight mirror of badge_variant_for() in src/pipeline/runtime.py --
// the frontend only ever receives the final description string from
// /status, not the full provider-priority table, so this pattern-matches
// on that string rather than re-implementing the real classification.
function variantForExecutionPath(path: string): StatusBadge['variant'] {
  if (path.startsWith('Snapdragon NPU')) return 'npu'
  if (path.startsWith('CPU')) return 'cpu'
  return 'accel'
}

function BadgePill({ label, variant }: StatusBadge) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-3.5 py-1.5 text-sm font-medium ${VARIANT_STYLES[variant]}`}
    >
      {variant === 'npu' && (
        <motion.span
          className="h-1.5 w-1.5 rounded-full bg-current"
          animate={{ opacity: [1, 0.4, 1] }}
          transition={{ duration: 2.2, repeat: Infinity, ease: 'easeInOut' }}
        />
      )}
      {label}
    </span>
  )
}

// Real data from backend/main.py's GET /status -- the same live,
// self-verifying execution-path + network-isolation proof app.py's badge
// row shows, now over HTTP instead of an in-process Python call.
export default function StatusBadges() {
  const status = useStatus()

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.2, ease: [0.22, 1, 0.36, 1] }}
      className="flex flex-wrap items-center justify-center gap-2.5"
    >
      {status.status === 'loading' && (
        <span className="inline-flex items-center gap-1.5 rounded-full border border-border bg-card px-3.5 py-1.5 text-sm text-muted-foreground">
          <motion.span
            className="h-1.5 w-1.5 rounded-full bg-current"
            animate={{ opacity: [1, 0.3, 1] }}
            transition={{ duration: 1, repeat: Infinity, ease: 'easeInOut' }}
          />
          Connecting to backend…
        </span>
      )}

      {status.status === 'error' && (
        <span className="inline-flex items-center gap-1.5 rounded-full border border-[#ef444466] bg-[#ef44441a] px-3.5 py-1.5 text-sm text-[#fca5a5]">
          ⚠ {status.message}
        </span>
      )}

      {status.status === 'ready' && (
        <>
          <BadgePill label={`» ${status.data.execution_path}`} variant={variantForExecutionPath(status.data.execution_path)} />
          <BadgePill
            label={status.data.network_isolated ? '✓ Network blocked & verified — inference needs no network' : '⚠ Network isolation not verified'}
            variant={status.data.network_isolated ? 'safe' : 'accel'}
          />
        </>
      )}
    </motion.div>
  )
}

import { motion, useMotionTemplate, useMotionValue } from 'motion/react'
import type { MouseEvent } from 'react'
import { Link } from 'react-router-dom'
import type { Feature } from '@/lib/features'

const ACCENT_VAR: Record<Feature['accent'], string> = {
  violet: 'var(--nxt-violet)',
  cyan: 'var(--nxt-cyan)',
  green: 'var(--nxt-green)',
  amber: 'var(--nxt-amber)',
}

// Two 21st.dev-style techniques combined: an animated conic-gradient
// "shine border" ring (the same rotating-gradient trick assets/theme.css
// already uses on .nxt-hero::before, reused here at card scale) and a
// cursor-following spotlight highlight, tracked via motion values so the
// mouse-move handler never triggers a React re-render.
export default function FeatureCard({ feature, index }: { feature: Feature; index: number }) {
  const mouseX = useMotionValue(0)
  const mouseY = useMotionValue(0)
  const spotlight = useMotionTemplate`radial-gradient(240px circle at ${mouseX}px ${mouseY}px, ${ACCENT_VAR[feature.accent]}22, transparent 75%)`

  function handleMouseMove(e: MouseEvent<HTMLAnchorElement>) {
    const rect = e.currentTarget.getBoundingClientRect()
    mouseX.set(e.clientX - rect.left)
    mouseY.set(e.clientY - rect.top)
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: index * 0.08, ease: [0.22, 1, 0.36, 1] }}
      whileHover={{ y: -6 }}
      className="group relative overflow-hidden rounded-2xl p-px"
    >
      <motion.div
        className="absolute inset-[-80%] opacity-30 transition-opacity duration-300 group-hover:opacity-70"
        style={{ background: `conic-gradient(from 0deg, ${ACCENT_VAR[feature.accent]}, transparent 30%, transparent 70%, ${ACCENT_VAR[feature.accent]})` }}
        animate={{ rotate: 360 }}
        transition={{ duration: 14, repeat: Infinity, ease: 'linear' }}
      />

      <Link
        to={feature.path}
        onMouseMove={handleMouseMove}
        className="relative flex h-full flex-col gap-3 overflow-hidden rounded-2xl border border-border bg-card p-6 shadow-[0_10px_30px_rgba(0,0,0,0.55)] transition-colors duration-300"
      >
        <motion.div className="pointer-events-none absolute inset-0" style={{ background: spotlight }} />

        <span
          className="relative flex h-11 w-11 items-center justify-center rounded-xl border border-border text-xl"
          style={{ color: ACCENT_VAR[feature.accent] }}
        >
          {feature.symbol}
        </span>
        <h3 className="relative font-display text-lg font-semibold">{feature.name}</h3>
        <p className="relative text-sm leading-relaxed text-muted-foreground">{feature.description}</p>
      </Link>
    </motion.div>
  )
}

import { motion } from 'motion/react'

// Same content + intent as app.py's vision_cta() -- href is a placeholder
// until the React app has real routing; not wired to a page yet.
export default function VisionCallout() {
  return (
    <motion.a
      href="#future-vision"
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
      whileHover={{ y: -3 }}
      className="relative block overflow-hidden rounded-2xl border p-6"
      style={{
        borderColor: 'rgba(124, 92, 255, 0.45)',
        background:
          'linear-gradient(135deg, rgba(124, 92, 255, 0.14), rgba(34, 211, 238, 0.08)), var(--nxt-panel)',
      }}
    >
      <motion.div
        className="pointer-events-none absolute inset-0"
        animate={{ boxShadow: ['0 0 0px rgba(124,92,255,0)', '0 0 32px rgba(124,92,255,0.28)', '0 0 0px rgba(124,92,255,0)'] }}
        transition={{ duration: 3.2, repeat: Infinity, ease: 'easeInOut' }}
      />
      <span className="mr-2 inline-block rounded-full bg-gradient-to-r from-[var(--nxt-violet)] to-[var(--nxt-cyan)] px-2 py-0.5 text-[0.68rem] font-bold uppercase tracking-wider text-black">
        Beyond the demo
      </span>
      <span className="text-sm text-muted-foreground">where NXTSight goes from here — not just today's build.</span>
      <div className="mt-1.5 font-display text-lg font-semibold">Future Vision  →</div>
    </motion.a>
  )
}

import { motion } from 'motion/react'

// Ambient, slowly-drifting glow blobs -- an animated evolution of the
// static radial-gradient blobs already in assets/theme.css's .stApp rule.
// Fixed + pointer-events:none so it never interferes with real content.
export default function AuroraBackground() {
  return (
    <div className="pointer-events-none fixed inset-0 -z-10 overflow-hidden bg-background">
      <motion.div
        className="absolute -top-40 left-[10%] h-[36rem] w-[36rem] rounded-full opacity-20 blur-[120px]"
        style={{ background: 'radial-gradient(circle, var(--nxt-violet), transparent 70%)' }}
        animate={{ x: [0, 60, -20, 0], y: [0, 40, -30, 0] }}
        transition={{ duration: 24, repeat: Infinity, ease: 'easeInOut' }}
      />
      <motion.div
        className="absolute top-1/3 right-[5%] h-[30rem] w-[30rem] rounded-full opacity-15 blur-[120px]"
        style={{ background: 'radial-gradient(circle, var(--nxt-cyan), transparent 70%)' }}
        animate={{ x: [0, -50, 30, 0], y: [0, -30, 40, 0] }}
        transition={{ duration: 28, repeat: Infinity, ease: 'easeInOut' }}
      />
      <motion.div
        className="absolute bottom-[-10%] left-1/3 h-[28rem] w-[28rem] rounded-full opacity-10 blur-[120px]"
        style={{ background: 'radial-gradient(circle, var(--nxt-violet), transparent 70%)' }}
        animate={{ x: [0, 30, -40, 0], y: [0, -20, 20, 0] }}
        transition={{ duration: 32, repeat: Infinity, ease: 'easeInOut' }}
      />
      {/* faint fixed grid -- the "scanning surface" cue, common to
          futuristic security/AI dashboards, kept subtle so it reads as
          texture, not noise */}
      <div
        className="absolute inset-0 opacity-[0.035]"
        style={{
          backgroundImage:
            'linear-gradient(rgba(255,255,255,0.6) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.6) 1px, transparent 1px)',
          backgroundSize: '64px 64px',
        }}
      />
    </div>
  )
}

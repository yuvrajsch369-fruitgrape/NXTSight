import { motion } from 'motion/react'

// Same copy as app.py's hero() call -- real NXTSight content, elevated
// with a rotating conic-gradient halo (the same technique as .nxt-hero's
// ::before in assets/theme.css, ported to motion) and a floating icon.
export default function Hero() {
  return (
    <div className="relative flex flex-col items-center gap-5 py-20 text-center">
      <motion.div
        className="relative flex h-20 w-20 items-center justify-center"
        animate={{ y: [0, -10, 0], rotate: [0, -3, 0] }}
        transition={{ duration: 3.4, repeat: Infinity, ease: 'easeInOut' }}
      >
        <motion.div
          className="absolute inset-[-30%] rounded-full opacity-60"
          style={{
            background:
              'conic-gradient(from 0deg, var(--nxt-violet), var(--nxt-cyan), var(--nxt-violet))',
            filter: 'blur(20px)',
          }}
          animate={{ rotate: 360 }}
          transition={{ duration: 18, repeat: Infinity, ease: 'linear' }}
        />
        <span className="relative text-5xl drop-shadow-[0_6px_16px_rgba(124,92,255,0.55)]">⟡</span>
      </motion.div>

      <motion.h1
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
        className="bg-gradient-to-r from-[var(--nxt-violet)] to-[var(--nxt-cyan)] bg-clip-text text-5xl font-bold text-transparent sm:text-6xl"
      >
        NXTSight
      </motion.h1>

      <motion.p
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.1, ease: [0.22, 1, 0.36, 1] }}
        className="max-w-xl text-balance text-lg text-muted-foreground"
      >
        Your on-device shield against financial fraud — no cloud, no account, nothing leaves the device.
      </motion.p>
    </div>
  )
}

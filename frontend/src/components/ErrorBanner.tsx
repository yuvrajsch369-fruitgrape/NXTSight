import { motion } from 'motion/react'

export default function ErrorBanner({ message }: { message: string }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: -8 }}
      animate={{ opacity: 1, y: 0 }}
      className="rounded-xl border border-[#ef444466] bg-[#ef44441a] px-4 py-3 text-sm text-[#fca5a5]"
    >
      ⚠ {message}
    </motion.div>
  )
}

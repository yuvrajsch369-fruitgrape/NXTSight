import { motion } from 'motion/react'
import { Link } from 'react-router-dom'
import type { ReactNode } from 'react'

export default function Layout({
  icon,
  title,
  subtitle,
  children,
}: {
  icon: string
  title: string
  subtitle: string
  children: ReactNode
}) {
  return (
    <main className="mx-auto flex min-h-screen max-w-3xl flex-col gap-8 px-6 pb-24 pt-10">
      <Link
        to="/"
        className="inline-flex w-fit items-center gap-1.5 text-sm text-muted-foreground transition-colors hover:text-foreground"
      >
        ← Back to NXTSight
      </Link>

      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
        className="flex items-center gap-4"
      >
        <span className="flex h-14 w-14 items-center justify-center rounded-2xl border border-border bg-card text-3xl">
          {icon}
        </span>
        <div>
          <h1 className="font-display text-3xl font-bold">{title}</h1>
          <p className="text-muted-foreground">{subtitle}</p>
        </div>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.1, ease: [0.22, 1, 0.36, 1] }}
        className="flex flex-col gap-6"
      >
        {children}
      </motion.div>
    </main>
  )
}

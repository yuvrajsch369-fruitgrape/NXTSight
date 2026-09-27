import { motion } from 'motion/react'

// Direct port of assets/theme.css's .stTabs styling (pill-shaped active
// tab, gradient fill) -- same visual language, generic over any two
// string-labeled options.
export default function TabSwitcher<T extends string>({
  options,
  value,
  onChange,
}: {
  options: readonly T[]
  value: T
  onChange: (value: T) => void
}) {
  return (
    <div className="inline-flex gap-1.5 rounded-2xl border border-border bg-card p-1.5">
      {options.map((option) => (
        <button
          key={option}
          type="button"
          onClick={() => onChange(option)}
          className="relative rounded-xl px-4 py-2 text-sm font-medium transition-colors"
          style={{ color: value === option ? '#0a0e17' : 'var(--nxt-text-muted)' }}
        >
          {value === option && (
            <motion.span
              layoutId="tab-highlight"
              className="absolute inset-0 rounded-xl"
              style={{ background: 'linear-gradient(135deg, var(--nxt-violet), var(--nxt-cyan))' }}
              transition={{ duration: 0.3, ease: [0.22, 1, 0.36, 1] }}
            />
          )}
          <span className="relative">{option}</span>
        </button>
      ))}
    </div>
  )
}

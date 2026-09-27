import { motion } from 'motion/react'
import type { ButtonHTMLAttributes } from 'react'

type Variant = 'primary' | 'secondary'

// motion.button's animation-callback props (onAnimationStart etc.) have a
// different signature than the native DOM event handlers of the same
// name -- omit those specifically so they don't collide when spreading
// native button props onto it.
type NativeButtonProps = Omit<
  ButtonHTMLAttributes<HTMLButtonElement>,
  'onAnimationStart' | 'onAnimationEnd' | 'onAnimationIteration' | 'onDrag' | 'onDragEnd' | 'onDragStart'
>

// Mirrors assets/theme.css's .stButton rules -- gradient primary,
// panel-toned secondary, both with the same lift-on-hover.
export default function Button({
  variant = 'primary',
  className = '',
  children,
  ...props
}: { variant?: Variant } & NativeButtonProps) {
  return (
    <motion.button
      whileHover={props.disabled ? undefined : { y: -2 }}
      whileTap={props.disabled ? undefined : { y: 0 }}
      className={`rounded-xl px-5 py-2.5 text-sm font-medium transition-shadow disabled:cursor-not-allowed disabled:opacity-50 ${
        variant === 'primary'
          ? 'text-[#0a0e17]'
          : 'border border-[var(--nxt-border-strong)] bg-card text-foreground hover:border-[var(--nxt-violet)]'
      } ${className}`}
      style={variant === 'primary' ? { background: 'linear-gradient(135deg, var(--nxt-violet), var(--nxt-cyan))' } : undefined}
      {...props}
    >
      {children}
    </motion.button>
  )
}

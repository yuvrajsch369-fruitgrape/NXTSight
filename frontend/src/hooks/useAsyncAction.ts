import { useCallback, useState } from 'react'
import { ApiError } from '@/lib/api'

export type AsyncState<T> =
  | { status: 'idle' }
  | { status: 'loading' }
  | { status: 'error'; message: string }
  | { status: 'success'; data: T }

// Every screen does the same thing: call an API function, track
// idle/loading/error/success. Shared here once rather than five
// near-identical useState+try/catch blocks across the feature pages.
export function useAsyncAction<T, Args extends unknown[]>(fn: (...args: Args) => Promise<T>) {
  const [state, setState] = useState<AsyncState<T>>({ status: 'idle' })

  const run = useCallback(
    async (...args: Args) => {
      setState({ status: 'loading' })
      try {
        const data = await fn(...args)
        setState({ status: 'success', data })
      } catch (err) {
        setState({ status: 'error', message: err instanceof ApiError ? err.message : 'Something went wrong.' })
      }
    },
    [fn],
  )

  return [state, run] as const
}

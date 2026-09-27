import { useEffect, useState } from 'react'
import { ApiError, getStatus, type StatusResponse } from '@/lib/api'

type StatusState =
  | { status: 'loading' }
  | { status: 'error'; message: string }
  | { status: 'ready'; data: StatusResponse }

export function useStatus(): StatusState {
  const [state, setState] = useState<StatusState>({ status: 'loading' })

  useEffect(() => {
    let cancelled = false
    getStatus()
      .then((data) => {
        if (!cancelled) setState({ status: 'ready', data })
      })
      .catch((err) => {
        if (!cancelled) {
          setState({ status: 'error', message: err instanceof ApiError ? err.message : 'Unknown error' })
        }
      })
    return () => {
      cancelled = true
    }
  }, [])

  return state
}

import { useState } from 'react'
import { motion } from 'motion/react'
import Layout from '@/components/Layout'
import Button from '@/components/Button'
import ErrorBanner from '@/components/ErrorBanner'
import { useAsyncAction } from '@/hooks/useAsyncAction'
import { moneyInsight } from '@/lib/api'

// Same sample batch as app.py's SAMPLE_TRANSACTIONS.
const SAMPLE_TRANSACTIONS = `Rs 450.00 debited from A/c XX1234 on 12-Sep-25 at SWIGGY BANGALORE. Avl Bal Rs 12,340.50
Rs 3,499.00 debited from A/c XX1234 at AMAZON on 12-Sep-25. Avl Bal Rs 11,200.
Rs 240.00 debited via UPI to UBER INDIA on 12-Sep-25. UPI Ref No 334455667788.
You have received Rs 45,000.00 in your account XX4521 via NEFT from ABC CORP PVT LTD SALARY on 01-Sep-25. Avl Bal: Rs 63,200.
Rs 649.00 debited from A/c XX1234 towards NETFLIX SUBSCRIPTION on 05-Sep-25. Avl Bal Rs 14,100.
Rs 5,000.00 withdrawn from A/c XX1234 at SBI ATM MG ROAD on 12-Sep-25. Avl Bal Rs 9,200.
Rs 5,000.00 debited from A/c XX1234 towards SIP MUTUAL FUND ZERODHA on 05-Sep-25. Avl Bal Rs 22,300.
Rs 1,240.00 debited from A/c XX1234 towards BESCOM ELECTRICITY BILL on 12-Sep-25. Avl Bal Rs 9,760.`

export default function MoneyInsightPage() {
  const [text, setText] = useState('')
  const [state, run] = useAsyncAction(moneyInsight)

  function handleAnalyze() {
    const lines = text
      .split('\n')
      .map((l) => l.trim())
      .filter(Boolean)
    run(lines)
  }

  return (
    <Layout icon="¤" title="Money Insight" subtitle="Turn a wall of bank SMS into a plain-language picture of where your money goes.">
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Paste one transaction message per line..."
        rows={8}
        className="rounded-2xl border border-border bg-card p-4 font-mono text-xs outline-none transition-colors focus:border-[var(--nxt-violet)]"
      />
      <div className="flex flex-wrap gap-3">
        <Button onClick={handleAnalyze} disabled={!text.trim() || state.status === 'loading'}>
          {state.status === 'loading' ? 'Analyzing…' : 'Analyze spending'}
        </Button>
        <Button variant="secondary" onClick={() => setText(SAMPLE_TRANSACTIONS)}>
          Load sample transactions
        </Button>
      </div>

      {state.status === 'error' && <ErrorBanner message={state.message} />}

      {state.status === 'success' && (
        <div className="flex flex-col gap-4">
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            className="rounded-2xl border border-border bg-card p-4 text-sm"
          >
            {state.data.insight}
          </motion.div>
          <div className="flex flex-col gap-2">
            {state.data.categorized.map((item, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, x: -8 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.04 }}
                className="flex items-center justify-between gap-4 rounded-xl border border-border bg-card px-4 py-3 text-sm"
              >
                <span className="min-w-0 flex-1 truncate text-muted-foreground">{item.text}</span>
                <span
                  className="shrink-0 rounded-full border border-border px-2.5 py-1 text-xs font-medium"
                  style={{ color: item.category === 'Unrecognized' ? 'var(--nxt-text-muted)' : 'var(--nxt-cyan)' }}
                >
                  {item.category}
                </span>
                {item.amount != null && <span className="w-24 shrink-0 text-right font-mono">₹{item.amount.toLocaleString('en-IN')}</span>}
              </motion.div>
            ))}
          </div>
        </div>
      )}
    </Layout>
  )
}

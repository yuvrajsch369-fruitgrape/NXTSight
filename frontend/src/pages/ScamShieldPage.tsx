import { useState } from 'react'
import Layout from '@/components/Layout'
import TabSwitcher from '@/components/TabSwitcher'
import Button from '@/components/Button'
import FileDropzone from '@/components/FileDropzone'
import VerdictCard from '@/components/VerdictCard'
import ErrorBanner from '@/components/ErrorBanner'
import { useAsyncAction } from '@/hooks/useAsyncAction'
import { scamShieldText, scamShieldScreenshot } from '@/lib/api'

const SAMPLE_SCAM_TEXT =
  'URGENT: Your account will be suspended in 2 hours unless you verify your details now. Tap here to avoid permanent closure: bit.ly/verify-acc-2847'

const MODES = ['Text', 'Screenshot'] as const

export default function ScamShieldPage() {
  const [mode, setMode] = useState<(typeof MODES)[number]>('Text')
  const [text, setText] = useState('')
  const [textState, runText] = useAsyncAction(scamShieldText)
  const [shotState, runShot] = useAsyncAction(scamShieldScreenshot)

  return (
    <Layout icon="◈" title="Scam Shield" subtitle="Paste a message or drop a screenshot — get a scam/legit verdict, on-device.">
      <TabSwitcher options={MODES} value={mode} onChange={setMode} />

      {mode === 'Text' && (
        <div className="flex flex-col gap-3">
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Paste a message here..."
            rows={6}
            className="rounded-2xl border border-border bg-card p-4 text-sm outline-none transition-colors focus:border-[var(--nxt-violet)]"
          />
          <div className="flex flex-wrap gap-3">
            <Button onClick={() => runText(text)} disabled={!text.trim() || textState.status === 'loading'}>
              {textState.status === 'loading' ? 'Checking…' : 'Check for scam'}
            </Button>
            <Button variant="secondary" onClick={() => setText(SAMPLE_SCAM_TEXT)}>
              Load sample scam message
            </Button>
          </div>
          {textState.status === 'error' && <ErrorBanner message={textState.message} />}
          {textState.status === 'success' && <VerdictCard verdict={textState.data} />}
        </div>
      )}

      {mode === 'Screenshot' && (
        <div className="flex flex-col gap-4">
          <FileDropzone accept="image/*" label="Upload a screenshot of a message" onSelect={(file) => runShot(file)} />
          {shotState.status === 'loading' && <p className="text-sm text-muted-foreground">Reading text from the image and checking for scam patterns…</p>}
          {shotState.status === 'error' && <ErrorBanner message={shotState.message} />}
          {shotState.status === 'success' && (
            <>
              <details className="rounded-xl border border-border bg-card p-4 text-sm">
                <summary className="cursor-pointer font-medium">Extracted text</summary>
                <p className="mt-2 whitespace-pre-wrap text-muted-foreground">{shotState.data.extracted_text}</p>
              </details>
              <VerdictCard verdict={shotState.data} />
            </>
          )}
        </div>
      )}
    </Layout>
  )
}

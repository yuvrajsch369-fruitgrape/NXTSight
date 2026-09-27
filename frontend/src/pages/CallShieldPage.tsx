import { useState } from 'react'
import Layout from '@/components/Layout'
import TabSwitcher from '@/components/TabSwitcher'
import Button from '@/components/Button'
import FileDropzone from '@/components/FileDropzone'
import VerdictCard from '@/components/VerdictCard'
import ErrorBanner from '@/components/ErrorBanner'
import { useAsyncAction } from '@/hooks/useAsyncAction'
import { callShieldTranscript, callShieldRecording } from '@/lib/api'

// Same sample as app.py's CALL_TRANSCRIPT_SAMPLES["Digital-arrest scam"].
const SAMPLE_TRANSCRIPT = `Caller: Namaste, I am Sub-Inspector Verma calling from the Cyber Crime Investigation Unit.
You: Yes, what is this regarding?
Caller: Your PAN card has been used to open a bank account involved in a 2 crore rupee money laundering racket linked to human trafficking. There is a digital arrest warrant against your name.
You: This can't be right, I've never opened any such account.
Caller: The evidence is very clear on our end. You must not disconnect this call or leave your house. Keep your camera on for the remainder of this investigation.
You: Okay, I'm scared, what do I do?
Caller: To prove you are not involved, transfer all funds from your savings account to the RBI secure holding account we provide for verification within the next hour, and read out the OTP the moment it arrives. Failing to comply means a police team arrives at your address immediately.`

const MODES = ['Transcript', 'Recording'] as const

export default function CallShieldPage() {
  const [mode, setMode] = useState<(typeof MODES)[number]>('Transcript')
  const [text, setText] = useState('')
  const [textState, runText] = useAsyncAction(callShieldTranscript)
  const [recState, runRec] = useAsyncAction(callShieldRecording)

  return (
    <Layout icon="☎" title="Call Shield" subtitle="A call transcript or recording, screened for scam patterns.">
      <TabSwitcher options={MODES} value={mode} onChange={setMode} />

      {mode === 'Transcript' && (
        <div className="flex flex-col gap-3">
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Paste a call transcript here..."
            rows={8}
            className="rounded-2xl border border-border bg-card p-4 text-sm outline-none transition-colors focus:border-[var(--nxt-violet)]"
          />
          <div className="flex flex-wrap gap-3">
            <Button onClick={() => runText(text)} disabled={!text.trim() || textState.status === 'loading'}>
              {textState.status === 'loading' ? 'Analyzing…' : 'Analyze call'}
            </Button>
            <Button variant="secondary" onClick={() => setText(SAMPLE_TRANSCRIPT)}>
              Load sample: digital-arrest scam
            </Button>
          </div>
          {textState.status === 'error' && <ErrorBanner message={textState.message} />}
          {textState.status === 'success' && <VerdictCard verdict={textState.data} />}
        </div>
      )}

      {mode === 'Recording' && (
        <div className="flex flex-col gap-4">
          <FileDropzone accept="audio/wav,audio/x-wav,.wav" label="Upload a WAV call recording" onSelect={(file) => runRec(file)} />
          {recState.status === 'loading' && <p className="text-sm text-muted-foreground">Transcribing and analyzing the recording…</p>}
          {recState.status === 'error' && <ErrorBanner message={recState.message} />}
          {recState.status === 'success' && (
            <>
              <details className="rounded-xl border border-border bg-card p-4 text-sm">
                <summary className="cursor-pointer font-medium">Transcript</summary>
                <p className="mt-2 whitespace-pre-wrap text-muted-foreground">{recState.data.transcript}</p>
              </details>
              <VerdictCard verdict={recState.data} />
            </>
          )}
        </div>
      )}
    </Layout>
  )
}

import { type ChangeEvent, type DragEvent, useId, useState } from 'react'

// Matches assets/theme.css's [data-testid="stFileUploaderDropzone"] look
// (dashed border, panel background, violet highlight on drag) -- same
// visual language, now a real HTML file input under the hood.
export default function FileDropzone({
  accept,
  label,
  onSelect,
}: {
  accept: string
  label: string
  onSelect: (file: File) => void
}) {
  const [dragging, setDragging] = useState(false)
  const [fileName, setFileName] = useState<string | null>(null)
  const inputId = useId()

  function handleFile(file: File | undefined) {
    if (!file) return
    setFileName(file.name)
    onSelect(file)
  }

  function handleDrop(e: DragEvent<HTMLLabelElement>) {
    e.preventDefault()
    setDragging(false)
    handleFile(e.dataTransfer.files[0])
  }

  function handleChange(e: ChangeEvent<HTMLInputElement>) {
    handleFile(e.target.files?.[0])
  }

  return (
    <label
      htmlFor={inputId}
      onDragOver={(e) => {
        e.preventDefault()
        setDragging(true)
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      className="flex cursor-pointer flex-col items-center gap-2 rounded-2xl border border-dashed px-6 py-10 text-center transition-colors"
      style={{
        borderColor: dragging ? 'var(--nxt-violet)' : 'var(--nxt-border-strong)',
        background: dragging ? 'var(--nxt-panel-hover)' : 'var(--nxt-panel)',
      }}
    >
      <input id={inputId} type="file" accept={accept} onChange={handleChange} className="hidden" />
      <span className="text-2xl">▤</span>
      <span className="text-sm font-medium">{fileName ?? label}</span>
      {!fileName && <span className="text-xs text-muted-foreground">Click to browse or drag a file here</span>}
    </label>
  )
}

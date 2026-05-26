import { useState, useEffect, useRef } from 'react'
import RecordForm, { emptyRecord } from './RecordForm'
import { loadDraft, saveDraft, clearDraft } from '../storage'

export default function NewRecord({ onSave }) {
  const [value, setValue] = useState(() => loadDraft() || emptyRecord())
  const [savedAt, setSavedAt] = useState(null)
  const firstRender = useRef(true)

  useEffect(() => {
    if (firstRender.current) {
      firstRender.current = false
      return
    }
    saveDraft(value)
    setSavedAt(new Date())
  }, [value])

  function change(field, val) {
    setValue((prev) => ({ ...prev, [field]: val }))
  }

  function submit() {
    const now = new Date().toISOString()
    const record = {
      ...value,
      id:
        typeof crypto !== 'undefined' && crypto.randomUUID
          ? crypto.randomUUID()
          : String(Date.now()),
      createdAt: now,
      updatedAt: now,
    }
    clearDraft()
    setValue(emptyRecord())
    onSave(record)
  }

  function reset() {
    clearDraft()
    setValue(emptyRecord())
    setSavedAt(null)
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h2 className="font-serif text-xl font-bold text-navy">
          New Test Record
        </h2>
        <span className="text-xs text-navy/50">
          {savedAt
            ? `Draft auto-saved at ${savedAt.toLocaleTimeString()}`
            : 'Draft auto-saves as you type'}
        </span>
      </div>

      <div className="bg-white rounded-lg border border-cream-dark shadow-sm p-6">
        <RecordForm value={value} onChange={change} />

        <div className="flex flex-wrap gap-3 mt-8 pt-6 border-t border-cream-dark">
          <button
            onClick={submit}
            className="px-5 py-2.5 rounded-md bg-navy text-cream font-semibold hover:bg-navy-dark transition-colors"
          >
            Save record
          </button>
          <button
            onClick={reset}
            className="px-5 py-2.5 rounded-md border border-cream-dark text-navy font-semibold hover:bg-cream-dark transition-colors"
          >
            Clear form
          </button>
        </div>
      </div>
    </div>
  )
}

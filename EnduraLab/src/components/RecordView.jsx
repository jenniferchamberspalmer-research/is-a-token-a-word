import { useState } from 'react'
import RecordForm from './RecordForm'
import { RATING_COLORS } from '../constants'

function RatingBadge({ rating }) {
  if (!rating) return <span className="text-navy/40">Not rated</span>
  return (
    <span
      className="inline-block px-3 py-1 rounded-full text-sm font-semibold text-white"
      style={{ backgroundColor: RATING_COLORS[rating] || '#64748b' }}
    >
      {rating}
    </span>
  )
}

function TextBlock({ label, text }) {
  return (
    <div>
      <h4 className="text-sm font-semibold text-navy mb-1">{label}</h4>
      {text && text.trim() ? (
        <div className="bg-cream/50 border border-cream-dark rounded-md p-4 whitespace-pre-wrap text-navy leading-relaxed">
          {text}
        </div>
      ) : (
        <p className="text-navy/40 italic text-sm">Empty</p>
      )}
    </div>
  )
}

export default function RecordView({ record, onUpdate, onDelete, onBack }) {
  const [editing, setEditing] = useState(false)
  const [value, setValue] = useState(record)
  const [confirmDelete, setConfirmDelete] = useState(false)

  function change(field, val) {
    setValue((prev) => ({ ...prev, [field]: val }))
  }

  function save() {
    onUpdate({ ...value, updatedAt: new Date().toISOString() })
    setEditing(false)
  }

  function cancel() {
    setValue(record)
    setEditing(false)
  }

  return (
    <div>
      <div className="flex flex-wrap items-center justify-between gap-3 mb-6">
        <button
          onClick={onBack}
          className="text-sm font-semibold text-navy/70 hover:text-navy"
        >
          &larr; Back to all records
        </button>
        {!editing && (
          <div className="flex gap-2">
            <button
              onClick={() => setEditing(true)}
              className="px-4 py-2 rounded-md bg-navy text-cream text-sm font-semibold hover:bg-navy-dark transition-colors"
            >
              Edit
            </button>
            {confirmDelete ? (
              <>
                <button
                  onClick={() => onDelete(record.id)}
                  className="px-4 py-2 rounded-md bg-red-700 text-white text-sm font-semibold hover:bg-red-800 transition-colors"
                >
                  Confirm delete
                </button>
                <button
                  onClick={() => setConfirmDelete(false)}
                  className="px-4 py-2 rounded-md border border-cream-dark text-navy text-sm font-semibold"
                >
                  Cancel
                </button>
              </>
            ) : (
              <button
                onClick={() => setConfirmDelete(true)}
                className="px-4 py-2 rounded-md border border-red-700 text-red-700 text-sm font-semibold hover:bg-red-700 hover:text-white transition-colors"
              >
                Delete
              </button>
            )}
          </div>
        )}
      </div>

      <div className="bg-white rounded-lg border border-cream-dark shadow-sm p-6">
        {editing ? (
          <>
            <RecordForm value={value} onChange={change} />
            <div className="flex gap-3 mt-8 pt-6 border-t border-cream-dark">
              <button
                onClick={save}
                className="px-5 py-2.5 rounded-md bg-navy text-cream font-semibold hover:bg-navy-dark transition-colors"
              >
                Save changes
              </button>
              <button
                onClick={cancel}
                className="px-5 py-2.5 rounded-md border border-cream-dark text-navy font-semibold hover:bg-cream-dark transition-colors"
              >
                Cancel
              </button>
            </div>
          </>
        ) : (
          <div className="space-y-6">
            <div className="flex flex-wrap items-start justify-between gap-4 pb-4 border-b border-cream-dark">
              <div>
                <h2 className="font-serif text-2xl font-bold text-navy">
                  {record.discipline} &middot; {record.assignmentType}
                </h2>
                <p className="text-navy/50 text-sm mt-1">
                  Tested {record.testDate}
                  {record.updatedAt && (
                    <>
                      {' '}
                      &middot; updated{' '}
                      {new Date(record.updatedAt).toLocaleDateString()}
                    </>
                  )}
                </p>
              </div>
              <div className="text-right">
                <div className="text-xs text-navy/50 uppercase tracking-wide">
                  Vulnerability Score
                </div>
                <div className="font-serif text-3xl font-bold text-navy">
                  {record.vulnerabilityScore === '' ||
                  record.vulnerabilityScore == null
                    ? '—'
                    : record.vulnerabilityScore}
                </div>
              </div>
            </div>

            <TextBlock
              label="Original assignment"
              text={record.originalAssignment}
            />

            <div className="flex items-center gap-3">
              <span className="text-sm font-semibold text-navy">
                Outcomes quality:
              </span>
              <RatingBadge rating={record.outcomesRating} />
            </div>
            <TextBlock
              label="Learning outcomes Endura generated"
              text={record.learningOutcomes}
            />

            <div className="flex items-center gap-3">
              <span className="text-sm font-semibold text-navy">
                Redesign quality:
              </span>
              <RatingBadge rating={record.redesignRating} />
            </div>
            <TextBlock
              label="Redesigned assignment Endura produced"
              text={record.redesignedAssignment}
            />

            <TextBlock label="My notes and observations" text={record.notes} />
            <TextBlock
              label="Action items for prompt improvements"
              text={record.actionItems}
            />
          </div>
        )}
      </div>
    </div>
  )
}

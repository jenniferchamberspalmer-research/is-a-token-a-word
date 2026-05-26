import { DISCIPLINES, ASSIGNMENT_TYPES, QUALITY_RATINGS } from '../constants'

export function today() {
  return new Date().toISOString().slice(0, 10)
}

export function emptyRecord() {
  return {
    testDate: today(),
    discipline: 'Communication',
    assignmentType: 'Essay',
    originalAssignment: '',
    vulnerabilityScore: '',
    learningOutcomes: '',
    outcomesRating: 'Good',
    redesignedAssignment: '',
    redesignRating: 'Good',
    notes: '',
    actionItems: '',
  }
}

const labelClass = 'block text-sm font-semibold text-navy mb-1'
const inputClass =
  'w-full rounded-md border border-cream-dark bg-cream/40 px-3 py-2 text-navy focus:outline-none focus:ring-2 focus:ring-navy-light focus:border-navy-light'
const textareaClass = inputClass + ' min-h-[140px] font-sans leading-relaxed'

function Field({ label, children }) {
  return (
    <div>
      <label className={labelClass}>{label}</label>
      {children}
    </div>
  )
}

export default function RecordForm({ value, onChange }) {
  const set = (field) => (e) => onChange(field, e.target.value)

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <Field label="Test date">
          <input
            type="date"
            className={inputClass}
            value={value.testDate}
            onChange={set('testDate')}
          />
        </Field>

        <Field label="Discipline">
          <select
            className={inputClass}
            value={value.discipline}
            onChange={set('discipline')}
          >
            {DISCIPLINES.map((d) => (
              <option key={d}>{d}</option>
            ))}
          </select>
        </Field>

        <Field label="Assignment type">
          <select
            className={inputClass}
            value={value.assignmentType}
            onChange={set('assignmentType')}
          >
            {ASSIGNMENT_TYPES.map((t) => (
              <option key={t}>{t}</option>
            ))}
          </select>
        </Field>

        <Field label="AI Vulnerability Score (0-100)">
          <input
            type="number"
            min="0"
            max="100"
            className={inputClass}
            value={value.vulnerabilityScore}
            onChange={set('vulnerabilityScore')}
            placeholder="e.g. 72"
          />
        </Field>
      </div>

      <Field label="Original assignment">
        <textarea
          className={textareaClass}
          value={value.originalAssignment}
          onChange={set('originalAssignment')}
          placeholder="Paste the original assignment here..."
        />
      </Field>

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_auto] gap-4 items-start">
        <Field label="Learning outcomes Endura generated">
          <textarea
            className={textareaClass}
            value={value.learningOutcomes}
            onChange={set('learningOutcomes')}
            placeholder="Paste the generated learning outcomes here..."
          />
        </Field>
        <Field label="Outcomes quality rating">
          <select
            className={inputClass}
            value={value.outcomesRating}
            onChange={set('outcomesRating')}
          >
            {QUALITY_RATINGS.map((r) => (
              <option key={r}>{r}</option>
            ))}
          </select>
        </Field>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_auto] gap-4 items-start">
        <Field label="Redesigned assignment Endura produced">
          <textarea
            className={textareaClass}
            value={value.redesignedAssignment}
            onChange={set('redesignedAssignment')}
            placeholder="Paste the redesigned assignment here..."
          />
        </Field>
        <Field label="Redesign quality rating">
          <select
            className={inputClass}
            value={value.redesignRating}
            onChange={set('redesignRating')}
          >
            {QUALITY_RATINGS.map((r) => (
              <option key={r}>{r}</option>
            ))}
          </select>
        </Field>
      </div>

      <Field label="My notes and observations">
        <textarea
          className={textareaClass}
          value={value.notes}
          onChange={set('notes')}
          placeholder="What stood out? What worked, what didn't?"
        />
      </Field>

      <Field label="Action items for prompt improvements">
        <textarea
          className={textareaClass}
          value={value.actionItems}
          onChange={set('actionItems')}
          placeholder="Concrete changes to try in the next prompt iteration..."
        />
      </Field>
    </div>
  )
}

#requires -Version 5.0
<#
    Endura Testing Lab - setup script
    Creates a complete Vite + React + Tailwind project.
    Usage:  powershell -ExecutionPolicy Bypass -File .\setup-enduralab.ps1
#>
param(
    [string]$Root = 'C:\Users\amptk\OneDrive\Desktop\EnduraLab'
)

$ErrorActionPreference = 'Stop'

Write-Host 'Creating Endura Testing Lab in:' $Root -ForegroundColor Cyan

# --- create directories ---
$dirs = @(
    $Root,
    (Join-Path $Root 'src'),
    (Join-Path $Root 'src\components')
)
foreach ($d in $dirs) {
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d -Force | Out-Null }
}

function Write-ProjectFile {
    param([string]$RelativePath, [string]$Content)
    $full = Join-Path $Root $RelativePath
    $dir = Split-Path $full -Parent
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    # Write UTF-8 without BOM
    $enc = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($full, $Content, $enc)
    Write-Host '  wrote' $RelativePath -ForegroundColor DarkGray
}

# --- package.json ---
$content = @'
{
  "name": "endura-testing-lab",
  "private": true,
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1"
  },
  "devDependencies": {
    "@vitejs/plugin-react": "^4.3.4",
    "autoprefixer": "^10.4.20",
    "postcss": "^8.4.49",
    "tailwindcss": "^3.4.17",
    "vite": "^5.4.11"
  }
}
'@
Write-ProjectFile -RelativePath 'package.json' -Content $content

# --- vite.config.js ---
$content = @'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
})
'@
Write-ProjectFile -RelativePath 'vite.config.js' -Content $content

# --- tailwind.config.js ---
$content = @'
/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        navy: {
          DEFAULT: '#1b2a4a',
          light: '#2c3e63',
          dark: '#121d36',
        },
        cream: {
          DEFAULT: '#f7f3e9',
          dark: '#ece5d3',
        },
      },
      fontFamily: {
        serif: ['Georgia', 'Cambria', 'Times New Roman', 'serif'],
      },
    },
  },
  plugins: [],
}
'@
Write-ProjectFile -RelativePath 'tailwind.config.js' -Content $content

# --- postcss.config.js ---
$content = @'
export default {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}
'@
Write-ProjectFile -RelativePath 'postcss.config.js' -Content $content

# --- index.html ---
$content = @'
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Endura Testing Lab</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
  </body>
</html>
'@
Write-ProjectFile -RelativePath 'index.html' -Content $content

# --- .gitignore ---
$content = @'
node_modules
dist
.DS_Store
*.local
'@
Write-ProjectFile -RelativePath '.gitignore' -Content $content

# --- src/main.jsx ---
$content = @'
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
'@
Write-ProjectFile -RelativePath 'src\main.jsx' -Content $content

# --- src/index.css ---
$content = @'
@tailwind base;
@tailwind components;
@tailwind utilities;

@layer base {
  body {
    @apply bg-cream text-navy antialiased;
  }
}
'@
Write-ProjectFile -RelativePath 'src\index.css' -Content $content

# --- src/constants.js ---
$content = @'
export const DISCIPLINES = [
  'Communication',
  'Nursing',
  'History',
  'Business',
  'STEM',
  'Education',
  'Other',
]

export const ASSIGNMENT_TYPES = [
  'Essay',
  'Discussion Post',
  'Lab Report',
  'Case Study',
  'Reflection',
  'Syllabus',
  'Other',
]

export const QUALITY_RATINGS = ['Excellent', 'Good', 'Needs Work', 'Poor']

export const RATING_COLORS = {
  Excellent: '#2f7a4d',
  Good: '#3f86c4',
  'Needs Work': '#d99a2b',
  Poor: '#c0492f',
}
'@
Write-ProjectFile -RelativePath 'src\constants.js' -Content $content

# --- src/storage.js ---
$content = @'
const RECORDS_KEY = 'endura-lab-records'
const DRAFT_KEY = 'endura-lab-draft'

export function loadRecords() {
  try {
    const raw = localStorage.getItem(RECORDS_KEY)
    return raw ? JSON.parse(raw) : []
  } catch {
    return []
  }
}

export function saveRecords(records) {
  localStorage.setItem(RECORDS_KEY, JSON.stringify(records))
}

export function loadDraft() {
  try {
    const raw = localStorage.getItem(DRAFT_KEY)
    return raw ? JSON.parse(raw) : null
  } catch {
    return null
  }
}

export function saveDraft(draft) {
  localStorage.setItem(DRAFT_KEY, JSON.stringify(draft))
}

export function clearDraft() {
  localStorage.removeItem(DRAFT_KEY)
}
'@
Write-ProjectFile -RelativePath 'src\storage.js' -Content $content

# --- src/App.jsx ---
$content = @'
import { useState, useEffect } from 'react'
import { loadRecords, saveRecords } from './storage'
import NewRecord from './components/NewRecord'
import AllRecords from './components/AllRecords'
import FindingsSummary from './components/FindingsSummary'
import RecordView from './components/RecordView'

const NAV = [
  { id: 'new', label: 'New Test Record' },
  { id: 'all', label: 'All Records' },
  { id: 'summary', label: 'Findings Summary' },
]

export default function App() {
  const [records, setRecords] = useState(() => loadRecords())
  const [view, setView] = useState('all')
  const [selectedId, setSelectedId] = useState(null)

  useEffect(() => {
    saveRecords(records)
  }, [records])

  function addRecord(record) {
    setRecords((prev) => [record, ...prev])
  }

  function updateRecord(updated) {
    setRecords((prev) => prev.map((r) => (r.id === updated.id ? updated : r)))
  }

  function deleteRecord(id) {
    setRecords((prev) => prev.filter((r) => r.id !== id))
  }

  function openRecord(id) {
    setSelectedId(id)
    setView('record')
  }

  const selected = records.find((r) => r.id === selectedId)

  return (
    <div className="min-h-screen flex flex-col">
      <header className="bg-navy text-cream shadow-md">
        <div className="max-w-6xl mx-auto px-6 py-5 flex flex-col sm:flex-row sm:items-end sm:justify-between gap-4">
          <div>
            <h1 className="font-serif text-2xl font-bold tracking-tight">
              Endura Testing Lab
            </h1>
            <p className="text-cream/70 text-sm mt-1">
              Research &amp; testing journal
            </p>
          </div>
          <nav className="flex flex-wrap gap-2">
            {NAV.map((item) => {
              const active = view === item.id
              return (
                <button
                  key={item.id}
                  onClick={() => setView(item.id)}
                  className={
                    'px-4 py-2 rounded-md text-sm font-semibold transition-colors ' +
                    (active
                      ? 'bg-cream text-navy'
                      : 'bg-navy-light text-cream hover:bg-cream/20')
                  }
                >
                  {item.label}
                </button>
              )
            })}
          </nav>
        </div>
      </header>

      <main className="flex-1 w-full max-w-6xl mx-auto px-6 py-8">
        {view === 'new' && (
          <NewRecord
            onSave={(record) => {
              addRecord(record)
              openRecord(record.id)
            }}
          />
        )}

        {view === 'all' && (
          <AllRecords
            records={records}
            onOpen={openRecord}
            onNew={() => setView('new')}
          />
        )}

        {view === 'summary' && <FindingsSummary records={records} />}

        {view === 'record' &&
          (selected ? (
            <RecordView
              key={selected.id}
              record={selected}
              onUpdate={updateRecord}
              onDelete={(id) => {
                deleteRecord(id)
                setView('all')
              }}
              onBack={() => setView('all')}
            />
          ) : (
            <div className="bg-white rounded-lg border border-cream-dark p-8 text-center">
              <p className="text-navy/70">That record no longer exists.</p>
              <button
                onClick={() => setView('all')}
                className="mt-4 px-4 py-2 rounded-md bg-navy text-cream text-sm font-semibold"
              >
                Back to all records
              </button>
            </div>
          ))}
      </main>

      <footer className="text-center text-xs text-navy/40 py-6">
        Endura Testing Lab &middot; data stored locally in this browser
      </footer>
    </div>
  )
}
'@
Write-ProjectFile -RelativePath 'src\App.jsx' -Content $content

# --- src/components/RecordForm.jsx ---
$content = @'
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
'@
Write-ProjectFile -RelativePath 'src\components\RecordForm.jsx' -Content $content

# --- src/components/NewRecord.jsx ---
$content = @'
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
'@
Write-ProjectFile -RelativePath 'src\components\NewRecord.jsx' -Content $content

# --- src/components/AllRecords.jsx ---
$content = @'
import { useState, useMemo } from 'react'
import { DISCIPLINES, QUALITY_RATINGS, RATING_COLORS } from '../constants'

const COLUMNS = [
  { key: 'testDate', label: 'Date' },
  { key: 'discipline', label: 'Discipline' },
  { key: 'assignmentType', label: 'Assignment Type' },
  { key: 'vulnerabilityScore', label: 'Vuln. Score' },
  { key: 'outcomesRating', label: 'Outcomes' },
  { key: 'redesignRating', label: 'Redesign' },
]

function RatingBadge({ rating }) {
  if (!rating) return <span className="text-navy/30">&mdash;</span>
  return (
    <span
      className="inline-block px-2 py-0.5 rounded-full text-xs font-semibold text-white"
      style={{ backgroundColor: RATING_COLORS[rating] || '#64748b' }}
    >
      {rating}
    </span>
  )
}

export default function AllRecords({ records, onOpen, onNew }) {
  const [sortKey, setSortKey] = useState('testDate')
  const [sortDir, setSortDir] = useState('desc')
  const [discFilter, setDiscFilter] = useState('All')
  const [ratingFilter, setRatingFilter] = useState('All')

  const rows = useMemo(() => {
    const filtered = records.filter((r) => {
      const discOk = discFilter === 'All' || r.discipline === discFilter
      const ratingOk =
        ratingFilter === 'All' ||
        r.outcomesRating === ratingFilter ||
        r.redesignRating === ratingFilter
      return discOk && ratingOk
    })

    return [...filtered].sort((a, b) => {
      let av = a[sortKey] ?? ''
      let bv = b[sortKey] ?? ''
      if (sortKey === 'vulnerabilityScore') {
        av = av === '' ? -Infinity : Number(av)
        bv = bv === '' ? -Infinity : Number(bv)
      }
      if (av < bv) return sortDir === 'asc' ? -1 : 1
      if (av > bv) return sortDir === 'asc' ? 1 : -1
      return 0
    })
  }, [records, sortKey, sortDir, discFilter, ratingFilter])

  function toggleSort(key) {
    if (sortKey === key) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'))
    } else {
      setSortKey(key)
      setSortDir('asc')
    }
  }

  const selectClass =
    'rounded-md border border-cream-dark bg-white px-3 py-2 text-sm text-navy focus:outline-none focus:ring-2 focus:ring-navy-light'

  return (
    <div>
      <div className="flex flex-wrap items-center justify-between gap-4 mb-6">
        <h2 className="font-serif text-xl font-bold text-navy">All Records</h2>
        <button
          onClick={onNew}
          className="px-4 py-2 rounded-md bg-navy text-cream text-sm font-semibold hover:bg-navy-dark transition-colors"
        >
          + New Record
        </button>
      </div>

      <div className="flex flex-wrap items-center gap-4 mb-4">
        <label className="text-sm text-navy/70 flex items-center gap-2">
          Discipline
          <select
            className={selectClass}
            value={discFilter}
            onChange={(e) => setDiscFilter(e.target.value)}
          >
            <option>All</option>
            {DISCIPLINES.map((d) => (
              <option key={d}>{d}</option>
            ))}
          </select>
        </label>
        <label className="text-sm text-navy/70 flex items-center gap-2">
          Rating
          <select
            className={selectClass}
            value={ratingFilter}
            onChange={(e) => setRatingFilter(e.target.value)}
          >
            <option>All</option>
            {QUALITY_RATINGS.map((r) => (
              <option key={r}>{r}</option>
            ))}
          </select>
        </label>
        <span className="text-sm text-navy/50 ml-auto">
          {rows.length} of {records.length} records
        </span>
      </div>

      <div className="bg-white rounded-lg border border-cream-dark shadow-sm overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-navy text-cream text-left">
                {COLUMNS.map((col) => (
                  <th
                    key={col.key}
                    onClick={() => toggleSort(col.key)}
                    className="px-4 py-3 font-semibold cursor-pointer select-none whitespace-nowrap hover:bg-navy-light"
                  >
                    {col.label}
                    {sortKey === col.key && (
                      <span className="ml-1">
                        {sortDir === 'asc' ? '▲' : '▼'}
                      </span>
                    )}
                  </th>
                ))}
                <th className="px-4 py-3" />
              </tr>
            </thead>
            <tbody>
              {rows.length === 0 && (
                <tr>
                  <td
                    colSpan={COLUMNS.length + 1}
                    className="px-4 py-12 text-center text-navy/50"
                  >
                    No records yet. Click "+ New Record" to add your first test.
                  </td>
                </tr>
              )}
              {rows.map((r, i) => (
                <tr
                  key={r.id}
                  className={
                    'border-t border-cream-dark ' +
                    (i % 2 ? 'bg-cream/30' : 'bg-white')
                  }
                >
                  <td className="px-4 py-3 whitespace-nowrap">{r.testDate}</td>
                  <td className="px-4 py-3 whitespace-nowrap">{r.discipline}</td>
                  <td className="px-4 py-3 whitespace-nowrap">
                    {r.assignmentType}
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap">
                    {r.vulnerabilityScore === '' ||
                    r.vulnerabilityScore == null ? (
                      <span className="text-navy/30">&mdash;</span>
                    ) : (
                      r.vulnerabilityScore
                    )}
                  </td>
                  <td className="px-4 py-3">
                    <RatingBadge rating={r.outcomesRating} />
                  </td>
                  <td className="px-4 py-3">
                    <RatingBadge rating={r.redesignRating} />
                  </td>
                  <td className="px-4 py-3 text-right whitespace-nowrap">
                    <button
                      onClick={() => onOpen(r.id)}
                      className="px-3 py-1.5 rounded-md border border-navy text-navy text-xs font-semibold hover:bg-navy hover:text-cream transition-colors"
                    >
                      View
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
'@
Write-ProjectFile -RelativePath 'src\components\AllRecords.jsx' -Content $content

# --- src/components/FindingsSummary.jsx ---
$content = @'
import { useMemo } from 'react'
import { DISCIPLINES, QUALITY_RATINGS, RATING_COLORS } from '../constants'

function Card({ title, children }) {
  return (
    <div className="bg-white rounded-lg border border-cream-dark shadow-sm p-6">
      <h3 className="font-serif text-lg font-bold text-navy mb-4">{title}</h3>
      {children}
    </div>
  )
}

function RatingChart({ counts, total }) {
  if (total === 0) {
    return <p className="text-navy/50 text-sm">No ratings recorded yet.</p>
  }
  return (
    <div className="space-y-3">
      {QUALITY_RATINGS.map((rating) => {
        const count = counts[rating] || 0
        const pct = total ? Math.round((count / total) * 100) : 0
        return (
          <div key={rating}>
            <div className="flex justify-between text-sm mb-1">
              <span className="font-medium text-navy">{rating}</span>
              <span className="text-navy/60">
                {count} ({pct}%)
              </span>
            </div>
            <div className="h-3 rounded-full bg-cream-dark overflow-hidden">
              <div
                className="h-full rounded-full transition-all"
                style={{
                  width: `${pct}%`,
                  backgroundColor: RATING_COLORS[rating],
                }}
              />
            </div>
          </div>
        )
      })}
    </div>
  )
}

export default function FindingsSummary({ records }) {
  const stats = useMemo(() => {
    const total = records.length

    const byDiscipline = {}
    records.forEach((r) => {
      const score = Number(r.vulnerabilityScore)
      if (r.vulnerabilityScore !== '' && !Number.isNaN(score)) {
        if (!byDiscipline[r.discipline]) {
          byDiscipline[r.discipline] = { sum: 0, count: 0 }
        }
        byDiscipline[r.discipline].sum += score
        byDiscipline[r.discipline].count += 1
      }
    })

    const avgByDiscipline = DISCIPLINES.map((d) => {
      const entry = byDiscipline[d]
      return {
        discipline: d,
        avg: entry ? Math.round(entry.sum / entry.count) : null,
        count: entry ? entry.count : 0,
      }
    }).filter((d) => d.count > 0)

    const outcomesCounts = {}
    const redesignCounts = {}
    let outcomesTotal = 0
    let redesignTotal = 0
    records.forEach((r) => {
      if (r.outcomesRating) {
        outcomesCounts[r.outcomesRating] =
          (outcomesCounts[r.outcomesRating] || 0) + 1
        outcomesTotal += 1
      }
      if (r.redesignRating) {
        redesignCounts[r.redesignRating] =
          (redesignCounts[r.redesignRating] || 0) + 1
        redesignTotal += 1
      }
    })

    const actionItems = records
      .filter((r) => r.actionItems && r.actionItems.trim())
      .map((r) => ({
        id: r.id,
        date: r.testDate,
        discipline: r.discipline,
        assignmentType: r.assignmentType,
        text: r.actionItems.trim(),
      }))

    return {
      total,
      avgByDiscipline,
      outcomesCounts,
      redesignCounts,
      outcomesTotal,
      redesignTotal,
      actionItems,
    }
  }, [records])

  const maxAvg = Math.max(100, ...stats.avgByDiscipline.map((d) => d.avg || 0))

  return (
    <div>
      <h2 className="font-serif text-xl font-bold text-navy mb-6">
        Findings Summary
      </h2>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
        <Card title="Total Tests Run">
          <p className="font-serif text-5xl font-bold text-navy">
            {stats.total}
          </p>
          <p className="text-navy/50 text-sm mt-2">
            records documented in this lab
          </p>
        </Card>

        <Card title="Outcomes Quality">
          <RatingChart
            counts={stats.outcomesCounts}
            total={stats.outcomesTotal}
          />
        </Card>

        <Card title="Redesign Quality">
          <RatingChart
            counts={stats.redesignCounts}
            total={stats.redesignTotal}
          />
        </Card>
      </div>

      <div className="mb-6">
        <Card title="Average Vulnerability Score by Discipline">
          {stats.avgByDiscipline.length === 0 ? (
            <p className="text-navy/50 text-sm">
              No vulnerability scores recorded yet.
            </p>
          ) : (
            <div className="space-y-3">
              {stats.avgByDiscipline.map((d) => {
                const pct = Math.round((d.avg / maxAvg) * 100)
                return (
                  <div key={d.discipline}>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="font-medium text-navy">
                        {d.discipline}{' '}
                        <span className="text-navy/40">
                          ({d.count} {d.count === 1 ? 'test' : 'tests'})
                        </span>
                      </span>
                      <span className="text-navy/60 font-semibold">
                        {d.avg}
                      </span>
                    </div>
                    <div className="h-4 rounded-full bg-cream-dark overflow-hidden">
                      <div
                        className="h-full rounded-full bg-navy transition-all"
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </Card>
      </div>

      <Card title={`Action Items Across All Records (${stats.actionItems.length})`}>
        {stats.actionItems.length === 0 ? (
          <p className="text-navy/50 text-sm">
            No action items recorded yet. Add them as you test to spot patterns.
          </p>
        ) : (
          <ul className="space-y-4">
            {stats.actionItems.map((item) => (
              <li
                key={item.id}
                className="border-l-4 border-navy-light pl-4 py-1"
              >
                <div className="text-xs text-navy/50 mb-1">
                  {item.date} &middot; {item.discipline} &middot;{' '}
                  {item.assignmentType}
                </div>
                <div className="text-navy whitespace-pre-wrap">{item.text}</div>
              </li>
            ))}
          </ul>
        )}
      </Card>
    </div>
  )
}
'@
Write-ProjectFile -RelativePath 'src\components\FindingsSummary.jsx' -Content $content

# --- src/components/RecordView.jsx ---
$content = @'
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
'@
Write-ProjectFile -RelativePath 'src\components\RecordView.jsx' -Content $content

Write-Host ''
Write-Host 'All files created.' -ForegroundColor Green
Write-Host ''
Write-Host 'Next steps:' -ForegroundColor Cyan
Write-Host ('  cd ' + $Root)
Write-Host '  npm install'
Write-Host '  npm run dev'
Write-Host ''
Write-Host 'Then open the http://localhost:5173 URL that Vite prints.' -ForegroundColor Cyan

$answer = Read-Host 'Run npm install now? (y/n)'
if ($answer -eq 'y') {
    Push-Location $Root
    npm install
    Pop-Location
    Write-Host 'Done. Run "npm run dev" inside the project folder to start.' -ForegroundColor Green
}

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

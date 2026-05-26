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

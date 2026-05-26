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

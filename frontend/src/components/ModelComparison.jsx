import React, { useState, useEffect } from 'react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts'
import { Trophy, ArrowUpDown, Filter, Sparkles, AlertCircle } from 'lucide-react'

export default function ModelComparison() {
  const [experiments, setExperiments] = useState([])
  const [loading, setLoading] = useState(true)
  const [selectedArch, setSelectedArch] = useState('all')
  const [sortField, setSortField] = useState('MAPE_percent')
  const [sortAsc, setSortAsc] = useState(true)

  useEffect(() => {
    fetch('/api/experiments')
      .then((res) => res.json())
      .then((data) => {
        setExperiments(data)
        setLoading(false)
      })
      .catch((err) => {
        console.error('Failed to load experiments:', err)
        setLoading(false)
      })
  }, [])

  const filteredExperiments = experiments.filter((exp) => {
    if (selectedArch === 'all') return true
    return exp.model.toLowerCase().includes(selectedArch.toLowerCase())
  })

  const sortedExperiments = [...filteredExperiments].sort((a, b) => {
    const valA = a[sortField]
    const valB = b[sortField]
    if (sortAsc) return valA > valB ? 1 : -1
    return valA < valB ? 1 : -1
  })

  const handleSort = (field) => {
    if (sortField === field) {
      setSortAsc(!sortAsc)
    } else {
      setSortField(field)
      setSortAsc(true)
    }
  }

  // Prepare chart data for 24h sequence length across architectures
  const chartData24h = experiments
    .filter((e) => e.seq_length === 24 && e.horizon === 1 && e.feature_mode === 'multivariate')
    .map((e) => ({
      name: e.model === 'simplernn' ? 'SimpleRNN' : e.model === 'single_lstm' ? 'Single LSTM' : 'Stacked LSTM',
      MAPE: e.MAPE_percent,
      MAE_GW: Number((e.MAE_MW / 1000).toFixed(2)),
      R2: Number((e.R2_Score * 100).toFixed(1)),
    }))

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header */}
      <div className="border-b border-slate-800 pb-5">
        <h2 className="text-2xl font-bold text-white tracking-tight">
          Comprehensive Model Benchmarking & Empirical Comparisons
        </h2>
        <p className="text-sm text-slate-400 mt-1">
          Rigorous evaluation across 3 neural architectures, 6 sequence lengths (6h to 168h), feature ablations, and forecasting horizons.
        </p>
      </div>

      {/* Highlights & Top Model Banner */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        <div className="bg-gradient-to-br from-blue-950/60 to-slate-900 border border-blue-500/30 rounded-xl p-5 shadow-lg">
          <div className="flex items-center space-x-2 text-blue-400 mb-2">
            <Trophy className="w-5 h-5 text-amber-400" />
            <span className="text-xs font-bold uppercase tracking-wider">Champion Architecture</span>
          </div>
          <div className="text-xl font-bold text-white">Two-Layer Stacked LSTM</div>
          <p className="text-xs text-slate-300 mt-2 leading-relaxed">
            Achieves lowest Test MAPE (3.13%) and highest variance explained (R² &gt; 0.65).
            Hierarchical LSTM layers extract both high-frequency diurnal ramps and multi-day cyclic dependencies.
          </p>
        </div>

        <div className="bg-slate-800/40 border border-slate-700/60 rounded-xl p-5">
          <div className="flex items-center space-x-2 text-slate-300 mb-2">
            <Sparkles className="w-5 h-5 text-indigo-400" />
            <span className="text-xs font-bold uppercase tracking-wider">Optimal Sequence Length</span>
          </div>
          <div className="text-xl font-bold text-white">24 & 48 Hours Lookback</div>
          <p className="text-xs text-slate-300 mt-2 leading-relaxed">
            Empirical results show that 24h to 48h sequence lengths provide the optimal trade-off between capturing diurnal periodicity
            and avoiding gradient attenuation over extended lookbacks.
          </p>
        </div>

        <div className="bg-slate-800/40 border border-slate-700/60 rounded-xl p-5">
          <div className="flex items-center space-x-2 text-slate-300 mb-2">
            <AlertCircle className="w-5 h-5 text-emerald-400" />
            <span className="text-xs font-bold uppercase tracking-wider">Renewable Feature Impact</span>
          </div>
          <div className="text-xl font-bold text-emerald-400">Solar & Wind Integration</div>
          <p className="text-xs text-slate-300 mt-2 leading-relaxed">
            Multivariate models incorporating Solar PV and Wind generation outperform univariate baselines by up to 14% lower MAE,
            reflecting the co-variability of solar ramps with daytime cooling demand.
          </p>
        </div>
      </div>

      {/* Visual Chart Comparison */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-slate-800/40 border border-slate-700/60 rounded-xl p-6">
          <h3 className="text-sm font-bold text-white mb-1">
            Architecture Performance at 24-Hour Lookback
          </h3>
          <p className="text-xs text-slate-400 mb-4">
            Comparison of Test MAPE (%) across SimpleRNN, Single LSTM, and Stacked LSTM models.
          </p>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData24h} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.6} />
                <XAxis dataKey="name" stroke="#94a3b8" tick={{ fontSize: 12 }} />
                <YAxis stroke="#94a3b8" tick={{ fontSize: 12 }} unit="%" domain={[0, 'dataMax + 2']} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }}
                  labelStyle={{ color: '#cbd5e1', fontWeight: 'bold' }}
                  formatter={(val) => [`${val}%`, 'Test MAPE']}
                />
                <Bar dataKey="MAPE" fill="#3b82f6" radius={[6, 6, 0, 0]} name="Test MAPE (%)" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-slate-800/40 border border-slate-700/60 rounded-xl p-6 flex flex-col justify-between">
          <div>
            <h3 className="text-sm font-bold text-white mb-1">
              Sequence Length Sensitivity Analysis (6h &rarr; 168h)
            </h3>
            <p className="text-xs text-slate-400 mb-3">
              Generated multi-metric comparative trajectory across all sequence lookback windows.
            </p>
          </div>
          <div className="overflow-hidden rounded-lg border border-slate-700">
            <img
              src="/static/figures/model_comparison_sequence_lengths.png"
              alt="Sequence Length Comparison"
              className="w-full object-cover"
              onError={(e) => {
                e.target.style.display = 'none'
              }}
            />
          </div>
        </div>
      </div>

      {/* Filter and Interactive Table */}
      <div className="bg-slate-800/40 border border-slate-700/60 rounded-xl p-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-700/60">
          <div>
            <h3 className="text-base font-bold text-white">Full Experimental Metrics Matrix</h3>
            <p className="text-xs text-slate-400">Click table headers to sort by MAE, RMSE, MAPE, or R².</p>
          </div>

          <div className="flex items-center space-x-2">
            <Filter className="w-4 h-4 text-slate-400" />
            <select
              value={selectedArch}
              onChange={(e) => setSelectedArch(e.target.value)}
              className="bg-slate-800 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:ring-1 focus:ring-blue-500"
            >
              <option value="all">All Architectures</option>
              <option value="simplernn">SimpleRNN</option>
              <option value="single_lstm">Single LSTM</option>
              <option value="stacked_lstm">Stacked LSTM</option>
            </select>
          </div>
        </div>

        <div className="overflow-x-auto mt-4">
          {loading ? (
            <div className="py-12 text-center text-slate-500 font-mono text-sm">
              Loading benchmark experimental records...
            </div>
          ) : (
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-800/80 text-slate-300 uppercase tracking-wider font-semibold border-b border-slate-700">
                <tr>
                  <th className="py-3 px-4">Architecture</th>
                  <th className="py-3 px-3 cursor-pointer hover:text-white" onClick={() => handleSort('seq_length')}>
                    <div className="flex items-center space-x-1">
                      <span>Lookback</span>
                      <ArrowUpDown className="w-3 h-3" />
                    </div>
                  </th>
                  <th className="py-3 px-3">Features</th>
                  <th className="py-3 px-3">Horizon</th>
                  <th className="py-3 px-3 cursor-pointer hover:text-white" onClick={() => handleSort('MAE_MW')}>
                    <div className="flex items-center space-x-1">
                      <span>MAE (MW)</span>
                      <ArrowUpDown className="w-3 h-3" />
                    </div>
                  </th>
                  <th className="py-3 px-3 cursor-pointer hover:text-white" onClick={() => handleSort('RMSE_MW')}>
                    <div className="flex items-center space-x-1">
                      <span>RMSE (MW)</span>
                      <ArrowUpDown className="w-3 h-3" />
                    </div>
                  </th>
                  <th className="py-3 px-3 cursor-pointer hover:text-white" onClick={() => handleSort('MAPE_percent')}>
                    <div className="flex items-center space-x-1">
                      <span>MAPE (%)</span>
                      <ArrowUpDown className="w-3 h-3" />
                    </div>
                  </th>
                  <th className="py-3 px-3 cursor-pointer hover:text-white" onClick={() => handleSort('R2_Score')}>
                    <div className="flex items-center space-x-1">
                      <span>R² Score</span>
                      <ArrowUpDown className="w-3 h-3" />
                    </div>
                  </th>
                  <th className="py-3 px-3">Time (s)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800">
                {sortedExperiments.map((exp, idx) => {
                  const isTop = exp.MAPE_percent < 3.25
                  return (
                    <tr
                      key={idx}
                      className={`hover:bg-slate-800/40 transition-colors ${
                        isTop ? 'bg-blue-950/20 font-medium' : ''
                      }`}
                    >
                      <td className="py-3 px-4 font-mono text-white flex items-center space-x-1.5">
                        {isTop && <span className="text-amber-400 text-xs">★</span>}
                        <span>{exp.model}</span>
                      </td>
                      <td className="py-3 px-3 font-mono text-slate-300">{exp.seq_length}h</td>
                      <td className="py-3 px-3">
                        <span
                          className={`px-2 py-0.5 rounded text-[10px] font-semibold ${
                            exp.feature_mode === 'multivariate'
                              ? 'bg-blue-500/10 text-blue-400 border border-blue-500/20'
                              : 'bg-slate-700 text-slate-300'
                          }`}
                        >
                          {exp.feature_mode}
                        </span>
                      </td>
                      <td className="py-3 px-3 font-mono text-slate-400">+{exp.horizon}h</td>
                      <td className="py-3 px-3 font-mono text-slate-200">{exp.MAE_MW.toLocaleString()}</td>
                      <td className="py-3 px-3 font-mono text-slate-200">{exp.RMSE_MW.toLocaleString()}</td>
                      <td className="py-3 px-3 font-mono font-bold text-blue-400">{exp.MAPE_percent}%</td>
                      <td className="py-3 px-3 font-mono text-emerald-400">{exp.R2_Score}</td>
                      <td className="py-3 px-3 font-mono text-slate-400">{exp.training_time_sec}s</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  )
}

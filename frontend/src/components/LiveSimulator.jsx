import React, { useState } from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Area,
  ComposedChart
} from 'recharts'
import { Play, RotateCcw, AlertTriangle, CheckCircle, Zap, TrendingUp, Clock } from 'lucide-react'

export default function LiveSimulator() {
  const [modelArch, setModelArch] = useState('stacked_lstm')
  const [seqLength, setSeqLength] = useState(24)
  const [featureMode, setFeatureMode] = useState('multivariate')
  const [horizon, setHorizon] = useState(24)
  const [offsetHours, setOffsetHours] = useState(0)

  const [loading, setLoading] = useState(false)
  const [forecastResult, setForecastResult] = useState(null)
  const [errorMsg, setErrorMsg] = useState(null)

  const handleRunForecast = async () => {
    setLoading(true)
    setErrorMsg(null)

    try {
      const res = await fetch('/api/forecast', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_arch: modelArch,
          seq_length: seqLength,
          feature_mode: featureMode,
          horizon: horizon,
          offset_hours: offsetHours
        })
      })

      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Failed to generate forecast.')
      }

      const data = await res.json()
      setForecastResult(data)
    } catch (err) {
      console.error(err)
      setErrorMsg(err.message)
    } finally {
      setLoading(false)
    }
  }

  // Format chart data combining lookback and forecast
  const chartData = forecastResult
    ? forecastResult.forecast.map((f) => ({
        step: `+${f.step}h`,
        time: f.timestamp.slice(5, 16).replace('T', ' '),
        forecast_mw: f.forecast_mw,
        ci_lower_mw: f.ci_lower_mw,
        ci_upper_mw: f.ci_upper_mw,
      }))
    : []

  // Compute forecast stats
  const peakForecast = forecastResult
    ? Math.max(...forecastResult.forecast.map((f) => f.forecast_mw))
    : null
  const minForecast = forecastResult
    ? Math.min(...forecastResult.forecast.map((f) => f.forecast_mw))
    : null
  const avgForecast = forecastResult
    ? forecastResult.forecast.reduce((a, b) => a + b.forecast_mw, 0) / forecastResult.forecast.length
    : null

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header */}
      <div className="border-b border-slate-800 pb-5">
        <h2 className="text-2xl font-bold text-white tracking-tight">
          Live Interactive Forecasting Simulator
        </h2>
        <p className="text-sm text-slate-400 mt-1">
          Perform real-time autoregressive inference using the trained Keras models on arbitrary sequence windows and forecast horizons.
        </p>
      </div>

      {/* Simulator Control Panel */}
      <div className="bg-slate-800/40 border border-slate-700/60 rounded-xl p-6 shadow-sm">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
          <div>
            <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2">
              Model Architecture
            </label>
            <select
              value={modelArch}
              onChange={(e) => setModelArch(e.target.value)}
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="stacked_lstm">Two-Layer Stacked LSTM (Recommended)</option>
              <option value="single_lstm">Single-Layer LSTM</option>
              <option value="simplernn">SimpleRNN</option>
            </select>
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2">
              Sequence Lookback Window
            </label>
            <select
              value={seqLength}
              onChange={(e) => setSeqLength(Number(e.target.value))}
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value={6}>6 Hours (Immediate Ramp)</option>
              <option value={12}>12 Hours (Half-Day)</option>
              <option value={24}>24 Hours (Full Diurnal Cycle)</option>
              <option value={48}>48 Hours (2 Days)</option>
              <option value={72}>72 Hours (3 Days)</option>
              <option value={168}>168 Hours (Full Week Lookback)</option>
            </select>
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2">
              Forecast Horizon
            </label>
            <select
              value={horizon}
              onChange={(e) => setHorizon(Number(e.target.value))}
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value={6}>Next 6 Hours</option>
              <option value={12}>Next 12 Hours</option>
              <option value={24}>Next 24 Hours (Day Ahead)</option>
              <option value={48}>Next 48 Hours (Two Days Ahead)</option>
            </select>
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2">
              Feature Mode
            </label>
            <select
              value={featureMode}
              onChange={(e) => setFeatureMode(e.target.value)}
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="multivariate">Multivariate (Demand + Solar + Wind + Time)</option>
              <option value="univariate">Univariate (Demand Only)</option>
            </select>
          </div>
        </div>

        <div className="mt-6 flex flex-wrap items-center justify-between pt-5 border-t border-slate-700/60 gap-4">
          <div className="flex items-center space-x-2 text-xs text-slate-400">
            <Clock className="w-4 h-4 text-blue-400" />
            <span>Autoregressive recursive dispatch loop with ±3.5% empirical confidence intervals</span>
          </div>

          <button
            onClick={handleRunForecast}
            disabled={loading}
            className={`flex items-center space-x-2 px-6 py-2.5 rounded-xl font-semibold text-sm transition-all shadow-lg ${
              loading
                ? 'bg-blue-600/50 text-slate-300 cursor-not-allowed'
                : 'bg-blue-600 hover:bg-blue-500 text-white shadow-blue-600/25 cursor-pointer'
            }`}
          >
            <Play className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
            <span>{loading ? 'Executing Neural Inference...' : 'Generate Demand Forecast'}</span>
          </button>
        </div>
      </div>

      {errorMsg && (
        <div className="p-4 bg-red-950/40 border border-red-500/30 rounded-xl text-red-300 text-xs flex items-center space-x-2">
          <AlertTriangle className="w-4 h-4 text-red-400" />
          <span>{errorMsg}</span>
        </div>
      )}

      {/* Forecast Output Telemetry & Visualization */}
      {forecastResult && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="bg-slate-800/40 border border-slate-700/60 rounded-xl p-4">
              <span className="text-xs text-slate-400 font-semibold uppercase">Forecast Peak Demand</span>
              <div className="text-2xl font-bold text-white font-mono mt-1">
                {(peakForecast / 1000).toFixed(1)} <span className="text-sm font-sans font-normal text-slate-400">GW</span>
              </div>
              <div className="text-xs text-blue-400 mt-1 font-mono">
                {peakForecast.toLocaleString()} MW
              </div>
            </div>

            <div className="bg-slate-800/40 border border-slate-700/60 rounded-xl p-4">
              <span className="text-xs text-slate-400 font-semibold uppercase">Forecast Minimum Trough</span>
              <div className="text-2xl font-bold text-white font-mono mt-1">
                {(minForecast / 1000).toFixed(1)} <span className="text-sm font-sans font-normal text-slate-400">GW</span>
              </div>
              <div className="text-xs text-emerald-400 mt-1 font-mono">
                {minForecast.toLocaleString()} MW
              </div>
            </div>

            <div className="bg-slate-800/40 border border-slate-700/60 rounded-xl p-4">
              <span className="text-xs text-slate-400 font-semibold uppercase">Mean Forecasted Load</span>
              <div className="text-2xl font-bold text-white font-mono mt-1">
                {(avgForecast / 1000).toFixed(1)} <span className="text-sm font-sans font-normal text-slate-400">GW</span>
              </div>
              <div className="text-xs text-slate-400 mt-1 font-mono">
                {avgForecast.toLocaleString()} MW
              </div>
            </div>
          </div>

          <div className="bg-slate-800/40 border border-slate-700/60 rounded-xl p-6">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="text-base font-bold text-white">
                  Predicted Demand Trajectory & 95% Confidence Band
                </h3>
                <p className="text-xs text-slate-400">
                  Model: {forecastResult.model} | Lookback: {forecastResult.seq_length}h | Horizon: +{forecastResult.horizon}h
                </p>
              </div>
            </div>

            <div className="h-80 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={chartData} margin={{ top: 10, right: 30, left: 10, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.6} />
                  <XAxis dataKey="step" stroke="#94a3b8" tick={{ fontSize: 11 }} />
                  <YAxis
                    stroke="#94a3b8"
                    tick={{ fontSize: 11 }}
                    domain={['dataMin - 15000', 'dataMax + 15000']}
                    tickFormatter={(val) => `${(val / 1000).toFixed(0)}k`}
                  />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }}
                    labelStyle={{ color: '#cbd5e1', fontWeight: 'bold' }}
                    formatter={(val, name) => [`${Number(val).toLocaleString()} MW`, name]}
                  />
                  <Legend wrapperStyle={{ paddingTop: '10px' }} />
                  <Area
                    type="monotone"
                    dataKey="ci_upper_mw"
                    stroke="none"
                    fill="#3b82f6"
                    fillOpacity={0.15}
                    name="Upper Bound (95% CI)"
                  />
                  <Line
                    type="monotone"
                    dataKey="forecast_mw"
                    stroke="#3b82f6"
                    strokeWidth={2.5}
                    dot={{ r: 4, fill: '#3b82f6' }}
                    name="Forecasted Demand (MW)"
                  />
                  <Line
                    type="monotone"
                    dataKey="ci_lower_mw"
                    stroke="#64748b"
                    strokeDasharray="2 2"
                    strokeWidth={1.2}
                    dot={false}
                    name="Lower Bound (95% CI)"
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

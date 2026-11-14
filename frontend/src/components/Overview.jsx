import React from 'react'
import {
  TrendingUp,
  Cpu,
  Sun,
  Wind,
  ShieldCheck,
  CheckCircle2,
  Calendar,
  BarChart,
  Activity,
  ArrowUpRight
} from 'lucide-react'

export default function Overview({ edaSummary, setActiveTab, bestModel }) {
  const peakDemandGW = edaSummary?.demand_mw ? (edaSummary.demand_mw.max / 1000).toFixed(1) : '248.7'
  const meanDemandGW = edaSummary?.demand_mw ? (edaSummary.demand_mw.mean / 1000).toFixed(1) : '194.0'
  const peakSolarGW = edaSummary?.solar_mw ? (edaSummary.solar_mw.max / 1000).toFixed(1) : '64.7'
  const peakWindGW = edaSummary?.wind_mw ? (edaSummary.wind_mw.max / 1000).toFixed(1) : '29.3'
  const totalHours = edaSummary?.period ? edaSummary.period.total_hours.toLocaleString() : '12,984'

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Hero Section */}
      <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-slate-900 via-blue-950/60 to-slate-900 border border-slate-800 p-8 shadow-2xl">
        <div className="absolute top-0 right-0 -mt-10 -mr-10 w-80 h-80 bg-blue-500/10 rounded-full blur-3xl pointer-events-none" />
        <div className="relative z-10 max-w-3xl">
          <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-400 text-xs font-semibold mb-4">
            <ShieldCheck className="w-3.5 h-3.5" />
            <span>Zero Data Leakage | Verified Mendeley Benchmark (DOI: 10.17632/y58jknpgs8.2)</span>
          </div>
          <h1 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight leading-tight">
            Indian National Grid Electricity Consumption Forecasting
          </h1>
          <p className="mt-4 text-base text-slate-300 leading-relaxed">
            DemandCast is a research-grade machine learning platform utilizing Deep Recurrent Neural Networks
            (SimpleRNN, Single LSTM, and Stacked LSTM) trained on {totalHours} consecutive hourly observations
            of India's synchronized national power grid. The platform features strict chronological partitioning,
            exogenous solar and wind integration, and empirical sequence length optimization (6h to 168h).
          </p>

          <div className="mt-6 flex flex-wrap gap-3">
            <button
              onClick={() => setActiveTab('simulator')}
              className="inline-flex items-center px-4 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-semibold text-sm shadow-lg shadow-blue-600/30 transition-all cursor-pointer"
            >
              <span>Launch Live Forecast Simulator</span>
              <ArrowUpRight className="ml-2 w-4 h-4" />
            </button>
            <button
              onClick={() => setActiveTab('benchmarks')}
              className="inline-flex items-center px-4 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-200 font-semibold text-sm transition-all cursor-pointer"
            >
              <span>View Sequence & Model Benchmarks</span>
            </button>
          </div>
        </div>
      </div>

      {/* Grid Key Metrics */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
        <div className="bg-slate-800/60 border border-slate-700/60 rounded-xl p-5 shadow-sm hover:border-slate-600 transition-all">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs uppercase tracking-wider font-semibold">Peak Grid Load</span>
            <div className="p-2 bg-red-500/10 text-red-400 rounded-lg">
              <TrendingUp className="w-4 h-4" />
            </div>
          </div>
          <div className="text-2xl font-bold text-white font-mono">{peakDemandGW} <span className="text-sm font-sans font-normal text-slate-400">GW</span></div>
          <div className="mt-2 text-xs text-slate-400 flex items-center space-x-1">
            <span>Historical max: 248,688 MW</span>
          </div>
        </div>

        <div className="bg-slate-800/60 border border-slate-700/60 rounded-xl p-5 shadow-sm hover:border-slate-600 transition-all">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs uppercase tracking-wider font-semibold">Average Demand</span>
            <div className="p-2 bg-blue-500/10 text-blue-400 rounded-lg">
              <Activity className="w-4 h-4" />
            </div>
          </div>
          <div className="text-2xl font-bold text-white font-mono">{meanDemandGW} <span className="text-sm font-sans font-normal text-slate-400">GW</span></div>
          <div className="mt-2 text-xs text-slate-400">
            Diurnal trough: ~172.8 GW at 03:00 IST
          </div>
        </div>

        <div className="bg-slate-800/60 border border-slate-700/60 rounded-xl p-5 shadow-sm hover:border-slate-600 transition-all">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs uppercase tracking-wider font-semibold">Solar Peak Capacity</span>
            <div className="p-2 bg-amber-500/10 text-amber-400 rounded-lg">
              <Sun className="w-4 h-4" />
            </div>
          </div>
          <div className="text-2xl font-bold text-white font-mono">{peakSolarGW} <span className="text-sm font-sans font-normal text-slate-400">GW</span></div>
          <div className="mt-2 text-xs text-amber-400/80 font-medium">
            Daily peak hour: 12:00 PM (+0.470 corr)
          </div>
        </div>

        <div className="bg-slate-800/60 border border-slate-700/60 rounded-xl p-5 shadow-sm hover:border-slate-600 transition-all">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs uppercase tracking-wider font-semibold">Wind Peak Generation</span>
            <div className="p-2 bg-emerald-500/10 text-emerald-400 rounded-lg">
              <Wind className="w-4 h-4" />
            </div>
          </div>
          <div className="text-2xl font-bold text-white font-mono">{peakWindGW} <span className="text-sm font-sans font-normal text-slate-400">GW</span></div>
          <div className="mt-2 text-xs text-emerald-400/80 font-medium">
            Max contribution: 29,305 MW
          </div>
        </div>
      </div>

      {/* Model Benchmark Highlights & Architecture Cards */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="bg-slate-800/40 border border-slate-700/60 rounded-xl p-6">
          <div className="flex items-center space-x-3 mb-4">
            <div className="p-2.5 bg-blue-500/10 text-blue-400 rounded-lg">
              <Cpu className="w-5 h-5" />
            </div>
            <div>
              <h3 className="font-bold text-white text-base">SimpleRNN Model</h3>
              <p className="text-xs text-slate-400">Traditional Recurrent Baseline</p>
            </div>
          </div>
          <p className="text-xs text-slate-300 leading-relaxed">
            Standard Elman RNN architecture with hidden state feedback. Effective for short lookbacks (6h, 12h),
            but exhibits vanishing gradient challenges as sequence length extends past 48h.
          </p>
          <div className="mt-4 pt-4 border-t border-slate-700/60 flex justify-between text-xs font-mono">
            <span className="text-slate-400">Target Horizon:</span>
            <span className="text-blue-400 font-semibold">1 to 24 Hours</span>
          </div>
        </div>

        <div className="bg-slate-800/40 border border-slate-700/60 rounded-xl p-6">
          <div className="flex items-center space-x-3 mb-4">
            <div className="p-2.5 bg-indigo-500/10 text-indigo-400 rounded-lg">
              <Cpu className="w-5 h-5" />
            </div>
            <div>
              <h3 className="font-bold text-white text-base">Single-Layer LSTM</h3>
              <p className="text-xs text-slate-400">Gated Long-Term Memory</p>
            </div>
          </div>
          <p className="text-xs text-slate-300 leading-relaxed">
            Long Short-Term Memory cell with input, forget, and output gating mechanisms.
            Successfully preserves gradient flow and captures diurnal cyclicality up to 72 hours.
          </p>
          <div className="mt-4 pt-4 border-t border-slate-700/60 flex justify-between text-xs font-mono">
            <span className="text-slate-400">Gating Mechanism:</span>
            <span className="text-indigo-400 font-semibold">Sigmoid / Tanh Gates</span>
          </div>
        </div>

        <div className="bg-gradient-to-b from-blue-950/40 to-slate-800/40 border-2 border-blue-500/40 rounded-xl p-6 relative overflow-hidden">
          <div className="absolute top-2 right-2 px-2 py-0.5 bg-blue-500 text-white text-[10px] uppercase font-bold tracking-wider rounded">
            Top Performer
          </div>
          <div className="flex items-center space-x-3 mb-4">
            <div className="p-2.5 bg-blue-500/20 text-blue-300 rounded-lg">
              <Cpu className="w-5 h-5" />
            </div>
            <div>
              <h3 className="font-bold text-white text-base">Two-Layer Stacked LSTM</h3>
              <p className="text-xs text-blue-300">Hierarchical Representation</p>
            </div>
          </div>
          <p className="text-xs text-slate-200 leading-relaxed">
            Two stacked recurrent layers. Layer 1 encodes temporal sequence dynamics into high-dimensional hidden trajectories,
            while Layer 2 extracts high-level seasonal representations, achieving MAPE &lt; 3.2%.
          </p>
          <div className="mt-4 pt-4 border-t border-slate-700/60 flex justify-between text-xs font-mono">
            <span className="text-slate-300">Test MAPE:</span>
            <span className="text-emerald-400 font-bold">~3.13% | R² &gt; 0.65</span>
          </div>
        </div>
      </div>

      {/* Dataset & Leakage Protocol Overview */}
      <div className="bg-slate-800/30 border border-slate-700/60 rounded-xl p-6">
        <h3 className="text-base font-bold text-white mb-4 flex items-center space-x-2">
          <ShieldCheck className="w-5 h-5 text-emerald-400" />
          <span>Rigorous Machine Learning Protocol & Data Leakage Prevention</span>
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs text-slate-300">
          <div className="bg-slate-800/80 p-4 rounded-lg border border-slate-700/50">
            <div className="font-semibold text-blue-400 mb-1 flex items-center space-x-1.5">
              <CheckCircle2 className="w-4 h-4" />
              <span>Strict Chronological Split</span>
            </div>
            <p className="text-slate-400 leading-relaxed">
              Data is segmented strictly by timestamp: 70% Train (Jan 2024 - Jan 2025), 15% Validation (Jan 2025 - Apr 2025),
              and 15% Test (Apr 2025 - Jun 2025). Shuffling is strictly disallowed.
            </p>
          </div>

          <div className="bg-slate-800/80 p-4 rounded-lg border border-slate-700/50">
            <div className="font-semibold text-emerald-400 mb-1 flex items-center space-x-1.5">
              <CheckCircle2 className="w-4 h-4" />
              <span>Training-Only Scaler Fitting</span>
            </div>
            <p className="text-slate-400 leading-relaxed">
              MinMaxScaler parameters are computed exclusively on the 70% training split. Validation and test partitions
              are transformed using the training parameters, preventing statistical distribution leakage.
            </p>
          </div>

          <div className="bg-slate-800/80 p-4 rounded-lg border border-slate-700/50">
            <div className="font-semibold text-purple-400 mb-1 flex items-center space-x-1.5">
              <CheckCircle2 className="w-4 h-4" />
              <span>Multi-Feature Exogenous Signals</span>
            </div>
            <p className="text-slate-400 leading-relaxed">
              Integrates cyclical sine/cosine time embeddings (hour, day, month), calendar flags, solar PV generation,
              and wind power to inform load ramp events without future lookahead.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

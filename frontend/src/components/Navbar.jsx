import React from 'react'
import { Zap, Activity, BarChart3, LineChart, BookOpen, Layers } from 'lucide-react'

export default function Navbar({ activeTab, setActiveTab, backendHealthy }) {
  const navItems = [
    { id: 'overview', label: 'Executive Overview', icon: Activity },
     { id: 'simulator', label: 'Live Forecast Simulator', icon: LineChart },
    { id: 'eda', label: 'Grid Consumption & Renewables EDA', icon: BarChart3 },
    { id: 'benchmarks', label: 'Model Benchmarking & Sequences', icon: Layers }

    
  ]

  return (
    <header className="sticky top-0 z-50 bg-slate-900/90 backdrop-blur-md border-b border-slate-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center space-x-3 cursor-pointer" onClick={() => setActiveTab('overview')}>
            <div className="p-2 bg-gradient-to-tr from-blue-600 to-indigo-500 rounded-xl shadow-lg shadow-blue-500/20 text-white">
              <Zap className="w-6 h-6" />
            </div>
            <div>
              <span className="text-xl font-bold bg-gradient-to-r from-blue-400 via-indigo-300 to-white bg-clip-text text-transparent">
                DemandCast
              </span>
              <span className="ml-2 text-xs text-slate-400 font-mono hidden sm:inline-block">
                LSTM Electricity Forecasting
              </span>
            </div>
          </div>

          <div className="flex items-center space-x-4">
            <nav className="hidden md:flex space-x-1">
              {navItems.map((item) => {
                const Icon = item.icon
                const isActive = activeTab === item.id
                return (
                  <button
                    key={item.id}
                    onClick={() => setActiveTab(item.id)}
                    className={`flex items-center space-x-2 px-3 py-2 rounded-lg text-xs font-medium transition-all ${
                      isActive
                        ? 'bg-blue-600/20 text-blue-400 border border-blue-500/30 shadow-sm shadow-blue-500/10'
                        : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
                    }`}
                  >
                    <Icon className="w-4 h-4" />
                    <span>{item.label}</span>
                  </button>
                )
              })}
            </nav>

            <div className="flex items-center space-x-2 pl-2 border-l border-slate-800">
              
              
            </div>
          </div>
        </div>

        {/* Mobile Navigation bar */}
        <div className="md:hidden flex overflow-x-auto py-2 space-x-2 border-t border-slate-800/60">
          {navItems.map((item) => {
            const Icon = item.icon
            const isActive = activeTab === item.id
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`flex items-center space-x-1 px-3 py-1.5 rounded-lg text-xs font-medium whitespace-nowrap ${
                  isActive
                    ? 'bg-blue-600/20 text-blue-400 border border-blue-500/30'
                    : 'text-slate-400 hover:bg-slate-800'
                }`}
              >
                <Icon className="w-3.5 h-3.5" />
                <span>{item.label}</span>
              </button>
            )
          })}
        </div>
      </div>
    </header>
  )
}

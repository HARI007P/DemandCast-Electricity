import React, { useState, useEffect } from 'react'
import Navbar from './components/Navbar'
import Overview from './components/Overview'
import EDAPanel from './components/EDAPanel'
import ModelComparison from './components/ModelComparison'
import LiveSimulator from './components/LiveSimulator'


export default function App() {
  const [activeTab, setActiveTab] = useState('overview')
  const [backendHealthy, setBackendHealthy] = useState(false)
  const [edaSummary, setEdaSummary] = useState(null)

  useEffect(() => {
    // Check backend health
    const checkHealth = async () => {
      try {
        const res = await fetch('/api/health')
        if (res.ok) {
          setBackendHealthy(true)
        }
      } catch (err) {
        setBackendHealthy(false)
      }
    }

    // Fetch EDA summary
    const fetchEda = async () => {
      try {
        const res = await fetch('/api/eda/summary')
        if (res.ok) {
          const data = await res.json()
          setEdaSummary(data)
        }
      } catch (err) {
        console.warn('Backend not ready yet for EDA summary:', err)
      }
    }

    checkHealth()
    fetchEda()
    const interval = setInterval(checkHealth, 10000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        backendHealthy={backendHealthy}
      />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === 'overview' && (
          <Overview
            edaSummary={edaSummary}
            setActiveTab={setActiveTab}
          />
        )}

        {activeTab === 'eda' && (
          <EDAPanel edaSummary={edaSummary} />
        )}

        {activeTab === 'benchmarks' && (
          <ModelComparison />
        )}

        {activeTab === 'simulator' && (
          <LiveSimulator />
        )}

        
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 bg-slate-900/60 py-6">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between text-xs text-slate-500 gap-4">
          <div>
            <span className="font-semibold text-slate-400">DemandCast Project</span> — Deep Learning Electricity Consumption Forecasting.
          </div>
          <div className="flex items-center space-x-4">
            <span>Grid-India SCADA</span>
            <span>•</span>
            
            <span>•</span>
           
          </div>
        </div>
      </footer>
    </div>
  )
}

import { Route, Routes } from 'react-router-dom'
import AuroraBackground from '@/components/AuroraBackground'
import DashboardPage from '@/pages/DashboardPage'
import ScamShieldPage from '@/pages/ScamShieldPage'
import MoneyInsightPage from '@/pages/MoneyInsightPage'
import ReceiptScannerPage from '@/pages/ReceiptScannerPage'
import CallShieldPage from '@/pages/CallShieldPage'
import PaymentPausePage from '@/pages/PaymentPausePage'

function App() {
  return (
    <div className="relative min-h-screen">
      <AuroraBackground />
      <Routes>
        <Route path="/" element={<DashboardPage />} />
        <Route path="/scam-shield" element={<ScamShieldPage />} />
        <Route path="/money-insight" element={<MoneyInsightPage />} />
        <Route path="/receipt-scanner" element={<ReceiptScannerPage />} />
        <Route path="/call-shield" element={<CallShieldPage />} />
        <Route path="/payment-pause" element={<PaymentPausePage />} />
      </Routes>
    </div>
  )
}

export default App

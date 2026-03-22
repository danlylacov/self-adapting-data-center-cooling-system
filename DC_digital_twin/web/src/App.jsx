import { BrowserRouter, Link, Route, Routes } from 'react-router-dom'
import HomePage from './HomePage.jsx'
import ConfigPage from './ConfigPage.jsx'
import OrchestratorPage from './OrchestratorPage.jsx'
import OrchestratorGAPage from './OrchestratorGAPage.jsx'

export default function App() {
  return (
    <BrowserRouter>
      <nav className="globalNav">
        <Link to="/">Прогон</Link>
        <span className="navSep">|</span>
        <Link to="/config">Конфигурация</Link>
        <span className="navSep">|</span>
        <Link to="/orch">ML-прогон</Link>
        <span className="navSep">|</span>
        <Link to="/orch-ga">GA-прогон</Link>
      </nav>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/config" element={<ConfigPage />} />
        <Route path="/orch" element={<OrchestratorPage />} />
        <Route path="/orch-ga" element={<OrchestratorGAPage />} />
      </Routes>
    </BrowserRouter>
  )
}

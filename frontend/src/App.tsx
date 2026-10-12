import { NavLink, Navigate, Route, Routes } from "react-router-dom";
import { useAuth } from "./auth";
import DashboardPage from "./pages/DashboardPage";
import WindowEventsPage from "./pages/WindowEventsPage";
import JobsPage from "./pages/JobsPage";
import LoginPage from "./pages/LoginPage";
import LowFrequencyPage from "./pages/LowFrequencyPage";
import ModelsPage from "./pages/ModelsPage";
import SitesPage from "./pages/SitesPage";

export default function App() {
  const { authenticated, loading, logout } = useAuth();

  if (loading) {
    return <p className="loading">正在加载…</p>;
  }
  if (!authenticated) {
    return (
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route path="*" element={<Navigate to="/login" replace />} />
      </Routes>
    );
  }
  return (
    <div className="shell">
      <header className="shell-header">
        <span className="brand">DAS Sentinel</span>
        <nav aria-label="主导航" className="lf-nav">
          <NavLink to="/dashboard">运行总览</NavLink>
          <NavLink to="/sites">站点</NavLink>
          <NavLink to="/jobs">任务</NavLink>
          <NavLink to="/events">事件复核</NavLink>
          <NavLink to="/low-frequency">低频扫描</NavLink>
          <NavLink to="/models">模型</NavLink>
        </nav>
        <button type="button" className="link-button" onClick={() => void logout()}>
          退出登录
        </button>
      </header>
      <main>
        <Routes>
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/sites" element={<SitesPage />} />
          <Route path="/jobs" element={<JobsPage />} />
          <Route path="/events" element={<WindowEventsPage />} />
          <Route path="/events/legacy" element={<Navigate to="/events" replace />} />
          <Route path="/low-frequency" element={<LowFrequencyPage />} />
          <Route path="/models" element={<ModelsPage />} />
          <Route path="*" element={<Navigate to="/dashboard" replace />} />
        </Routes>
      </main>
    </div>
  );
}

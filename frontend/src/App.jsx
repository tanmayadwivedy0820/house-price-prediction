import { NavLink, Routes, Route } from "react-router-dom";
import Predict from "./pages/Predict.jsx";
import DatasetStats from "./pages/DatasetStats.jsx";
import ModelInsights from "./pages/ModelInsights.jsx";

export default function App() {
  return (
    <div className="app">
      <nav className="nav">
        <div className="brand">Estate<span>·</span>Valuer</div>
        <div className="nav-links">
          <NavLink to="/" end>Predict</NavLink>
          <NavLink to="/data">Dataset</NavLink>
          <NavLink to="/model">Model</NavLink>
        </div>
      </nav>
      <Routes>
        <Route path="/" element={<Predict />} />
        <Route path="/data" element={<DatasetStats />} />
        <Route path="/model" element={<ModelInsights />} />
      </Routes>
    </div>
  );
}

import { useEffect, useState } from "react";
import { getDatasetStats } from "../api";

export default function DatasetStats() {
  const [stats, setStats] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    getDatasetStats().then(setStats).catch((e) => setError(e.message));
  }, []);

  if (error) return <div className="error">{error}</div>;
  if (!stats) return <div className="loading">Loading dataset…</div>;

  const fmt = (n) => new Intl.NumberFormat("en-US").format(Math.round(n));
  const cat = stats.categorical_summary[stats.categorical_features[0]] || {};
  const maxCat = Math.max(...Object.values(cat));

  return (
    <div>
      <div className="page-head">
        <div className="eyebrow">Exploratory Analysis</div>
        <h1>The dataset</h1>
        <p>Summary of the California Housing data the model was trained on.</p>
      </div>

      <div className="metrics">
        <div className="metric"><div className="m-label">Rows</div><div className="m-value">{fmt(stats.n_rows)}</div></div>
        <div className="metric"><div className="m-label">Features</div><div className="m-value">{stats.n_features}</div></div>
        <div className="metric"><div className="m-label">Median Value</div><div className="m-value">${fmt(stats.target_summary.median)}</div></div>
        <div className="metric"><div className="m-label">Mean Value</div><div className="m-value">${fmt(stats.target_summary.mean)}</div></div>
      </div>

      <h2 className="section-title">Ocean Proximity Distribution</h2>
      <div className="card">
        {Object.entries(cat).sort((a, b) => b[1] - a[1]).map(([name, count]) => (
          <div className="bar-row" key={name}>
            <div className="b-name">{name}</div>
            <div className="bar-track"><div className="bar-fill" style={{ width: `${(count / maxCat) * 100}%` }} /></div>
            <div className="b-val">{fmt(count)}</div>
          </div>
        ))}
      </div>

      <h2 className="section-title">Numeric Feature Summary</h2>
      <div className="card">
        <table>
          <thead>
            <tr><th>Feature</th><th>Mean</th><th>Std</th><th>Min</th><th>Median</th><th>Max</th></tr>
          </thead>
          <tbody>
            {stats.numeric_features.map((f) => {
              const s = stats.numeric_summary[f];
              return (
                <tr key={f}>
                  <td>{f}</td>
                  <td>{s.mean.toLocaleString()}</td>
                  <td>{s.std.toLocaleString()}</td>
                  <td>{s.min.toLocaleString()}</td>
                  <td>{s["50%"].toLocaleString()}</td>
                  <td>{s.max.toLocaleString()}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

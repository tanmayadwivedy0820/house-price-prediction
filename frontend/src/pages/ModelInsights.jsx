import { useEffect, useState } from "react";
import { getModelInfo } from "../api";

export default function ModelInsights() {
  const [info, setInfo] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    getModelInfo().then(setInfo).catch((e) => setError(e.message));
  }, []);

  if (error) return <div className="error">{error}</div>;
  if (!info) return <div className="loading">Computing feature importances… (first load takes a few seconds)</div>;

  const maxImp = Math.max(...info.feature_importances.map((f) => f.importance));

  return (
    <div>
      <div className="page-head">
        <div className="eyebrow">Interpretability</div>
        <h1>Model insights</h1>
        <p>How the model is configured and which features drive its predictions.</p>
      </div>

      <div className="metrics">
        <div className="metric"><div className="m-label">Algorithm</div><div className="m-value" style={{ fontSize: 18 }}>HistGB</div></div>
        <div className="metric"><div className="m-label">Features</div><div className="m-value">{info.n_features}</div></div>
        <div className="metric" style={{ gridColumn: "span 2" }}>
          <div className="m-label">Importance Method</div>
          <div className="m-value" style={{ fontSize: 15, fontFamily: "inherit", fontWeight: 500 }}>Permutation (R² drop)</div>
        </div>
      </div>

      <h2 className="section-title">Feature Importances</h2>
      <p className="muted" style={{ marginBottom: 16 }}>
        Each bar shows how much model accuracy (R²) drops when that feature's values are shuffled —
        bigger drop means the model relies on it more.
      </p>
      <div className="card">
        {info.feature_importances.map((f) => (
          <div className="bar-row" key={f.feature}>
            <div className="b-name">{f.feature}</div>
            <div className="bar-track">
              <div className="bar-fill" style={{ width: `${Math.max((f.importance / maxImp) * 100, 1)}%` }} />
            </div>
            <div className="b-val">{f.importance.toFixed(3)}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

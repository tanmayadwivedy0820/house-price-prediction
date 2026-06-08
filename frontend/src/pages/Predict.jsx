import { useState } from "react";
import { predict } from "../api";

// The 8 numeric features with sensible defaults (a real California block group).
const NUMERIC_FIELDS = [
  { name: "longitude", label: "Longitude", step: "0.01" },
  { name: "latitude", label: "Latitude", step: "0.01" },
  { name: "housing_median_age", label: "Housing Median Age", step: "1" },
  { name: "total_rooms", label: "Total Rooms", step: "1" },
  { name: "total_bedrooms", label: "Total Bedrooms", step: "1" },
  { name: "population", label: "Population", step: "1" },
  { name: "households", label: "Households", step: "1" },
  { name: "median_income", label: "Median Income (×$10k)", step: "0.0001" },
];

// Exact category values the OneHotEncoder learned — must match training data.
const OCEAN_OPTIONS = ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"];

const DEFAULTS = {
  longitude: -122.23, latitude: 37.88, housing_median_age: 41,
  total_rooms: 880, total_bedrooms: 129, population: 322,
  households: 126, median_income: 8.3252, ocean_proximity: "NEAR BAY",
};

export default function Predict() {
  const [form, setForm] = useState(DEFAULTS);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const update = (name, value) => setForm((f) => ({ ...f, [name]: value }));

  const onSubmit = async () => {
    setError(""); setResult(null); setLoading(true);
    try {
      // Convert numeric strings to numbers; keep ocean_proximity as string.
      const payload = { ...form };
      NUMERIC_FIELDS.forEach((f) => { payload[f.name] = Number(payload[f.name]); });
      const data = await predict(payload);
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const fmt = (n) =>
    new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 }).format(n);

  return (
    <div>
      <div className="page-head">
        <div className="eyebrow">California Housing</div>
        <h1>Estimate a home's value</h1>
        <p>Enter the block-group details below. The model predicts the median house value
           using a tuned gradient-boosting pipeline served over an API.</p>
      </div>

      <div className="card">
        <div className="grid">
          {NUMERIC_FIELDS.map((f) => (
            <div className="field" key={f.name}>
              <label>{f.label}</label>
              <input
                type="number" step={f.step} value={form[f.name]}
                onChange={(e) => update(f.name, e.target.value)}
              />
            </div>
          ))}
          <div className="field">
            <label>Ocean Proximity</label>
            <select value={form.ocean_proximity} onChange={(e) => update("ocean_proximity", e.target.value)}>
              {OCEAN_OPTIONS.map((o) => <option key={o} value={o}>{o}</option>)}
            </select>
          </div>
        </div>

        <button className="btn" onClick={onSubmit} disabled={loading}>
          {loading ? "Estimating…" : "Estimate Value"}
        </button>

        {error && <div className="error">{error}</div>}

        {result && (
          <div className="result">
            <div className="label">Estimated Median House Value</div>
            <div className="value">{fmt(result.predicted_value)}</div>
            <div className="note">{result.note}</div>
          </div>
        )}
      </div>
    </div>
  );
}

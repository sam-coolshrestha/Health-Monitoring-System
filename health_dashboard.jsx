import { useState, useEffect } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line, AreaChart, Area, RadarChart, PolarGrid, PolarAngleAxis,
  Radar, PieChart, Pie, Cell, ScatterChart, Scatter, ZAxis, Legend
} from "recharts";

// ── DATASET-DERIVED CONSTANTS ───────────────────────────────────────────────
const STATS = {
  total: 5735, avgBMI: 29.4, avgSBP: 125.1, avgDBP: 69.5,
  avgAge: 48.1, hypRate: 38.9, obeseRate: 39.2, smokerRate: 59.4,
};

const BMI_DIST = [
  { label: "Underweight", value: 97, color: "#60a5fa" },
  { label: "Normal", value: 1565, color: "#34d399" },
  { label: "Overweight", value: 1784, color: "#fbbf24" },
  { label: "Obese I", value: 1195, color: "#f97316" },
  { label: "Obese II+", value: 1021, color: "#ef4444" },
];

const HTN_BY_AGE = [
  { age: "18–30", rate: 13.2 }, { age: "31–45", rate: 30.3 },
  { age: "46–60", rate: 47.4 }, { age: "60+", rate: 58.6 },
];

const SBP_HIST = [
  { range: "80–92", count: 27 }, { range: "92–104", count: 126 },
  { range: "104–116", count: 892 }, { range: "116–128", count: 1622 },
  { range: "128–140", count: 1044 }, { range: "140–152", count: 418 },
  { range: "152–164", count: 267 }, { range: "164–176", count: 72 },
  { range: "176–200", count: 48 },
];

const RISK_DIST = [
  { label: "Low Risk (0)", value: 806, color: "#34d399" },
  { label: "Moderate (1)", value: 2550, color: "#fbbf24" },
  { label: "High (2)", value: 1806, color: "#f97316" },
  { label: "Very High (3)", value: 573, color: "#ef4444" },
];

const AGE_DIST = [
  { age: "18–30", count: 1286 }, { age: "31–45", count: 1394 },
  { age: "46–60", count: 1365 }, { age: "60+", count: 1690 },
];

const CORR_DATA = [
  { subject: "Age→SBP", value: 0.46 }, { subject: "BMI→Waist", value: 0.91 },
  { subject: "BMI→Arm", value: 0.87 }, { subject: "BMI→SBP", value: 0.15 },
  { subject: "SBP→DBP", value: 0.32 }, { subject: "Waist→Arm", value: 0.81 },
];

const GENDER_DATA = [
  { name: "Male", value: 2759 }, { name: "Female", value: 2976 },
];

// ── RISK CALCULATOR ──────────────────────────────────────────────────────────
function calcRisk({ age, bmi, sbp, dbp, smoking, alcohol }) {
  let score = 0;
  const factors = [];
  if (sbp >= 130 || dbp >= 80) { score += 30; factors.push({ label: "Hypertension", sev: "high", pts: 30 }); }
  else if (sbp >= 120) { score += 15; factors.push({ label: "Elevated BP", sev: "med", pts: 15 }); }
  if (bmi >= 30) { score += 25; factors.push({ label: "Obesity", sev: "high", pts: 25 }); }
  else if (bmi >= 25) { score += 12; factors.push({ label: "Overweight", sev: "med", pts: 12 }); }
  if (smoking === "yes") { score += 20; factors.push({ label: "Smoker", sev: "high", pts: 20 }); }
  if (age >= 60) { score += 15; factors.push({ label: "Age ≥ 60", sev: "med", pts: 15 }); }
  else if (age >= 45) { score += 8; factors.push({ label: "Age 45–60", sev: "low", pts: 8 }); }
  if (alcohol === "heavy") { score += 10; factors.push({ label: "Heavy Alcohol", sev: "med", pts: 10 }); }
  const pct = Math.min(score, 100);
  const level = pct >= 60 ? "Very High" : pct >= 40 ? "High" : pct >= 20 ? "Moderate" : "Low";
  const color = pct >= 60 ? "#ef4444" : pct >= 40 ? "#f97316" : pct >= 20 ? "#fbbf24" : "#34d399";
  return { pct, level, color, factors };
}

// ── COMPONENTS ───────────────────────────────────────────────────────────────
const Card = ({ children, className = "" }) => (
  <div className={`rounded-2xl bg-gray-900 border border-gray-800 p-5 ${className}`}>{children}</div>
);

const SectionTitle = ({ children }) => (
  <h2 className="text-xs font-bold uppercase tracking-widest text-cyan-400 mb-4">{children}</h2>
);

const StatPill = ({ label, value, sub, color = "text-cyan-300" }) => (
  <div className="flex flex-col items-center bg-gray-800 rounded-xl px-4 py-3 min-w-[100px]">
    <span className={`text-2xl font-black ${color}`}>{value}</span>
    <span className="text-gray-400 text-xs mt-0.5 text-center">{label}</span>
    {sub && <span className="text-gray-600 text-[10px]">{sub}</span>}
  </div>
);

const TOOLTIP_STYLE = {
  contentStyle: { background: "#111827", border: "1px solid #374151", borderRadius: 8, fontSize: 12 },
  labelStyle: { color: "#9ca3af" }, itemStyle: { color: "#e5e7eb" },
};

// ── MAIN DASHBOARD ────────────────────────────────────────────────────────────
export default function HealthDashboard() {
  const [tab, setTab] = useState("overview");
  const [form, setForm] = useState({ age: 45, bmi: 28, sbp: 125, dbp: 80, smoking: "no", alcohol: "light" });
  const [result, setResult] = useState(null);
  const [animated, setAnimated] = useState(false);

  useEffect(() => { setTimeout(() => setAnimated(true), 100); }, []);

  const risk = result || calcRisk(form);

  const TABS = ["overview", "vitals", "risk calculator", "correlations"];

  return (
    <div style={{ fontFamily: "'DM Sans', sans-serif", background: "#030712", minHeight: "100vh", color: "#f3f4f6" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Space+Grotesk:wght@700;800&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 4px; } ::-webkit-scrollbar-track { background: #030712; } ::-webkit-scrollbar-thumb { background: #1f2937; border-radius: 2px; }
        .tab-btn { transition: all 0.2s; }
        .tab-btn:hover { background: rgba(6,182,212,0.1); }
        .tab-btn.active { background: rgba(6,182,212,0.15); border-bottom: 2px solid #06b6d4; }
        .fade-in { opacity: 0; transform: translateY(12px); animation: fadeUp 0.5s forwards; }
        @keyframes fadeUp { to { opacity: 1; transform: translateY(0); } }
        .risk-bar { transition: width 1.2s cubic-bezier(.4,0,.2,1); }
        input[type=range] { accent-color: #06b6d4; }
        select { background: #111827; color: #f3f4f6; border: 1px solid #374151; border-radius: 8px; padding: 6px 10px; font-size: 13px; width: 100%; }
      `}</style>

      {/* HEADER */}
      <div style={{ background: "linear-gradient(135deg, #0c1120 0%, #0f172a 60%, #061018 100%)", borderBottom: "1px solid #1f2937", padding: "20px 24px 0" }}>
        <div style={{ maxWidth: 1100, margin: "0 auto" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 16 }}>
            <div style={{ width: 36, height: 36, borderRadius: 10, background: "linear-gradient(135deg,#06b6d4,#3b82f6)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 18 }}>🩺</div>
            <div>
              <h1 style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: 20, fontWeight: 800, letterSpacing: "-0.5px", color: "#f9fafb" }}>
                Lifestyle Disease Risk Dashboard
              </h1>
              <p style={{ fontSize: 11, color: "#6b7280", marginTop: 1 }}>NHANES Dataset · 5,735 Participants · Health Analytics</p>
            </div>
          </div>
          <div style={{ display: "flex", gap: 4 }}>
            {TABS.map(t => (
              <button key={t} onClick={() => setTab(t)}
                className={`tab-btn ${tab === t ? "active" : ""}`}
                style={{ padding: "8px 16px", fontSize: 12, fontWeight: 600, background: "transparent", border: "none", color: tab === t ? "#06b6d4" : "#9ca3af", cursor: "pointer", borderRadius: "8px 8px 0 0", textTransform: "capitalize" }}>
                {t}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div style={{ maxWidth: 1100, margin: "0 auto", padding: "24px 24px" }}>

        {/* OVERVIEW TAB */}
        {tab === "overview" && (
          <div className="fade-in" style={{ animationDelay: "0s" }}>
            {/* KPI Row */}
            <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginBottom: 20 }}>
              <StatPill label="Participants" value="5,735" color="text-cyan-300" />
              <StatPill label="Avg Age" value={STATS.avgAge} sub="years" color="text-blue-300" />
              <StatPill label="Avg BMI" value={STATS.avgBMI} sub="kg/m²" color="text-yellow-300" />
              <StatPill label="Avg SBP" value={STATS.avgSBP} sub="mmHg" color="text-orange-300" />
              <StatPill label="Hypertension" value={`${STATS.hypRate}%`} color="text-red-400" />
              <StatPill label="Obesity Rate" value={`${STATS.obeseRate}%`} color="text-orange-400" />
              <StatPill label="Smokers" value={`${STATS.smokerRate}%`} color="text-pink-400" />
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }}>
              {/* BMI Distribution */}
              <Card>
                <SectionTitle>BMI Category Distribution</SectionTitle>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={BMI_DIST} barSize={28}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                    <XAxis dataKey="label" tick={{ fill: "#6b7280", fontSize: 11 }} />
                    <YAxis tick={{ fill: "#6b7280", fontSize: 11 }} />
                    <Tooltip {...TOOLTIP_STYLE} />
                    <Bar dataKey="value" radius={[6, 6, 0, 0]}>
                      {BMI_DIST.map((d, i) => <Cell key={i} fill={d.color} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </Card>

              {/* Risk Distribution */}
              <Card>
                <SectionTitle>Disease Risk Level Distribution</SectionTitle>
                <div style={{ display: "flex", gap: 12, height: 200, alignItems: "center" }}>
                  <ResponsiveContainer width="60%" height={180}>
                    <PieChart>
                      <Pie data={RISK_DIST} dataKey="value" cx="50%" cy="50%" innerRadius={50} outerRadius={80} paddingAngle={3}>
                        {RISK_DIST.map((d, i) => <Cell key={i} fill={d.color} />)}
                      </Pie>
                      <Tooltip {...TOOLTIP_STYLE} />
                    </PieChart>
                  </ResponsiveContainer>
                  <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                    {RISK_DIST.map((d, i) => (
                      <div key={i} style={{ display: "flex", alignItems: "center", gap: 6 }}>
                        <div style={{ width: 10, height: 10, borderRadius: 3, background: d.color, flexShrink: 0 }} />
                        <div>
                          <div style={{ fontSize: 11, color: "#e5e7eb", fontWeight: 600 }}>{d.label}</div>
                          <div style={{ fontSize: 10, color: "#6b7280" }}>{d.value.toLocaleString()} people</div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </Card>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
              {/* Age Distribution */}
              <Card>
                <SectionTitle>Age Group Distribution</SectionTitle>
                <ResponsiveContainer width="100%" height={180}>
                  <BarChart data={AGE_DIST}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                    <XAxis dataKey="age" tick={{ fill: "#6b7280", fontSize: 11 }} />
                    <YAxis tick={{ fill: "#6b7280", fontSize: 11 }} />
                    <Tooltip {...TOOLTIP_STYLE} />
                    <Bar dataKey="count" fill="#3b82f6" radius={[6, 6, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </Card>

              {/* Gender split */}
              <Card>
                <SectionTitle>Gender & Key Averages</SectionTitle>
                <div style={{ display: "flex", gap: 20, marginBottom: 12 }}>
                  {GENDER_DATA.map((g, i) => (
                    <div key={i} style={{ flex: 1, background: "#111827", borderRadius: 12, padding: "12px 14px", textAlign: "center" }}>
                      <div style={{ fontSize: 28 }}>{i === 0 ? "♂" : "♀"}</div>
                      <div style={{ fontSize: 20, fontWeight: 800, color: i === 0 ? "#60a5fa" : "#f472b6", marginTop: 4 }}>{g.value.toLocaleString()}</div>
                      <div style={{ fontSize: 11, color: "#6b7280" }}>{g.name}</div>
                    </div>
                  ))}
                </div>
                <div style={{ display: "flex", gap: 8 }}>
                  {[
                    { label: "Avg Waist (M)", val: "100.8 cm" },
                    { label: "Avg Waist (F)", val: "98.4 cm" },
                  ].map((s, i) => (
                    <div key={i} style={{ flex: 1, background: "#0f172a", borderRadius: 8, padding: "8px 10px" }}>
                      <div style={{ fontSize: 14, fontWeight: 700, color: "#e5e7eb" }}>{s.val}</div>
                      <div style={{ fontSize: 10, color: "#6b7280" }}>{s.label}</div>
                    </div>
                  ))}
                </div>
              </Card>
            </div>
          </div>
        )}

        {/* VITALS TAB */}
        {tab === "vitals" && (
          <div className="fade-in" style={{ animationDelay: "0s" }}>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }}>
              {/* SBP Histogram */}
              <Card>
                <SectionTitle>Systolic BP Distribution (mmHg)</SectionTitle>
                <div style={{ fontSize: 11, color: "#6b7280", marginBottom: 8 }}>
                  Avg: <span style={{ color: "#f97316", fontWeight: 700 }}>125.1 mmHg</span> · Normal threshold: 120 mmHg
                </div>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={SBP_HIST}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                    <XAxis dataKey="range" tick={{ fill: "#6b7280", fontSize: 9 }} angle={-30} textAnchor="end" height={40} />
                    <YAxis tick={{ fill: "#6b7280", fontSize: 11 }} />
                    <Tooltip {...TOOLTIP_STYLE} />
                    <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                      {SBP_HIST.map((d, i) => (
                        <Cell key={i} fill={parseInt(d.range) >= 130 ? "#ef4444" : parseInt(d.range) >= 120 ? "#f97316" : "#06b6d4"} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
                <div style={{ display: "flex", gap: 10, marginTop: 8 }}>
                  {[["#06b6d4","Normal (<120)"],["#f97316","Elevated (120–130)"],["#ef4444","High (≥130)"]].map(([c,l],i) => (
                    <div key={i} style={{ display: "flex", alignItems: "center", gap: 4 }}>
                      <div style={{ width: 8, height: 8, borderRadius: 2, background: c }} />
                      <span style={{ fontSize: 9, color: "#9ca3af" }}>{l}</span>
                    </div>
                  ))}
                </div>
              </Card>

              {/* Hypertension by age */}
              <Card>
                <SectionTitle>Hypertension Rate by Age Group (%)</SectionTitle>
                <div style={{ fontSize: 11, color: "#6b7280", marginBottom: 8 }}>
                  Overall rate: <span style={{ color: "#ef4444", fontWeight: 700 }}>38.9%</span> of all participants
                </div>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={HTN_BY_AGE}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                    <XAxis dataKey="age" tick={{ fill: "#6b7280", fontSize: 11 }} />
                    <YAxis tick={{ fill: "#6b7280", fontSize: 11 }} domain={[0, 70]} unit="%" />
                    <Tooltip {...TOOLTIP_STYLE} formatter={(v) => [`${v.toFixed(1)}%`, "Hypertension Rate"]} />
                    <Bar dataKey="rate" radius={[6, 6, 0, 0]}>
                      {HTN_BY_AGE.map((d, i) => (
                        <Cell key={i} fill={`hsl(${360 - i * 30}, 85%, 55%)`} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </Card>
            </div>

            <Card>
              <SectionTitle>Key Vital Statistics Summary</SectionTitle>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12 }}>
                {[
                  { label: "Mean SBP", val: "125.1 mmHg", icon: "🫀", note: "Slightly elevated", color: "#f97316" },
                  { label: "Mean DBP", val: "69.5 mmHg", icon: "💉", note: "Within normal range", color: "#34d399" },
                  { label: "Mean BMI", val: "29.4 kg/m²", icon: "⚖️", note: "Borderline overweight", color: "#fbbf24" },
                  { label: "Mean Waist", val: "99.6 cm", icon: "📏", note: "Above ideal for both genders", color: "#f97316" },
                ].map((s, i) => (
                  <div key={i} style={{ background: "#111827", borderRadius: 12, padding: "14px", borderLeft: `3px solid ${s.color}` }}>
                    <div style={{ fontSize: 22, marginBottom: 6 }}>{s.icon}</div>
                    <div style={{ fontSize: 18, fontWeight: 800, color: s.color }}>{s.val}</div>
                    <div style={{ fontSize: 11, color: "#e5e7eb", fontWeight: 600, marginTop: 2 }}>{s.label}</div>
                    <div style={{ fontSize: 10, color: "#6b7280", marginTop: 2 }}>{s.note}</div>
                  </div>
                ))}
              </div>
            </Card>
          </div>
        )}

        {/* RISK CALCULATOR TAB */}
        {tab === "risk calculator" && (
          <div className="fade-in" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
            <Card>
              <SectionTitle>Enter Your Health Parameters</SectionTitle>
              <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                {[
                  { key: "age", label: "Age", min: 18, max: 90, unit: "years" },
                  { key: "bmi", label: "BMI", min: 15, max: 50, unit: "kg/m²" },
                  { key: "sbp", label: "Systolic BP", min: 80, max: 200, unit: "mmHg" },
                  { key: "dbp", label: "Diastolic BP", min: 50, max: 130, unit: "mmHg" },
                ].map(({ key, label, min, max, unit }) => (
                  <div key={key}>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                      <label style={{ fontSize: 12, color: "#9ca3af", fontWeight: 600 }}>{label}</label>
                      <span style={{ fontSize: 13, color: "#06b6d4", fontWeight: 700 }}>{form[key]} {unit}</span>
                    </div>
                    <input type="range" min={min} max={max} value={form[key]}
                      onChange={e => { setForm(f => ({ ...f, [key]: +e.target.value })); setResult(null); }}
                      style={{ width: "100%", cursor: "pointer" }} />
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 9, color: "#374151" }}>
                      <span>{min}</span><span>{max}</span>
                    </div>
                  </div>
                ))}

                <div>
                  <label style={{ fontSize: 12, color: "#9ca3af", fontWeight: 600, display: "block", marginBottom: 4 }}>Smoking Status</label>
                  <select value={form.smoking} onChange={e => { setForm(f => ({ ...f, smoking: e.target.value })); setResult(null); }}>
                    <option value="no">Non-smoker</option>
                    <option value="yes">Smoker</option>
                  </select>
                </div>

                <div>
                  <label style={{ fontSize: 12, color: "#9ca3af", fontWeight: 600, display: "block", marginBottom: 4 }}>Alcohol Consumption</label>
                  <select value={form.alcohol} onChange={e => { setForm(f => ({ ...f, alcohol: e.target.value })); setResult(null); }}>
                    <option value="none">None</option>
                    <option value="light">Light</option>
                    <option value="heavy">Heavy</option>
                  </select>
                </div>

                <button onClick={() => setResult(calcRisk(form))}
                  style={{ background: "linear-gradient(135deg,#06b6d4,#3b82f6)", border: "none", color: "#fff", fontWeight: 800, fontSize: 14, padding: "12px", borderRadius: 10, cursor: "pointer", letterSpacing: "0.5px" }}>
                  CALCULATE RISK
                </button>
              </div>
            </Card>

            <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
              <Card>
                <SectionTitle>Risk Assessment Result</SectionTitle>
                <div style={{ textAlign: "center", padding: "12px 0" }}>
                  <div style={{ fontSize: 56, fontFamily: "'Space Grotesk', sans-serif", fontWeight: 800, color: risk.color, lineHeight: 1 }}>
                    {risk.pct}%
                  </div>
                  <div style={{ fontSize: 18, fontWeight: 700, color: risk.color, marginTop: 6 }}>{risk.level} Risk</div>
                  <div style={{ background: "#111827", borderRadius: 999, height: 8, margin: "14px 0" }}>
                    <div className="risk-bar" style={{ height: 8, borderRadius: 999, background: `linear-gradient(90deg, #34d399, ${risk.color})`, width: `${animated ? risk.pct : 0}%` }} />
                  </div>
                </div>

                {risk.factors.length > 0 ? (
                  <div>
                    <div style={{ fontSize: 11, color: "#6b7280", fontWeight: 600, marginBottom: 8 }}>CONTRIBUTING FACTORS</div>
                    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                      {risk.factors.map((f, i) => (
                        <div key={i} style={{ display: "flex", alignItems: "center", justifyContent: "space-between", background: "#111827", borderRadius: 8, padding: "7px 10px" }}>
                          <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
                            <div style={{ width: 6, height: 6, borderRadius: "50%", background: f.sev === "high" ? "#ef4444" : f.sev === "med" ? "#f97316" : "#fbbf24" }} />
                            <span style={{ fontSize: 12, color: "#e5e7eb" }}>{f.label}</span>
                          </div>
                          <span style={{ fontSize: 12, fontWeight: 700, color: "#06b6d4" }}>+{f.pts} pts</span>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div style={{ textAlign: "center", padding: "12px", background: "#052e16", borderRadius: 10, color: "#34d399", fontSize: 13, fontWeight: 600 }}>
                    ✓ No major risk factors detected
                  </div>
                )}
              </Card>

              <Card>
                <SectionTitle>Dataset Benchmarks</SectionTitle>
                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                  {[
                    { label: "Your BMI vs Dataset Avg", you: form.bmi, avg: 29.4, unit: "kg/m²" },
                    { label: "Your SBP vs Dataset Avg", you: form.sbp, avg: 125.1, unit: "mmHg" },
                    { label: "Your Age vs Dataset Avg", you: form.age, avg: 48.1, unit: "yrs" },
                  ].map((b, i) => (
                    <div key={i} style={{ background: "#111827", borderRadius: 8, padding: "10px 12px" }}>
                      <div style={{ fontSize: 10, color: "#6b7280", marginBottom: 6 }}>{b.label}</div>
                      <div style={{ display: "flex", gap: 16 }}>
                        <div><div style={{ fontSize: 15, fontWeight: 800, color: "#06b6d4" }}>{b.you}</div><div style={{ fontSize: 10, color: "#4b5563" }}>You</div></div>
                        <div><div style={{ fontSize: 15, fontWeight: 800, color: "#9ca3af" }}>{b.avg}</div><div style={{ fontSize: 10, color: "#4b5563" }}>Avg</div></div>
                        <div style={{ flex: 1, display: "flex", alignItems: "center" }}>
                          <span style={{ fontSize: 11, color: b.you > b.avg ? "#ef4444" : "#34d399", fontWeight: 600 }}>
                            {b.you > b.avg ? `↑ ${(b.you - b.avg).toFixed(1)} above avg` : b.you < b.avg ? `↓ ${(b.avg - b.you).toFixed(1)} below avg` : "= At avg"}
                          </span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </Card>
            </div>
          </div>
        )}

        {/* CORRELATIONS TAB */}
        {tab === "correlations" && (
          <div className="fade-in" style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
              <Card>
                <SectionTitle>Key Variable Correlations</SectionTitle>
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={CORR_DATA} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                    <XAxis type="number" domain={[0, 1]} tick={{ fill: "#6b7280", fontSize: 11 }} />
                    <YAxis type="category" dataKey="subject" tick={{ fill: "#9ca3af", fontSize: 11 }} width={90} />
                    <Tooltip {...TOOLTIP_STYLE} />
                    <Bar dataKey="value" radius={[0, 6, 6, 0]}>
                      {CORR_DATA.map((d, i) => (
                        <Cell key={i} fill={d.value >= 0.7 ? "#ef4444" : d.value >= 0.4 ? "#f97316" : "#06b6d4"} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
                <div style={{ display: "flex", gap: 8, marginTop: 6 }}>
                  {[["#ef4444","Strong (≥0.7)"],["#f97316","Moderate (0.4–0.7)"],["#06b6d4","Weak (<0.4)"]].map(([c,l],i) => (
                    <div key={i} style={{ display: "flex", alignItems: "center", gap: 4 }}>
                      <div style={{ width: 8, height: 8, borderRadius: 2, background: c }} />
                      <span style={{ fontSize: 9, color: "#9ca3af" }}>{l}</span>
                    </div>
                  ))}
                </div>
              </Card>

              <Card>
                <SectionTitle>Key Findings</SectionTitle>
                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                  {[
                    { icon: "🔴", title: "BMI ↔ Waist Circumference", val: "r = 0.91", desc: "Nearly perfect correlation — waist is a strong proxy for BMI.", color: "#ef4444" },
                    { icon: "🔴", title: "BMI ↔ Arm Circumference", val: "r = 0.87", desc: "Arm circumference closely tracks overall body fat.", color: "#ef4444" },
                    { icon: "🟠", title: "Age ↔ Systolic BP", val: "r = 0.46", desc: "Aging is a significant predictor of rising blood pressure.", color: "#f97316" },
                    { icon: "🟠", title: "Waist ↔ Arm Circ.", val: "r = 0.81", desc: "Both anthropometric measures are highly linked.", color: "#f97316" },
                    { icon: "🔵", title: "BMI ↔ Systolic BP", val: "r = 0.15", desc: "Weak but notable — higher BMI slightly raises BP.", color: "#06b6d4" },
                  ].map((f, i) => (
                    <div key={i} style={{ display: "flex", gap: 10, background: "#111827", borderRadius: 8, padding: "9px 12px", borderLeft: `3px solid ${f.color}` }}>
                      <span style={{ fontSize: 16 }}>{f.icon}</span>
                      <div>
                        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                          <span style={{ fontSize: 12, fontWeight: 700, color: "#e5e7eb" }}>{f.title}</span>
                          <span style={{ fontSize: 11, color: f.color, fontWeight: 700 }}>{f.val}</span>
                        </div>
                        <div style={{ fontSize: 11, color: "#6b7280", marginTop: 1 }}>{f.desc}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </Card>
            </div>

            <Card>
              <SectionTitle>Disease Risk Radar — Dataset Profile</SectionTitle>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, alignItems: "center" }}>
                <ResponsiveContainer width="100%" height={240}>
                  <RadarChart data={[
                    { factor: "Hypertension", value: 38.9 },
                    { factor: "Obesity", value: 39.2 },
                    { factor: "Smoking", value: 59.4 },
                    { factor: "Overweight", value: 71.2 },
                    { factor: "Elevated BP", value: 52.1 },
                  ]}>
                    <PolarGrid stroke="#1f2937" />
                    <PolarAngleAxis dataKey="factor" tick={{ fill: "#9ca3af", fontSize: 11 }} />
                    <Radar name="Prevalence %" dataKey="value" stroke="#06b6d4" fill="#06b6d4" fillOpacity={0.25} />
                    <Tooltip {...TOOLTIP_STYLE} formatter={(v) => [`${v}%`, "Prevalence"]} />
                  </RadarChart>
                </ResponsiveContainer>
                <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                  <div style={{ fontSize: 13, color: "#9ca3af", lineHeight: 1.6 }}>
                    <p style={{ color: "#e5e7eb", fontWeight: 700, marginBottom: 6 }}>📊 Population-Level Insights</p>
                    <ul style={{ listStyle: "none", display: "flex", flexDirection: "column", gap: 6 }}>
                      {[
                        "59.4% of participants are smokers — the single largest risk factor in this dataset.",
                        "71.2% are overweight or obese (BMI ≥ 25), indicating a widespread weight management challenge.",
                        "Hypertension affects 38.9% — rising sharply after age 45.",
                        "Age is the strongest single predictor of high systolic blood pressure (r = 0.46).",
                      ].map((t, i) => (
                        <li key={i} style={{ fontSize: 11, color: "#9ca3af", paddingLeft: 12, borderLeft: "2px solid #1f2937" }}>{t}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
            </Card>
          </div>
        )}
      </div>
    </div>
  );
}

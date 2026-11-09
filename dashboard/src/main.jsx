import React from 'react';
import { createRoot } from 'react-dom/client';
import {
  LineChart, Line, XAxis, YAxis,
  Tooltip, CartesianGrid, ResponsiveContainer
} from 'recharts';
import './style.css';

const demo = [
  { round: 1, loss: 1.08, dice: 0.31 },
  { round: 2, loss: 0.82, dice: 0.43 },
  { round: 3, loss: 0.66, dice: 0.52 }
];

function App() {
  const [metrics, setMetrics] = React.useState(demo);
  const [status, setStatus] = React.useState('Connecting...');
  const [source, setSource] = React.useState('demo');

  React.useEffect(() => {
    fetch('http://localhost:8000/api/metrics')
      .then(response => {
        if (!response.ok) throw new Error('API request failed');
        return response.json();
      })
      .then(data => {
        if (data.history?.length) {
          setMetrics(data.history);
        }

        setSource(data.source || 'unknown');

        setStatus(
          data.source === 'illustrative_demo_data'
            ? 'API connected — illustrative demo data'
            : 'API connected — training metrics received'
        );
      })
      .catch(error => {
        console.error('Metrics API error:', error);
        setStatus('Backend connection unavailable');
      });
  }, []);

  const latest = metrics.at(-1);

  return (
    <main>
      <header>
        <div>
          <h1>FedMed</h1>
          <p>Cross-Silo Federated Learning Monitor</p>
        </div>
        <span>{status}</span>
      </header>

      <section className="cards">
        <article>
          <small>Participating hospitals</small>
          <b>3</b>
        </article>

        <article>
          <small>Federated rounds</small>
          <b>{metrics.length}</b>
        </article>

        <article>
          <small>Latest training loss</small>
          <b>{latest?.loss?.toFixed(3) ?? '—'}</b>
        </article>

        <article>
          <small>Latest Dice</small>
          <b>{latest?.dice?.toFixed(3) ?? '—'}</b>
        </article>
      </section>

      <section className="panel">
        <h2>Global training convergence</h2>

        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={metrics}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="round" />
            <YAxis />
            <Tooltip />
            <Line
              type="monotone"
              dataKey="loss"
              stroke="#2563eb"
              name="Loss"
            />
            <Line
              type="monotone"
              dataKey="dice"
              stroke="#16a34a"
              name="Dice"
            />
          </LineChart>
        </ResponsiveContainer>

        <p className="note">
          {source === 'illustrative_demo_data'
            ? 'Illustrative demo values — not measured training results.'
            : 'Metrics received from the backend API.'}
        </p>
      </section>
    </main>
  );
}

createRoot(document.getElementById('root')).render(<App />);
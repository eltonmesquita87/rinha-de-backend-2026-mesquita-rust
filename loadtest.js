import http from 'k6/http';
import { check } from 'k6';

export const options = {
  vus: 50,
  duration: '15s',
  thresholds: {
    http_req_failed: ['rate<0.01'],
    http_req_duration: ['p(99)<2000'],
  },
};

const payload = JSON.stringify({
  transaction: { amount: 384.88, installments: 3, requested_at: '2026-03-11T20:23:35Z' },
  customer: { avg_amount: 769.76, tx_count_24h: 3, known_merchants: ['MERC-009', 'MERC-001', 'MERC-002'] },
  merchant: { id: 'MERC-001', mcc: '5912', avg_amount: 298.95 },
  terminal: { is_online: false, card_present: true, km_from_home: 13.709052 },
  last_transaction: { timestamp: '2026-03-11T14:58:35Z', km_from_current: 18.862648 },
});

const params = { headers: { 'Content-Type': 'application/json' } };

export default function () {
  const res = http.post('http://localhost:9999/fraud-score', payload, params);
  check(res, { 'status 200': (r) => r.status === 200 });
}

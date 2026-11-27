# Session 6 — hardening, calibration and monitored deployment

Session 5 includes a local FastAPI prototype, Streamlit reviewer interface,
SQLite feedback store, Docker image and offline API tests.

Before production deployment:
1. Replace developer API keys with managed secrets and proper identity/role auth.
2. Add migrations, persistent database encryption and reviewer audit policies.
3. Implement signed, versioned, trusted model artifact distribution.
4. Calibrate scores on an independent validation partition and define review bands.
5. Add request rate limiting, structured logging, observability and p95 latency checks.
6. Benchmark with a fully judged query/catalog retrieval dataset.
7. Add a hardened deployment configuration and separate environment approvals.
8. Never expose the development UI or model joblib loader to untrusted users.

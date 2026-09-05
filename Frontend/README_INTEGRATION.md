# BBB Predictor frontend

This is the original BBB Predictor frontend with its existing UI preserved.
Its API client is connected to the BBB backend included in the companion
`bbb_backend` archive.

## Connection behavior

- By default, browser requests use `/api`, which is the recommended setup when
  the frontend and backend are served behind the same host or reverse proxy.
- For a frontend hosted on a different domain, set `VITE_BBB_API_URL` before
  building the frontend. See `.env.example`.
- During Vite development, `/api` is proxied to `BBB_API_URL`, which defaults
  to `http://localhost:8000`.

## Run

```bash
npm install
npm run dev
```

Build for production with:

```bash
npm run build
```

The frontend uses the backend endpoints for single prediction, batch
prediction, file upload, model information, health, and SMILES validation.
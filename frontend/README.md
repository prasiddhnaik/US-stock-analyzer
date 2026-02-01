# Stock Analyzer - Windows XP Luna Blue Edition

A React + TypeScript frontend with authentic Windows XP Luna Blue styling for the Stock Analyzer application.

## Features

- 🎨 Authentic Windows XP Luna Blue theme
- 📊 ML-powered stock analysis with interactive charts
- 📈 Technical indicators: RSI, MACD, Bollinger Bands, SMAs
- 🔮 5-day price direction predictions
- ⚡ Fast, responsive UI with Plotly charts

## Quick Start

### 1. Start the Backend Server

From the project root directory:

```bash
# Install Python dependencies (if not already done)
pip install -r requirements.txt

# Start the FastAPI server
python api_server.py
```

The API will be available at `http://localhost:8000`

### 2. Start the Frontend

```bash
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

The frontend will be available at `http://localhost:5173`

## Tech Stack

| Layer | Technology |
|-------|------------|
| Framework | React 18 |
| Language | TypeScript |
| Build Tool | Vite |
| Styling | XP.css + Custom Luna Blue CSS |
| Charts | Plotly.js |
| API | FastAPI (Python backend) |

## XP Components

This project includes custom Windows XP-styled React components:

- `XPButton` - Command buttons with XP styling
- `XPInput` - Text input fields
- `XPPanel` - Group boxes / panels
- `XPTabs` - Tab navigation
- `XPProgress` - Progress bar with animated stripes
- `XPCard` - Data display cards
- `XPLoading` - Loading spinner with XP aesthetics

## Luna Blue Color Palette

The theme uses authentic Windows XP Luna Blue colors:

| Element | Color |
|---------|-------|
| Title Bar Start | `#0A246A` |
| Title Bar End | `#A6CAF0` |
| Window Background | `#ECE9D8` |
| Selection Blue | `#316AC5` |
| Button Face | `#ECE9D8` |
| Start Green | `#3C8A3C` |

## Project Structure

```
frontend/
├── src/
│   ├── App.tsx              # Main application
│   ├── main.tsx             # Entry point
│   ├── index.css            # Luna Blue theme CSS
│   ├── api/
│   │   └── stockApi.ts      # API client
│   └── components/
│       ├── XPButton.tsx
│       ├── XPInput.tsx
│       ├── XPPanel.tsx
│       ├── XPTabs.tsx
│       ├── XPProgress.tsx
│       ├── XPCard.tsx
│       └── XPLoading.tsx
├── package.json
├── tsconfig.json
└── vite.config.ts
```

## Development

```bash
# Run development server with hot reload
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## Notes

- The frontend proxies API requests to `http://localhost:8000`
- Make sure the FastAPI backend is running before using the app
- Requires Alpaca API credentials configured in the backend `.env` file


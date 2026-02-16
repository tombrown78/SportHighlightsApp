# Frontend Development Rules

## Project Structure

```
frontend/
├── src/
│   └── app/              # Next.js App Router
│       ├── page.tsx      # Home page
│       ├── analyze/      # Video analysis pages
│       └── globals.css   # Global styles
├── public/               # Static assets
└── package.json          # Dependencies
```

## TypeScript Interfaces

### Syncing with Backend

When backend schemas change, update frontend interfaces:

```typescript
// Must match backend/app/models/schemas.py

interface Player {
  id: string;
  video_id: string;
  jersey_number: string;
  team: string;
  team_color?: string;  // New field - add when backend adds it
  confidence: number;
  segment_count: number;
  action_count: number;
}
```

## Styling

### Use Tailwind CSS

```tsx
// Good - Tailwind classes
<div className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-sm">

// Avoid - Inline styles (except for dynamic values)
<div style={{ backgroundColor: 'white' }}>

// OK - Dynamic values that can't be Tailwind classes
<div 
  className="w-3 h-3 rounded-full"
  style={{ backgroundColor: player.team_color }}
/>
```

## API Calls

### Pattern for Data Fetching

```typescript
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const fetchData = async () => {
  try {
    const response = await fetch(`${API_URL}/api/endpoint`);
    if (!response.ok) throw new Error('Request failed');
    const data = await response.json();
    setData(data);
  } catch (err) {
    console.error('Failed to fetch:', err);
    setError(err instanceof Error ? err.message : 'Unknown error');
  }
};
```

## State Management

### Use React Hooks

```typescript
// Local state
const [data, setData] = useState<DataType[]>([]);
const [loading, setLoading] = useState(true);
const [error, setError] = useState<string | null>(null);

// Refs for DOM elements
const videoRef = useRef<HTMLVideoElement>(null);

// Effects for data fetching
useEffect(() => {
  fetchData();
}, [dependency]);
```

## Components

### Icon Imports

Use lucide-react for icons:

```typescript
import {
  Play,
  Pause,
  Download,
  Trash2,
  // Add new icons as needed
} from 'lucide-react';
```

### Accessibility

Always include accessibility attributes:

```tsx
<button
  onClick={handleClick}
  title="Button description"  // Required for icon-only buttons
  aria-label="Accessible label"
>
  <Icon className="w-4 h-4" />
</button>
```

## Environment Variables

Frontend env vars must be prefixed with `NEXT_PUBLIC_`:

```bash
# .env.local
NEXT_PUBLIC_API_URL=http://localhost:8000
```

Access in code:
```typescript
const apiUrl = process.env.NEXT_PUBLIC_API_URL;
```

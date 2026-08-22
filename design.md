# AI Root Cause Analyzer Design System

This document defines the creative, new-age design system for the AI Root Cause Analyzer (RCA Engine) dashboard. It focuses on a high-fidelity, futuristic cyber-glassmorphic aesthetic.

## 🎨 Theme & Visual Philosophy
The user interface uses a **Deep Space Cyberpunk Dark Mode**. The layout feels premium, highly interactive, and alive through neon ambient glows, micro-animations, and semi-transparent glassmorphism layers.

### Design Tokens

#### 1. Color Palette
- **Primary / Accent Blue**: `#3b82f6` (Electric Blue) - Represents analytical signals and core system actions.
- **Secondary / Accent Purple**: `#8b5cf6` (Cyber Purple) - Represents AI & ML intelligence components.
- **Tertiary / Accent Cyan**: `#06b6d4` (Neon Cyan) - Represents performance metrics and data drift.
- **Success / Healthy**: `#10b981` (Emerald Green) - Normal operations, healthy signals, solved issues.
- **Warning / Moderate Drift**: `#f59e0b` (Amber Orange) - Minor drift detected, warning threshold exceeded.
- **Danger / High Severity**: `#ef4444` (Pulse Rose) - High-severity anomalies, system failures.
- **Backgrounds**:
  - Main Background: `#060813` (Ultra Deep Blue)
  - Card/Container Background: `rgba(15, 23, 42, 0.7)` (Frosted Slate Slate-900 at 70% opacity)
  - Glass Highlights: `rgba(255, 255, 255, 0.03)` (Frosted highlights)
- **Borders**:
  - Glass Border: `rgba(255, 255, 255, 0.08)`
  - Hover Glass Border: `rgba(255, 255, 255, 0.18)`

#### 2. Typography
- **Headline Font**: `Space Grotesk` (Modern, futuristic sans-serif with geometric curves)
- **Body Font**: `Inter` (Extremely legible sans-serif for descriptions and tabular content)
- **Monospace Font**: `JetBrains Mono` (For data metrics, latency numbers, and reasoning chain steps)

#### 3. Shapes & Roundness
- **Card Corners**: `16px` (Rounded corners)
- **Buttons / Input Corners**: `12px`
- **Badges / Pill Corners**: `9999px` (Fully rounded)

#### 4. Shadows & Glowing Effects
- **Glow Blue**: `0 0 20px rgba(59, 130, 246, 0.25)`
- **Glow Purple**: `0 0 20px rgba(139, 92, 246, 0.25)`
- **Glow Danger**: `0 0 25px rgba(239, 68, 68, 0.35)`

---

## 🎛️ Dynamic Components & Layouts

### 1. The Glass-Shell Layout
The application uses a grid shell with a sticky sidebar on the left and a scrollable content area on the right.
- **Sidebar**: High-contrast, absolute dark overlay, with a glowing border separating it from the main content. Nav items slide and glow on hover.
- **Header**: Includes status dots that pulse slowly, showing connection health.

### 2. Stats & Metric Cards
- Cards feature top border glowing accents matching the status of the metric (Blue for performance, Emerald for healthy, Amber/Red for drift anomalies).
- Big numbers are styled using linear gradients.

### 3. Simulator Control Panel
- Input fields and dropdowns are styled with absolute minimal backgrounds, transitioning borders to cyan or purple when focused.
- Sliders use a custom neon blue track with a purple thumb.

### 4. Interactive RCA detail nodes
- Reasoning chain steps are rendered as connected nodes. Completed steps glow emerald; steps with high anomaly drift glow rose.

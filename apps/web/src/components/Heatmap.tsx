import { useEffect, useRef } from "react";

interface HeatmapProps {
  data: Float64Array;
  rows: number;
  cols: number;
  cellSize?: number;
}

function diverging(value: number, absMax: number): [number, number, number] {
  if (absMax === 0) return [255, 255, 255];
  const t = Math.max(-1, Math.min(1, value / absMax));
  if (t >= 0) {
    // white -> red
    const r = 255;
    const g = Math.round(255 * (1 - t));
    const b = Math.round(255 * (1 - t));
    return [r, g, b];
  }
  // white -> blue
  const s = -t;
  const r = Math.round(255 * (1 - s));
  const g = Math.round(255 * (1 - s));
  const b = 255;
  return [r, g, b];
}

export function Heatmap({ data, rows, cols, cellSize }: HeatmapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const size = cellSize ?? Math.max(2, Math.min(24, Math.floor(480 / Math.max(rows, cols))));

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = cols * size;
    canvas.height = rows * size;

    let absMax = 0;
    for (let i = 0; i < data.length; i++) absMax = Math.max(absMax, Math.abs(data[i]));

    const img = ctx.createImageData(canvas.width, canvas.height);
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const v = data[r * cols + c];
        const [red, green, blue] = diverging(v, absMax);
        for (let py = 0; py < size; py++) {
          for (let px = 0; px < size; px++) {
            const x = c * size + px;
            const y = r * size + py;
            const idx = (y * canvas.width + x) * 4;
            img.data[idx] = red;
            img.data[idx + 1] = green;
            img.data[idx + 2] = blue;
            img.data[idx + 3] = 255;
          }
        }
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [data, rows, cols, size]);

  return <canvas ref={canvasRef} className="heatmap-canvas" title="Diverging color scale: blue = negative, red = positive, white = ~0" />;
}

import React, { useEffect, useRef } from 'react';

// Define the props we expect to receive from the main App
interface WaveformCanvasProps {
  buffers: {
    ecgBuffer: React.MutableRefObject<number[]>;
    abpBuffer: React.MutableRefObject<number[]>;
    respBuffer: React.MutableRefObject<number[]>;
  };
  heartRate: number;
}

export const WaveformCanvas: React.FC<WaveformCanvasProps> = ({ buffers, heartRate }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    let animationId: number;

    const draw = () => {
      const canvas = canvasRef.current;
      const ctx = canvas?.getContext('2d');

      if (ctx && canvas) {
        // Clear the canvas for the next frame
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Helper function to draw a single waveform line
        const drawLine = (buf: number[], color: string, vOffset: number, scale: number) => {
          ctx.beginPath();
          ctx.strokeStyle = color;
          ctx.lineWidth = 1.5;
          
          // We limit the buffer to 250 points in the hook, so we scale the x-axis accordingly
          const step = canvas.width / 250; 

          for (let i = 0; i < buf.length; i++) {
            const x = i * step;
            const y = vOffset - buf[i] * scale;
            
            if (i === 0) {
              ctx.moveTo(x, y);
            } else {
              ctx.lineTo(x, y);
            }
          }
          ctx.stroke();
        };

        // Draw ECG (Green), ABP (Cyan), and RESP (Purple)
        drawLine(buffers.ecgBuffer.current, '#34d399', 40, 25);
        drawLine(buffers.abpBuffer.current, '#22d3ee', 110, 1.2);
        drawLine(buffers.respBuffer.current, '#c084fc', 165, 15);
      }

      // Loop the drawing function on the browser's animation frame
      animationId = requestAnimationFrame(draw);
    };

    // Start the drawing loop
    draw();

    // Clean up the animation frame when the component unmounts
    return () => cancelAnimationFrame(animationId);
  }, [buffers]);

  return (
    <div className="bg-black/80 border border-slate-800 rounded-xl p-3 relative h-48 overflow-hidden w-full">
      <canvas 
        ref={canvasRef} 
        width={400} 
        height={180} 
        className="absolute inset-0 w-full h-full" 
      />
      {/* Overlay text for the canvas */}
      <div className="relative flex justify-between text-[10px] font-mono text-emerald-400 z-10 px-2 pointer-events-none">
        <span>ECG (mV)</span>
        <span>HR: {heartRate} bpm</span>
      </div>
    </div>
  );
};
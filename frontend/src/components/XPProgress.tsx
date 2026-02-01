import React from 'react';

interface XPProgressProps {
  value?: number; // 0-100
  indeterminate?: boolean;
  className?: string;
  style?: React.CSSProperties;
}

export const XPProgress: React.FC<XPProgressProps> = ({
  value = 0,
  indeterminate = false,
  className = '',
  style,
}) => {
  const progressStyle: React.CSSProperties = indeterminate
    ? { width: '100%' }
    : { width: `${Math.min(100, Math.max(0, value))}%` };

  return (
    <div className={`xp-progress ${className}`} style={style}>
      <div 
        className="xp-progress-bar"
        style={progressStyle}
      />
    </div>
  );
};

export default XPProgress;


import React from 'react';

interface XPInputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
}

export const XPInput: React.FC<XPInputProps> = ({
  label,
  className = '',
  id,
  ...props
}) => {
  const inputId = id || `xp-input-${Math.random().toString(36).substr(2, 9)}`;
  
  return (
    <div className="xp-input-wrapper" style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
      {label && (
        <label htmlFor={inputId} style={{ fontSize: '12px', color: 'var(--xp-text-primary)' }}>
          {label}
        </label>
      )}
      <input
        id={inputId}
        className={`xp-input ${className}`}
        {...props}
      />
    </div>
  );
};

export default XPInput;


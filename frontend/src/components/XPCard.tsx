import React from 'react';

interface XPCardProps {
  title: string;
  value: string | number;
  variant?: 'default' | 'success' | 'danger' | 'warning';
  className?: string;
}

export const XPCard: React.FC<XPCardProps> = ({
  title,
  value,
  variant = 'default',
  className = '',
}) => {
  const valueClass = variant !== 'default' ? `xp-card-value--${variant}` : '';

  return (
    <div className={`xp-card ${className}`}>
      <div className="xp-card-header">{title}</div>
      <div className={`xp-card-value ${valueClass}`}>{value}</div>
    </div>
  );
};

export default XPCard;


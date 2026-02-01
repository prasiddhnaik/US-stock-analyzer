import React from 'react';

interface XPPanelProps {
  title?: string;
  children: React.ReactNode;
  className?: string;
}

export const XPPanel: React.FC<XPPanelProps> = ({
  title,
  children,
  className = '',
}) => {
  return (
    <div className={`xp-panel ${className}`}>
      {title && <span className="xp-panel-title">{title}</span>}
      {children}
    </div>
  );
};

export default XPPanel;


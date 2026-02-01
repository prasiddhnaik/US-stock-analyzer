import React from 'react';
import XPProgress from './XPProgress';

interface XPLoadingProps {
  message?: string;
}

export const XPLoading: React.FC<XPLoadingProps> = ({
  message = 'Loading...',
}) => {
  return (
    <div className="xp-loading">
      <div className="xp-loading-spinner" />
      <div className="xp-loading-text">{message}</div>
      <XPProgress indeterminate style={{ width: '200px' }} />
    </div>
  );
};

export default XPLoading;


import React from 'react';

interface XPButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'default' | 'primary';
  children: React.ReactNode;
}

export const XPButton: React.FC<XPButtonProps> = ({
  variant = 'default',
  children,
  className = '',
  disabled,
  ...props
}) => {
  const classes = [
    'xp-button',
    variant === 'primary' ? 'xp-button--primary' : '',
    className,
  ].filter(Boolean).join(' ');

  return (
    <button className={classes} disabled={disabled} {...props}>
      {children}
    </button>
  );
};

export default XPButton;


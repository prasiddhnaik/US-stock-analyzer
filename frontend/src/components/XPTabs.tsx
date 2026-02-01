import React, { useState } from 'react';

interface Tab {
  id: string;
  label: string;
  content: React.ReactNode;
}

interface XPTabsProps {
  tabs: Tab[];
  defaultTab?: string;
  className?: string;
}

export const XPTabs: React.FC<XPTabsProps> = ({
  tabs,
  defaultTab,
  className = '',
}) => {
  const [activeTab, setActiveTab] = useState(defaultTab || tabs[0]?.id);

  const activeContent = tabs.find(tab => tab.id === activeTab)?.content;

  return (
    <div className={className}>
      <div className="xp-tabs">
        {tabs.map(tab => (
          <button
            key={tab.id}
            className={`xp-tab ${activeTab === tab.id ? 'xp-tab--active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>
      <div className="xp-tab-content">
        {activeContent}
      </div>
    </div>
  );
};

export default XPTabs;


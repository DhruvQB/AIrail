import { useAuth } from '../context/AuthContext';

export default function Sidebar({
  chats,
  activeChat,
  onSelectChat,
  onNewChat,
  searchQuery,
  onSearchChange,
}) {
  const { user, logout } = useAuth();

  const filtered = chats.filter((c) =>
    c.title.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="sidebar">
      {/* Header */}
      <div className="sidebar-header">
        <div className="brand">
          <span className="brand-icon">🚆</span>
          <span>AIrail</span>
        </div>
      </div>

      {/* Search */}
      <div className="sidebar-search">
        <span className="search-icon">🔍</span>
        <input
          type="text"
          placeholder="Search chats…"
          value={searchQuery}
          onChange={(e) => onSearchChange(e.target.value)}
        />
      </div>

      {/* Chat list */}
      <div className="sidebar-section">
        <div className="sidebar-section-title">Chats</div>
      </div>
      <div className="chat-list">
        {filtered.length === 0 && (
          <div style={{ padding: '12px 16px', color: 'var(--text-muted)', fontSize: '0.8rem' }}>
            No chats yet. Start a new conversation!
          </div>
        )}
        {filtered.map((chat) => (
          <div
            key={chat.id}
            className={`chat-item ${activeChat === chat.id ? 'active' : ''}`}
            onClick={() => onSelectChat(chat.id)}
          >
            <span className="chat-item-icon">💬</span>
            <div className="chat-item-text">
              <div className="chat-item-title">{chat.title}</div>
              {chat.preview && (
                <div className="chat-item-preview">{chat.preview}</div>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* New Chat Button */}
      <div className="sidebar-footer">
        <button className="new-chat-btn" onClick={onNewChat}>
          <span>+</span> New chat
        </button>
      </div>

      {/* User Menu */}
      {user && (
        <div className="user-menu">
          <div className="user-avatar">
            {user.name?.charAt(0)?.toUpperCase() || 'U'}
          </div>
          <div className="user-info">
            <div className="user-name">{user.name}</div>
            <div className="user-email">{user.email}</div>
          </div>
          <button className="logout-btn" onClick={logout} title="Logout">
            ⏻
          </button>
        </div>
      )}
    </div>
  );
}

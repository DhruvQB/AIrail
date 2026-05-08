import { useState } from 'react';

export default function ChatInput({ onSend, disabled }) {
  const [text, setText] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    const trimmed = text.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setText('');
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      handleSubmit(e);
    }
  };

  return (
    <div className="chat-input-container">
      <div className="chat-input-wrapper">
        <form onSubmit={handleSubmit} className="chat-input-box">
          <input
            id="chat-input"
            type="text"
            placeholder="Ask about trains, PNR, policies…"
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={disabled}
            autoComplete="off"
          />
          <button
            type="submit"
            className="send-btn"
            disabled={!text.trim() || disabled}
            title="Send"
          >
            →
          </button>
        </form>
        <div className="chat-disclaimer">
          AIrail can make mistakes. Verify important railway information.
        </div>
      </div>
    </div>
  );
}

import { useRef, useEffect } from 'react';
import MessageBubble from './MessageBubble';
import FeatureCards from './FeatureCards';

export default function ChatArea({ messages, isLoading, onFeatureClick }) {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const isEmpty = messages.length === 0;

  return (
    <div className="messages-area">
      {isEmpty ? (
        <div className="welcome-screen">
          <div className="welcome-icon">🚆</div>
          <h2 className="welcome-title">How can I help you today?</h2>
          <p className="welcome-subtitle">
            Your AI-powered Indian Railways assistant. Ask about live train
            status, PNR bookings, schedules, and railway policies.
          </p>
          <FeatureCards onCardClick={onFeatureClick} />
        </div>
      ) : (
        <>
          {messages.map((msg, i) => (
            <MessageBubble key={i} role={msg.role} content={msg.content} />
          ))}
          {isLoading && (
            <div className="message">
              <div className="message-avatar assistant">🚆</div>
              <div className="message-content">
                <div className="message-role">AIrail</div>
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            </div>
          )}
        </>
      )}
      <div ref={bottomRef} />
    </div>
  );
}

import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

export default function MessageBubble({ role, content }) {
  const isUser = role === 'user';

  return (
    <div className="message">
      <div className={`message-avatar ${isUser ? 'user' : 'assistant'}`}>
        {isUser ? '👤' : '🚆'}
      </div>
      <div className="message-content">
        <div className="message-role">{isUser ? 'You' : 'AIrail'}</div>
        <div className="message-text">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
        </div>
      </div>
    </div>
  );
}

import { useState, useCallback, useEffect } from 'react';
import Sidebar from '../components/Sidebar';
import ChatArea from '../components/ChatArea';
import ChatInput from '../components/ChatInput';
import api from '../api';

function generateId() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 10);
}

function loadChats() {
  try {
    return JSON.parse(localStorage.getItem('airail_chats') || '[]');
  } catch {
    return [];
  }
}

function saveChats(chats) {
  localStorage.setItem('airail_chats', JSON.stringify(chats));
}

export default function ChatPage() {
  const [chats, setChats] = useState(loadChats);
  const [activeChatId, setActiveChatId] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  // Persist chats
  useEffect(() => {
    saveChats(chats);
  }, [chats]);

  const activeChat = chats.find((c) => c.id === activeChatId);
  const messages = activeChat?.messages || [];

  const createNewChat = useCallback(() => {
    const newChat = {
      id: generateId(),
      sessionId: generateId() + '-' + generateId(),
      title: 'New Chat',
      preview: '',
      messages: [],
      createdAt: Date.now(),
    };
    setChats((prev) => [newChat, ...prev]);
    setActiveChatId(newChat.id);
    return newChat;
  }, []);

  const sendMessage = useCallback(
    async (text) => {
      let chat = activeChat;
      if (!chat) {
        chat = {
          id: generateId(),
          sessionId: generateId() + '-' + generateId(),
          title: 'New Chat',
          preview: '',
          messages: [],
          createdAt: Date.now(),
        };
        setChats((prev) => [chat, ...prev]);
        setActiveChatId(chat.id);
      }

      const userMsg = { role: 'user', content: text };
      const updatedTitle =
        chat.messages.length === 0 ? text.slice(0, 40) : chat.title;

      setChats((prev) =>
        prev.map((c) =>
          c.id === chat.id
            ? {
                ...c,
                messages: [...c.messages, userMsg],
                title: updatedTitle,
                preview: text.slice(0, 60),
              }
            : c
        )
      );

      setIsLoading(true);

      try {
        const res = await api.post('/chat', {
          session_id: chat.sessionId,
          message: text,
        });

        const assistantMsg = {
          role: 'assistant',
          content: res.data.response,
        };

        setChats((prev) =>
          prev.map((c) =>
            c.id === chat.id
              ? {
                  ...c,
                  messages: [...c.messages, assistantMsg],
                  preview: res.data.response.slice(0, 60),
                }
              : c
          )
        );
      } catch (err) {
        const errMsg = {
          role: 'assistant',
          content: `⚠️ Error: ${err.response?.data?.detail || err.message || 'Something went wrong.'}`,
        };

        setChats((prev) =>
          prev.map((c) =>
            c.id === chat.id
              ? { ...c, messages: [...c.messages, errMsg] }
              : c
          )
        );
      } finally {
        setIsLoading(false);
      }
    },
    [activeChat]
  );

  const handleFeatureClick = useCallback(
    (prompt) => {
      sendMessage(prompt);
    },
    [sendMessage]
  );

  return (
    <div className="chat-layout">
      <Sidebar
        chats={chats}
        activeChat={activeChatId}
        onSelectChat={setActiveChatId}
        onNewChat={createNewChat}
        searchQuery={searchQuery}
        onSearchChange={setSearchQuery}
      />

      <div className="chat-main">
        <div className="chat-topbar">
          <div className="chat-topbar-left">
            <span className="chat-topbar-title">
              {activeChat?.title || 'AIrail Chat'}
            </span>
            <span className="chat-topbar-badge">AI Assistant</span>
          </div>
          <div className="chat-topbar-right">
            {/* placeholder for future icons */}
          </div>
        </div>

        <ChatArea
          messages={messages}
          isLoading={isLoading}
          onFeatureClick={handleFeatureClick}
        />

        <ChatInput onSend={sendMessage} disabled={isLoading} />
      </div>
    </div>
  );
}

import { useState, useCallback, useEffect, useRef } from 'react';
import Sidebar from '../components/Sidebar';
import ChatArea from '../components/ChatArea';
import ChatInput from '../components/ChatInput';
import api from '../api';

function generateId() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 10);
}

export default function ChatPage() {
  const [chats, setChats] = useState([]);
  const [activeChatId, setActiveChatId] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [historyLoaded, setHistoryLoaded] = useState(false);
  const loadedSessionIds = useRef(new Set()); // track which sessions have been fully loaded

  const activeChat = chats.find((c) => c.id === activeChatId);
  const messages = activeChat?.messages || [];

  // ---------------------------------------------------------------------------
  // Load session list from backend on mount
  // ---------------------------------------------------------------------------
  useEffect(() => {
    const fetchSessions = async () => {
      try {
        const res = await api.get('/chat/sessions');
        const sessions = res.data.sessions || [];
        const chatList = sessions.map((s) => ({
          id: s.sessionId,
          sessionId: s.sessionId,
          title: s.title || 'New Chat',
          preview: s.preview || '',
          messages: [], // lazy-loaded when selected
          createdAt: new Date(s.createdAt).getTime(),
        }));
        setChats(chatList);
        if (chatList.length > 0) {
          setActiveChatId(chatList[0].id);
        }
      } catch (err) {
        console.error('Failed to load sessions:', err);
      } finally {
        setHistoryLoaded(true);
      }
    };
    fetchSessions();
  }, []);

  // ---------------------------------------------------------------------------
  // Lazy-load messages when a session is selected
  // ---------------------------------------------------------------------------
  useEffect(() => {
    if (!activeChatId) return;
    if (loadedSessionIds.current.has(activeChatId)) return; // already loaded

    const fetchMessages = async () => {
      try {
        const res = await api.get(`/chat/sessions/${activeChatId}/messages`);
        const msgs = res.data.messages || [];
        loadedSessionIds.current.add(activeChatId);
        setChats((prev) =>
          prev.map((c) =>
            c.id === activeChatId
              ? { ...c, messages: msgs.map((m) => ({ role: m.role, content: m.content })) }
              : c
          )
        );
      } catch (err) {
        console.error('Failed to load messages:', err);
      }
    };
    fetchMessages();
  }, [activeChatId]);

  // ---------------------------------------------------------------------------
  // Create a new chat session
  // ---------------------------------------------------------------------------
  const createNewChat = useCallback(() => {
    const newId = generateId() + '-' + generateId();
    const newChat = {
      id: newId,
      sessionId: newId,
      title: 'New Chat',
      preview: '',
      messages: [],
      createdAt: Date.now(),
    };
    loadedSessionIds.current.add(newId); // mark as loaded (it's empty)
    setChats((prev) => [newChat, ...prev]);
    setActiveChatId(newChat.id);
    return newChat;
  }, []);

  // ---------------------------------------------------------------------------
  // Send a message
  // ---------------------------------------------------------------------------
  const sendMessage = useCallback(
    async (text) => {
      let chat = activeChat;
      if (!chat) {
        chat = createNewChat();
        // Wait a tick for state to settle
        await new Promise((r) => setTimeout(r, 0));
      }

      const userMsg = { role: 'user', content: text };
      const updatedTitle = chat.messages.length === 0 ? text.slice(0, 40) : chat.title;

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
            c.id === chat.id ? { ...c, messages: [...c.messages, errMsg] } : c
          )
        );
      } finally {
        setIsLoading(false);
      }
    },
    [activeChat, createNewChat]
  );

  // ---------------------------------------------------------------------------
  // Delete a session
  // ---------------------------------------------------------------------------
  const deleteChat = useCallback(
    async (chatId) => {
      try {
        await api.delete(`/chat/sessions/${chatId}`);
      } catch (err) {
        console.error('Failed to delete session:', err);
      }
      loadedSessionIds.current.delete(chatId);
      setChats((prev) => prev.filter((c) => c.id !== chatId));
      if (activeChatId === chatId) setActiveChatId(null);
    },
    [activeChatId]
  );

  const handleFeatureClick = useCallback(
    (prompt) => sendMessage(prompt),
    [sendMessage]
  );

  return (
    <div className="chat-layout">
      <Sidebar
        chats={chats}
        activeChat={activeChatId}
        onSelectChat={setActiveChatId}
        onNewChat={createNewChat}
        onDeleteChat={deleteChat}
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
          isLoading={isLoading || !historyLoaded}
          onFeatureClick={handleFeatureClick}
        />

        <ChatInput onSend={sendMessage} disabled={isLoading} />
      </div>
    </div>
  );
}

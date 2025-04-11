// src/Chat.js
import React, { useState } from 'react';
import "./Chat.css"

const KEY = import.meta.env.VITE_OPENAI_API_KEY;

const Chat = () => {
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Hi! Ask me anything.' }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const newMessages = [...messages, { role: 'user', content: input }];
    setMessages(newMessages);
    setInput('');
    setLoading(true);

    try {
      const res = await fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          "Authorization": `Bearer ${KEY}`
        },
        body: JSON.stringify({
          model: 'gpt-3.5-turbo',
          messages: newMessages
        })
      });

      const data = await res.json();
      const reply = data.choices[0].message.content;

      setMessages((previous) => [...previous,  { role: 'assistant', content: reply }]);
    } catch (err) {
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') sendMessage();
  };

  return (
    <div style={{ maxWidth: '750px', margin: 'auto' }}>
      <div style={{ minHeight: '300px', borderRadius: '1rem', border: '2px solid #CBC3E3', padding: '1rem', marginBottom: '1rem' }}>
        {messages.map((msg, idx) => (
          <div key={idx} style={{ marginBottom: '1rem', textAlign: msg.role === 'user' ? 'right' : 'left' }}>
            <strong style={{ color: msg.role === 'user' ? '' : '#bda4dd'}}>{msg.role === 'user' ? 'You' : 'Bot'}:</strong> {msg.content}
          </div>
        ))}
      </div>
      <div className="flex">
      <textarea
        value={input}
        placeholder="Type your message..."
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={handleKeyDown}
        style={{flex: 1, padding: '5px 10px'}}
      />
      <button onClick={sendMessage} disabled={loading} style={{flex: '0 0 auto', height: '3rem'}}>
        {loading ? 'Sending...' : 'Send'}
      </button>
      </div>
    </div>
  );
};

export default Chat;

import React, { useState, useEffect } from 'react';
import "./Chat.css";
import Products from '../Products/Products';
import axios from "axios";

const Chat = () => {
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Hi! What can I help you look for? List your recipient, recipient\'s interests, age, the occasion, and your budget for the best results! e.g. \'Looking for a birthday gift for my niece turning 5 who likes mermaids. Budget is 20-50\'' }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [query, setQuery] = useState({});
  const [products, setProducts] = useState([]);
  const [productMsg, setProductMsg] = useState('');

  const getAgeGroup = (age) => {
    switch(true) {
      case age < 2:
        return "infant";
      case age < 5:
        return "toddler";
      case age < 10:
        return "child";
      case age < 13:
        return "preteen";
      case age < 18:
        return "teen";
      case age < 150:
        return "adult";
      default:
        return "";
    }
  };

  const queryComplete = (q) => {
    return (q.recipient != null && q.occasion != null && q.interests != null && (q.min_budget != null || q.max_budget != null));
  }

  const getProducts = async (search) => {
    let interests = search.interests;
    let age = search.age;
    try {
      if (Array.isArray(search.interests)) {
        interests = search.interests.join(' ');
      }
      if (interests == null) {
        interests = 'gift';
      }
      if (age != null && age != "N/A") {
        const ageGroup = getAgeGroup(age);
        interests = interests + " " + ageGroup;
      }
      const response = await axios.post('http://localhost:5001/api/search', {
        recipient: search.recipient,
        occasion: search.occasion,
        interests: interests,
        price: search.min_budget ?? search.max_budget
      });

      if (response.status !== 200) {
        console.error('Error fetching products:', response.statusText);
        return;
      }

      const prods = response.data;

      setProducts(prods);
      const productMessage = prods.slice(0, 5).map((item, index) => (
        `\n${index + 1}. [${item.title}](${item.url}) - $${item.price}\n`
      )).join('\n');

      if (prods != null) {
        setProductMsg(productMessage);
      }
    } catch (error) {
      console.error('Error fetching products:', error);
    }
  };

  const sendMessage = async () => {
    if (!input.trim()) return;

    const newMessages = [...messages, { role: 'user', content: input }];
    setMessages(newMessages);
    setInput('');
    setLoading(true);

    try {
      const res = await axios.post('http://localhost:5001/api/chatbot', {
        messages: newMessages
      });
      const queryRes = await axios.post('http://localhost:5001/api/chatbot/json', {
        messages: newMessages,
        previousStructuredData: query
      });
      const reply = res.data.message;
      const reply2 = queryRes.data.structured;
      
      if (reply2 != null) {
        setQuery(prevQuery => ({
          ...prevQuery,
          recipient: reply2.recipient || prevQuery.recipient,
          interests: reply2.interests || prevQuery.interests, 
          age: reply2.age || prevQuery.age, 
          min_budget: reply2.min_budget || prevQuery.min_budget, 
          max_budget: reply2.max_budget || prevQuery.max_budget,
          occasion: reply2.occasion || prevQuery.occasion
        }));
      }
      setMessages((previous) => [...previous,  { role: 'assistant', content: reply }]);
    } catch (err) {
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if(queryComplete(query)) {
      getProducts(query);
    }
  }, [query]);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
       e.preventDefault();
       sendMessage();
    }
  };

  return (
    <div className='flex'>
    <div style={{ maxWidth: '700px', margin: 'auto' }}>
      <div className='ubuntu-regular' style={{ minHeight: '300px', maxHeight: '500px', overflowY: 'auto', border: '2px solid #CBC3E3', padding: '1rem', marginBottom: '1rem' }}>
        {messages.map((msg, idx) => (
          <div key={idx} style={{ marginBottom: '1rem', textAlign: msg.role === 'user' ? 'right' : 'left' }}>
            <strong style={{ color: msg.role === 'user' ? '' : '#bda4dd'}}>{msg.role === 'user' ? 'You' : 'GiftFindr'}: </strong>{msg.content}
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
    <Products products={productMsg}/>
    </div>
  );
};

export default Chat;

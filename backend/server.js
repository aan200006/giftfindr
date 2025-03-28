const express = require('express');
const cors = require('cors');
const axios = require('axios');
require('dotenv').config();

const app = express();

// Middleware
app.use(cors());
app.use(express.json()); // to parse incoming JSON requests

// Example route
app.get('/', (req, res) => {
  res.send('Hello from the backend!');
});

// Example API route
app.get('/api/hello', (req, res) => {
  res.json({ message: 'Hello from the API!' });
});

// AI Chatbot Route
app.post('/api/chatbot', async (req, res) => {
    const userMessage = req.body.message; // User's message from frontend
  
    try {
      // Call OpenAI API to get the chatbot's response
      const response = await axios.post(
        'https://api.openai.com/v1/completions',
        {
          model: 'gpt-3.5-turbo',  // You can use GPT-4 if available
          messages: [
            {
              role: 'system',
              content: 'You are a helpful assistant.',
            },
            {
              role: 'user',
              content: userMessage,
            },
          ],
          max_tokens: 150,
        },
        {
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
          },
        }
      );
  
      // Send the AI's response back to the frontend
      const aiMessage = response.data.choices[0].message.content;
      res.json({ message: aiMessage });
  
    } catch (error) {
      console.error('Error fetching from OpenAI:', error);
      res.status(500).json({ error: 'Something went wrong while processing the request.' });
    }
  });

// Set up the server to listen on a port
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
const mongoose = require("mongoose");
mongoose.set("strictQuery", false);
mongoose.connect("mongodb://127.0.0.1/cs6320", {
  useNewUrlParser: true,
  useUnifiedTopology: true,
});

const express = require('express');
const cors = require('cors');
const hbs = require("hbs");
const fetch = require("node-fetch");
const { OpenAI } = require('openai');
const axios = require("axios");
require('dotenv').config({ path: "../.env.local"});

const app = express();
app.use(express.static(__dirname));
app.set("view engine", "hbs");
app.set("views", `${process.cwd()}/views`);

// Middleware
app.use(cors());
app.use(express.json()); // to parse incoming JSON requests
const openai = new OpenAI({ apiKey: process.env.VITE_OPENAI_API_KEY });

const clientID = process.env.ETSY_API_KEY;

app.get('/', async (req, res) => {
  res.render("index");
});

app.post('/api/etsy/listings', async (req, res) => {
  const search = req.body.search;
  const min_budget = req.body.min;
  const max_budget = req.body.max;
  console.log("HERE", search, min_budget, max_budget);
  try {
    const response = await axios.get('https://openapi.etsy.com/v3/application/listings/active', {
      headers: {
        'x-api-key': clientID
      },
      params: {
        keywords: search,
        min_price: min_budget,
        max_price: max_budget,
        limit: 5
      }
    });
    return res.json(response.data.results);
  } catch (error) {
    console.error('Error fetching Etsy listings:', error);
    res.status(500).json({ error: 'Failed to fetch Etsy listings' });
  }
});

app.get('/api/hello', (req, res) => {
  res.json({ message: 'Hello from the API!' });
});

app.get('/ping/', async (req, res) => {
  const requestOptions = {
    'method': 'GET',
    'headers': {
        'x-api-key': 'kuxmbxjpeigri4blytdym41h',
    },
  };
  const response = await fetch(
    'https://api.etsy.com/v3/application/openapi-ping',
    requestOptions
  );

  if (response.ok) {
      const data = await response.json();
      res.send(data);
  } else {
      res.send("oops");
  }
});

app.post('/api/chatbot', async (req, res) => {
    const messages = req.body.messages;
    try {
      const systemMessage = { role: "system", content: 'Ask for things like "recipient", "name", "age","interests", "budget", "occasion"'};
    
      const conv = [{role: 'system', content: systemMessage.content}, ...messages];
      const response = await openai.chat.completions.create({
        model: "gpt-3.5-turbo",
        messages: conv
      });
  
      const aiMessage = response.choices[0].message.content;
      return res.json({ message: aiMessage });
  
    } catch (error) {
      console.error('Error fetching from OpenAI:', error);
      return res.status(500).json({ error: 'Something went wrong while processing the request.' });
    }
});

app.post('/api/chatbot/json', async (req, res) => {
  const messages = req.body.messages;
  try {
    const systemMessage = { role: "system", content: 'Extract the gift preferences and format as a structured JSON with keys "recipient", "name", "age", "interests", "min_budget","max_budget", "occasion". If not applicable, set to null. Additionally, provide a natural response to continue the conversation.'};
    const conv = [{role: 'system', content: systemMessage.content}, ...messages];
    
    const response = await openai.chat.completions.create({
      model: "gpt-3.5-turbo",
      messages: conv, 
    });

    const aiMessage = response.choices[0].message.content;
    let structured = null;
    const jsonMatch = aiMessage.match(/({.*?})/s);
    console.log(aiMessage);
    if (jsonMatch && jsonMatch[1]){
      try {
        structured = JSON.parse(jsonMatch[1]);
        console.log(structured);
      } catch (error) {
        console.error("Failed to parse JSON from AI response:", aiMessage);
        structured = { error: "Unable to parse structured response:", message: aiMessage };
      }
    }
      if (req.body.previousStructuredData) {
        console.log(req.body.previousStructuredData);
        structured = {
          ...req.body.previousStructuredData,
          ...structured, 
        };
      }
    return res.json({ message: aiMessage, structured: structured });
  
  } catch (error) {
    console.error('Error fetching from OpenAI:', error);
    return res.status(500).json({ error: 'Something went wrong while processing the request.' });
  }
});

// Set up the server to listen on a port
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
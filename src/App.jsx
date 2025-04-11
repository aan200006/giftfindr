import { useState } from 'react'
import './App.css'
import Chat from './components/Chat'

function App() {
  const [count, setCount] = useState(0)

  return (
    <>
      <h1>GiftFindr</h1>
      <Chat />
    </>
  )
}

export default App

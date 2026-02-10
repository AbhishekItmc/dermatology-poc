import React from 'react';
import './App.css';
import Viewer3DExample from './components/Viewer3DExample';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>Dermatological Analysis PoC</h1>
        <p>AI-driven Dermatological Analysis Platform</p>
      </header>
      <main className="App-main">
        <Viewer3DExample />
      </main>
    </div>
  );
}

export default App;

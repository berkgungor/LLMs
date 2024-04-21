import React from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import "bootstrap/dist/css/bootstrap.min.css";
import "bootstrap/dist/js/bootstrap.min.js";

import Chat from "./pages/Chat";
import Home from "./pages/Home";
import Marketing from "./pages/Marketing";
import Sales from "./pages/Sales";
import Software from "./pages/Software";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/chat" element={<Chat />} />
        <Route path="/marketing" element={<Marketing />} />
        <Route path="/sales" element={<Sales />} />
        <Route path="/software" element={<Software />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;

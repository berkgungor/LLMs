import React from 'react';
import './Home.css';
import Navbar from "../components/Navbar";
import { useNavigate } from 'react-router-dom';

function Home() {
    const navigate = useNavigate();;
  
    const handleMarketingClick = () => {
        navigate('/marketing'); // Navigate to the marketing page
    };

    const handleSalesClick = () => {
        navigate('/sales'); // Navigate to the marketing page
    };

    const handleSoftwareClick = () => {
        navigate('/software'); // Navigate to the marketing page
    };
  
    return (
      <div className="button-container">
        <button className="button" onClick={handleMarketingClick}>Marketing</button>
        <button className="button" onClick={handleSalesClick}>Sales</button>
        <button className="button" onClick={handleSoftwareClick}>Software</button>
      </div>
    );
  }
  
  export default Home;
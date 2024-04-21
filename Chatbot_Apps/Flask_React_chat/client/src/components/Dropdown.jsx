import React, { useState } from "react";

function Dropdown() {
  const [selectedOption, setSelectedOption] = useState("");

  const handleChange = (e) => {
    const option = e.target.value;
    setSelectedOption(option);

    // Send selected option to Flask backend
    fetch('/dropdown', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ selectedOption: option })
    })
    .then(response => response.json())
    .then(data => console.log(data))
    .catch(error => console.error('Error:', error));
  };

  return (
    <div>
      <label htmlFor="models">Choose a model:</label>
      <select id="models" value={selectedOption} onChange={handleChange}>
        <option value="">Select a model</option>
        <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
        <option value="gpt-4">GPT 4</option>
        <option value="pv-assistant">PV Assistant</option>
      </select>
    </div>
  );
}

export default Dropdown;
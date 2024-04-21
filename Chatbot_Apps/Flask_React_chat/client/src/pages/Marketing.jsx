import React, { useState } from "react";
import Navbar from "../components/Navbar";
import Chat from "./Chat";
import Dropdown from "../components/Dropdown";
import Upload from "./Upload";

function Marketing() {
  const [selectedOption, setSelectedOption] = useState("");

  const handleDropdownChange = (option) => {
    setSelectedOption(option);
  };

  const handleFileSubmit = (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);
    formData.append("team", "Marketing");
    formData.append("selectedOption", selectedOption);

    if (selectedOption === "") {
      console.log("No option selected");
    } else {
      console.log("Option selected:", selectedOption);
    }

    // Send formData to your server using an HTTP request (e.g., axios or fetch).
    // Replace 'YOUR_UPLOAD_API_ENDPOINT' with your actual API endpoint.
    fetch("/marketing", {
      method: "POST",
      body: formData,
    })
      .then((response) => response.json())
      .then((data) => {
        // Handle the response from the server.
        console.log(data);
      })
      .catch((error) => {
        console.error("Error uploading the file:", error);
      });
  };

  return (
    <div>
      <Navbar />
      
      <div className="container pt-5">
        <div className="row">
          <div className="col-md-6">
          <div style={{ marginBottom: '50px' }}> {/* Add margin bottom here */}
          <h1>Marketing Assistant</h1>
            </div>
          <div style={{ marginBottom: '30px' }}> {/* Add margin bottom here */}
              <Dropdown onChange={handleDropdownChange} />
            </div>
            <form encType="multipart/form-data" onSubmit={handleFileSubmit}>
              <div className="mb-3">
                <label htmlFor="formFile" className="form-label">
                  Upload your file
                </label>
                <input
                  name="file"
                  className="form-control"
                  type="file"
                  id="formFile"
                />
              </div>
              <div className="form-group">
                <button className="btn btn-primary" type="submit">
                Activate Assistant
                </button>
              </div>
            </form>
          </div>
          <div className="col-md-6">
            <Chat />{/* Content for the right half of the page */}
          </div>
        </div>
      </div>
    </div>
  );
}

export default Marketing;
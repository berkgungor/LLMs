import React from "react";
import Navbar from "../components/Navbar";
import Chat from "./Chat";
import Dropdown from "../components/Dropdown";

function Software() {
  const handleFileSubmit = (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);
    formData.append("team", "Software");
    // Send formData to your server using an HTTP request (e.g., axios or fetch).
    fetch("/software", {
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
          <h1>Software Assistant</h1>
            </div>
          <div style={{ marginBottom: '30px' }}> {/* Add margin bottom here */}
              <Dropdown />
            </div>
            <form encType="multipart/form-data" onSubmit={handleFileSubmit}>
              <div className="mb-3">
                <label htmlFor="formFile" className="form-label">
                  Upload your file or Code
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

export default Software;
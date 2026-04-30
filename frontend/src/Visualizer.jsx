import { useState, useRef } from 'react';
import { UploadCloud, Image as ImageIcon, Send, Loader2 } from 'lucide-react';
import PipelineVisualizer from './PipelineVisualizer';
import './App.css';

function Visualizer() {
  const [image, setImage]               = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [text, setText]                 = useState('');
  const [isDragging, setIsDragging]     = useState(false);
  const [isLoading, setIsLoading]       = useState(false);
  const [response, setResponse]         = useState(null);

  const fileInputRef = useRef(null);

  const handleDragOver  = (e) => { e.preventDefault(); setIsDragging(true); };
  const handleDragLeave = (e) => { e.preventDefault(); setIsDragging(false); };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files?.[0]) handleFile(e.dataTransfer.files[0]);
  };

  const handleFileChange = (e) => { if (e.target.files?.[0]) handleFile(e.target.files[0]); };

  const handlePaste = (e) => {
    for (const item of e.clipboardData.items) {
      if (item.type.startsWith('image/')) { handleFile(item.getAsFile()); break; }
    }
  };

  const handleFile = (file) => {
    if (!file.type.startsWith('image/')) { alert('Please upload an image file'); return; }
    setImage(file);
    const reader = new FileReader();
    reader.onload = (e) => setImagePreview(e.target.result);
    reader.readAsDataURL(file);
  };

  const handleSubmit = async () => {
    if (!image || !text.trim()) { alert('Please provide both an image and text.'); return; }

    setIsLoading(true);
    setResponse(null);

    const formData = new FormData();
    formData.append('image', image);
    formData.append('text', text);

    try {
      const res  = await fetch('http://localhost:8000/api/inference', { method: 'POST', body: formData });
      const data = await res.json();
      setResponse(data);
    } catch (err) {
      console.error(err);
      setResponse({ error: 'Failed to connect to the backend.' });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container" onPaste={handlePaste}>
      <header className="app-header">
        <h1>Multimodal Inference</h1>
        <p>Upload an image and specify a prompt to visualize the model's processing pipeline.</p>
      </header>

      <main className="main-content">
        {/* ── Left panel: input controls ── */}
        <div className="glass-card left-panel">
          <div
            className={`upload-zone ${isDragging ? 'dragging' : ''} ${imagePreview ? 'has-image' : ''}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current.click()}
          >
            <input type="file" ref={fileInputRef} onChange={handleFileChange} accept="image/*" style={{ display: 'none' }} />

            {imagePreview ? (
              <div className="image-preview-wrapper">
                <img src={imagePreview} alt="Preview" className="image-preview" />
                <div className="image-overlay">
                  <ImageIcon size={24} />
                  <span>Click or drag to replace</span>
                </div>
              </div>
            ) : (
              <div className="upload-placeholder">
                <UploadCloud size={48} className="upload-icon" />
                <h3>Upload or Paste Image</h3>
                <p>Drag and drop, click to browse, or Ctrl+V</p>
              </div>
            )}
          </div>

          <div className="input-section">
            <textarea
              className="text-prompt"
              placeholder="What do you want to ask about this image?"
              value={text}
              onChange={(e) => setText(e.target.value)}
              rows={4}
            />
            <button
              className={`submit-btn ${isLoading ? 'loading' : ''}`}
              onClick={handleSubmit}
              disabled={isLoading || !image || !text.trim()}
            >
              {isLoading ? <Loader2 className="spinner" size={20} /> : <Send size={20} />}
              <span>{isLoading ? 'Processing...' : 'Send Inference Request'}</span>
            </button>
          </div>
        </div>

        {/* ── Right panel: pipeline visualizer ── */}
        {response && (
          <div className="glass-card right-panel output-section slide-in">
            {response.error ? (
              <p className="error-text">{response.error}</p>
            ) : (
              <PipelineVisualizer data={response} imagePreview={imagePreview} />
            )}
          </div>
        )}
      </main>
    </div>
  );
}

export default Visualizer;

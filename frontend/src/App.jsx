import { Routes, Route } from 'react-router-dom';
import NotesLayout from './NotesLayout';
import LandingPage from './LandingPage';
import ArticleView from './ArticleView';
import Bibliography from './Bibliography';
import Visualizer from './Visualizer';
import './index.css';

function App() {
  return (
    <Routes>
      <Route path="/" element={<NotesLayout />}>
        <Route index element={<LandingPage />} />
        <Route path="notes/:slug" element={<ArticleView />} />
      </Route>
      <Route path="/bibliography" element={<NotesLayout />}>
        <Route index element={<Bibliography />} />
      </Route>
      <Route path="/visualizer" element={<Visualizer />} />
    </Routes>
  );
}

export default App;

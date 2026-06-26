import { Routes, Route } from 'react-router-dom';
import NotesLayout from './NotesLayout';
import LandingPage from './LandingPage';
import ArticleView from './ArticleView';
import Bibliography from './Bibliography';
import Visualizer from './Visualizer';
import SimulationPage from './SimulationPage';
import ProfilerPage from './ProfilerPage';
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
      <Route path="/simulations" element={<SimulationPage />} />
      <Route path="/profiler" element={<ProfilerPage />} />
    </Routes>
  );
}

export default App;


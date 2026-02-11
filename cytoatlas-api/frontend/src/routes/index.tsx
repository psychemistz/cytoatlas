import { createBrowserRouter, type RouteObject } from 'react-router';
import { lazy, Suspense } from 'react';
import RootLayout from './root-layout';

// Eagerly loaded pages
import Home from './home';

// Lazy loaded pages
const About = lazy(() => import('./about'));
const Explore = lazy(() => import('./explore'));
const Search = lazy(() => import('./search'));
const AtlasDetail = lazy(() => import('./atlas-detail'));
const Validate = lazy(() => import('./validate'));
const Compare = lazy(() => import('./compare'));
const GeneDetail = lazy(() => import('./gene-detail'));
const Perturbation = lazy(() => import('./perturbation'));
const Spatial = lazy(() => import('./spatial'));
const Submit = lazy(() => import('./submit'));
const Chat = lazy(() => import('./chat'));
const NotFound = lazy(() => import('./not-found'));

function LazyPage({ children }: { children: React.ReactNode }) {
  return (
    <Suspense
      fallback={
        <div className="flex min-h-[60vh] items-center justify-center text-text-muted">Loading...</div>
      }
    >
      {children}
    </Suspense>
  );
}

const routes: RouteObject[] = [
  {
    element: <RootLayout />,
    children: [
      { index: true, element: <Home /> },
      { path: 'about', element: <LazyPage><About /></LazyPage> },
      { path: 'explore', element: <LazyPage><Explore /></LazyPage> },
      { path: 'search', element: <LazyPage><Search /></LazyPage> },
      { path: 'atlas/:name', element: <LazyPage><AtlasDetail /></LazyPage> },
      { path: 'validate', element: <LazyPage><Validate /></LazyPage> },
      { path: 'compare', element: <LazyPage><Compare /></LazyPage> },
      { path: 'gene/:symbol', element: <LazyPage><GeneDetail /></LazyPage> },
      { path: 'perturbation', element: <LazyPage><Perturbation /></LazyPage> },
      { path: 'spatial', element: <LazyPage><Spatial /></LazyPage> },
      { path: 'submit', element: <LazyPage><Submit /></LazyPage> },
      { path: 'chat', element: <LazyPage><Chat /></LazyPage> },
      { path: '*', element: <LazyPage><NotFound /></LazyPage> },
    ],
  },
];

export const router = createBrowserRouter(routes);

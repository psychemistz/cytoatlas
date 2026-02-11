import { create } from 'zustand';

interface AtlasState {
  activeTab: string;
  setActiveTab: (tab: string) => void;
}

export const useAtlasStore = create<AtlasState>()((set) => ({
  activeTab: 'overview',
  setActiveTab: (activeTab) => set({ activeTab }),
}));

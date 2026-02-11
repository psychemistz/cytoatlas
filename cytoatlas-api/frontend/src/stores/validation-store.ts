import { create } from 'zustand';

interface ValidationState {
  activeTab: string;
  selectedAtlas: string;
  selectedTarget: string;
  setActiveTab: (tab: string) => void;
  setSelectedAtlas: (atlas: string) => void;
  setSelectedTarget: (target: string) => void;
}

export const useValidationStore = create<ValidationState>((set) => ({
  activeTab: 'summary',
  selectedAtlas: '',
  selectedTarget: '',
  setActiveTab: (tab) => set({ activeTab: tab, selectedTarget: '' }),
  setSelectedAtlas: (atlas) => set({ selectedAtlas: atlas, selectedTarget: '' }),
  setSelectedTarget: (target) => set({ selectedTarget: target }),
}));

import { create } from 'zustand';

interface FilterState {
  selectedSignature: string;
  selectedCellType: string;
  selectedDisease: string;
  selectedOrgan: string;
  setSelectedSignature: (v: string) => void;
  setSelectedCellType: (v: string) => void;
  setSelectedDisease: (v: string) => void;
  setSelectedOrgan: (v: string) => void;
  reset: () => void;
}

export const useFilterStore = create<FilterState>()((set) => ({
  selectedSignature: '',
  selectedCellType: '',
  selectedDisease: '',
  selectedOrgan: '',
  setSelectedSignature: (selectedSignature) => set({ selectedSignature }),
  setSelectedCellType: (selectedCellType) => set({ selectedCellType }),
  setSelectedDisease: (selectedDisease) => set({ selectedDisease }),
  setSelectedOrgan: (selectedOrgan) => set({ selectedOrgan }),
  reset: () => set({ selectedSignature: '', selectedCellType: '', selectedDisease: '', selectedOrgan: '' }),
}));

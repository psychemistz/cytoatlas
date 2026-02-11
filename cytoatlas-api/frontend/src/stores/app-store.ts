import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { SignatureType } from '@/lib/constants';

interface AppState {
  signatureType: SignatureType;
  setSignatureType: (type: SignatureType) => void;
}

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      signatureType: 'CytoSig',
      setSignatureType: (signatureType) => set({ signatureType }),
    }),
    { name: 'cytoatlas-app' },
  ),
);

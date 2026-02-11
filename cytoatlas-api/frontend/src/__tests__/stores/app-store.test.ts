import { describe, it, expect, beforeEach } from 'vitest';
import { useAppStore } from '@/stores/app-store';

beforeEach(() => {
  // Reset store state between tests
  useAppStore.setState({ signatureType: 'CytoSig' });
});

describe('app-store', () => {
  it('defaults to CytoSig signature type', () => {
    expect(useAppStore.getState().signatureType).toBe('CytoSig');
  });

  it('switches to SecAct', () => {
    useAppStore.getState().setSignatureType('SecAct');
    expect(useAppStore.getState().signatureType).toBe('SecAct');
  });

  it('switches back to CytoSig', () => {
    useAppStore.getState().setSignatureType('SecAct');
    useAppStore.getState().setSignatureType('CytoSig');
    expect(useAppStore.getState().signatureType).toBe('CytoSig');
  });
});

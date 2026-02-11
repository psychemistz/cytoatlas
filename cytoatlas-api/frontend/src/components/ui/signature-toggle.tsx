import { useAppStore } from '@/stores/app-store';
import { ToggleGroup } from './filter-bar';

const OPTIONS = [
  { value: 'CytoSig', label: 'CytoSig' },
  { value: 'SecAct', label: 'SecAct' },
];

export function SignatureToggle() {
  const signatureType = useAppStore((s) => s.signatureType);
  const setSignatureType = useAppStore((s) => s.setSignatureType);

  return (
    <ToggleGroup
      options={OPTIONS}
      value={signatureType}
      onChange={(v) => setSignatureType(v as 'CytoSig' | 'SecAct')}
      label="Signature"
    />
  );
}

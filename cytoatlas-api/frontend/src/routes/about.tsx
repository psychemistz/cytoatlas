export default function About() {
  return (
    <div className="mx-auto max-w-4xl px-4 py-12">
      <div className="mb-8">
        <h1 className="mb-2 text-3xl font-bold">About</h1>
        <p className="text-text-secondary">
          Computational methods and data sources used in the Pan-Disease Cytokine Activity Atlas
        </p>
      </div>

      <div className="flex flex-col gap-8">
        {/* Data Sources */}
        <Card icon="&#128451;" title="Data Sources">
          <div className="grid gap-4 sm:grid-cols-2">
            <DataSource name="CIMA" stats="6.5M cells · 421 donors">
              Healthy adult immune profiling with matched blood biochemistry and metabolomics
            </DataSource>
            <DataSource name="Inflammation Atlas" stats="4.9M cells · 817 samples">
              Pan-disease immune profiling across inflammatory conditions (RA, IBD, MS, SLE)
            </DataSource>
            <DataSource name="scAtlas Normal" stats="2.3M cells · 35 organs">
              Human tissue reference atlas with 376 cell types from the Human Cell Atlas
            </DataSource>
            <DataSource name="scAtlas Cancer" stats="4.1M cells · 464 donors">
              Pan-cancer immune profiling with 156 cell type annotations
            </DataSource>
          </div>
        </Card>

        {/* Activity Inference */}
        <Card icon="&#128300;" title="Cytokine Activity Inference">
          <p className="mb-4 text-sm text-text-secondary">
            Cytokine signaling activities were inferred from single-cell RNA-seq data using two complementary
            approaches:
          </p>
          <div className="flex flex-col gap-4">
            <MethodItem title="CytoSig" citation="Jiang et al. (2021) Nature Methods">
              Predicts activities of <strong>43 cytokines</strong> based on their downstream transcriptional signatures.
              Ridge regression is used to infer activity from signature gene expression.
            </MethodItem>
            <MethodItem title="SecAct" citation="Ru et al. (2026) Nature Methods">
              Estimates activities of <strong>1,170 secreted proteins</strong> using signatures learned from spatial
              Moran&apos;s I autocorrelation patterns.
            </MethodItem>
          </div>
          <p className="mt-4 text-xs text-text-muted">
            Activity scores are z-scored ridge regression coefficients. Positive values indicate upregulation of
            downstream targets.
          </p>
        </Card>

        {/* Statistical Analysis */}
        <Card icon="&#128202;" title="Statistical Analysis">
          <div className="flex flex-col gap-4">
            <MethodItem title="Correlation Analysis">
              Spearman&apos;s rank correlation (&rho;) for continuous variables (age, BMI, biochemistry). P-values
              adjusted with Benjamini-Hochberg FDR.
            </MethodItem>
            <MethodItem title="Differential Analysis">
              Mann-Whitney U test for two-group comparisons. Effect sizes reported as activity difference (&Delta;) in
              z-score units.
            </MethodItem>
            <MethodItem title="Meta-Analysis">
              <>
                Fixed-effects meta-analysis with inverse-variance weighting. Heterogeneity assessed via I&sup2;
                statistic.
                <div className="mt-2 flex flex-wrap gap-2 text-xs">
                  <span className="rounded bg-accent/10 px-2 py-0.5 text-accent">I&sup2; &lt; 25%: Low</span>
                  <span className="rounded bg-primary/10 px-2 py-0.5 text-primary">25-50%: Moderate</span>
                  <span className="rounded bg-warning/10 px-2 py-0.5 text-warning">50-75%: Substantial</span>
                  <span className="rounded bg-danger/10 px-2 py-0.5 text-danger">&gt; 75%: High</span>
                </div>
              </>
            </MethodItem>
          </div>
        </Card>

        {/* Validation */}
        <Card icon="&#9989;" title="Activity Validation">
          <p className="mb-4 text-sm text-text-secondary">
            Activity predictions validated by correlating predicted scores with signature gene expression:
          </p>
          <div className="grid gap-4 sm:grid-cols-3">
            <ValidationLevel title="Pseudobulk Level">
              Mean expression vs. activity per sample &times; cell type (highest correlations)
            </ValidationLevel>
            <ValidationLevel title="Single-Cell Level">
              Per-cell expression vs. activity (reflects biological variability)
            </ValidationLevel>
            <ValidationLevel title="Atlas Level">
              Mean values across cell types (demonstrates cell type patterns)
            </ValidationLevel>
          </div>
        </Card>

        {/* Cross-Atlas Integration */}
        <Card icon="&#128279;" title="Cross-Atlas Integration">
          <p className="mb-4 text-sm text-text-secondary">
            Cell type annotations harmonized using manual mapping based on canonical marker genes. Signature conservation
            assessed via Spearman correlation across harmonized cell types:
          </p>
          <div className="flex flex-col gap-2">
            <ConservationItem level="Highly Conserved" description="r > 0.7 in 2+ atlas pairs" color="text-accent" />
            <ConservationItem
              level="Moderately Conserved"
              description="r > 0.5 in 2+ atlas pairs"
              color="text-primary"
            />
            <ConservationItem
              level="Atlas-Specific"
              description="Low correlation across pairs"
              color="text-text-muted"
            />
          </div>
        </Card>

        {/* References */}
        <Card icon="&#128218;" title="References">
          <h4 className="mb-2 text-sm font-semibold">Methods</h4>
          <ol className="mb-4 list-decimal space-y-1 pl-5 text-sm text-text-secondary">
            <li>
              <strong>Jiang et al.</strong> (2021) Systematic investigation of cytokine signaling activity at the tissue
              and single-cell levels. <em>Nature Methods</em> 18:1181-1191
            </li>
            <li>
              <strong>Ru et al.</strong> (2026) Inference of secreted protein activities in intercellular communication.{' '}
              <em>Nature Methods</em> (in press)
            </li>
            <li>
              <strong>Higgins et al.</strong> (2003) Measuring inconsistency in meta-analyses. <em>BMJ</em> 327:557-560
            </li>
          </ol>
          <h4 className="mb-2 text-sm font-semibold">Data Sources</h4>
          <ol className="list-decimal space-y-1 pl-5 text-sm text-text-secondary" start={4}>
            <li>
              <strong>Yin et al.</strong> (2026) Chinese immune multi-omics atlas. <em>Science</em> 391:eadt3130
            </li>
            <li>
              <strong>Jim&eacute;nez-Gracia et al.</strong> (2026) Interpretable inflammation landscape of circulating
              immune cells. <em>Nature Medicine</em>
            </li>
            <li>
              <strong>Shi et al.</strong> (2025) Cross-tissue multicellular coordination and its rewiring in cancer.{' '}
              <em>Nature</em>
            </li>
          </ol>
        </Card>
      </div>
    </div>
  );
}

function Card({ icon, title, children }: { icon: string; title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-xl border border-border-light p-6 shadow-sm">
      <div className="mb-4 flex items-center gap-3">
        <span className="text-2xl" dangerouslySetInnerHTML={{ __html: icon }} />
        <h2 className="text-xl font-bold">{title}</h2>
      </div>
      {children}
    </div>
  );
}

function DataSource({ name, stats, children }: { name: string; stats: string; children: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-border-light p-4">
      <h4 className="font-semibold">{name}</h4>
      <div className="text-xs font-medium text-primary">{stats}</div>
      <p className="mt-1 text-sm text-text-secondary">{children}</p>
    </div>
  );
}

function MethodItem({
  title,
  citation,
  children,
}: {
  title: string;
  citation?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-lg bg-bg-secondary p-4">
      <h4 className="mb-1 font-semibold">{title}</h4>
      <p className="text-sm text-text-secondary">{children}</p>
      {citation && <div className="mt-1 text-xs text-text-muted">{citation}</div>}
    </div>
  );
}

function ValidationLevel({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-border-light p-4 text-center">
      <h4 className="mb-1 font-semibold">{title}</h4>
      <p className="text-xs text-text-secondary">{children}</p>
    </div>
  );
}

function ConservationItem({ level, description, color }: { level: string; description: string; color: string }) {
  return (
    <div className="flex items-center gap-3 rounded-lg bg-bg-secondary px-4 py-2">
      <strong className={`text-sm ${color}`}>{level}</strong>
      <span className="text-sm text-text-secondary">{description}</span>
    </div>
  );
}

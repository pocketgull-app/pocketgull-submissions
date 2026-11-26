-- CASMI 2026 Cheminformatics Property Graph (ChEMBL + PubChem + LOTUS -> BigQuery GQL)
-- Lineage of Gratitude: Hengck23, Eveliaveldrine, Enveda Biosciences, ICEBERG/GLACIER, LOTUS/COCONUT, Lovelace & Helmholtz
CREATE OR REPLACE PROPERTY GRAPH `gen-lang-client-0540208645.casmi_2026_graph.metabolite_fragmentation_graph`
  NODE TABLES (
    `gen-lang-client-0540208645.casmi_2026_graph.pubchem_lotus_precursors`
      KEY (inchikey_block1)
      LABEL PrecursorMolecule
      PROPERTIES (inchikey_block1, pubchem_cid, canonical_smiles, exact_mass_da, ccs_angstrom2, lotus_taxonomy_prior),
    `gen-lang-client-0540208645.casmi_2026_graph.ms2_rda_fragments`
      KEY (fragment_id)
      LABEL SpectralFragment
      PROPERTIES (fragment_id, mz_ratio, relative_intensity, rda_transition_energy_ev, wasserstein1_distance),
    `gen-lang-client-0540208645.casmi_2026_graph.chembl_cyp_targets`
      KEY (chembl_target_id)
      LABEL CytochromeTarget
      PROPERTIES (chembl_target_id, enzyme_symbol, ic50_nm, gut_aglycone_clearance_rate)
  )
  EDGE TABLES (
    `gen-lang-client-0540208645.casmi_2026_graph.fragmentation_edges`
      KEY (edge_id)
      SOURCE KEY (inchikey_block1) REFERENCES PrecursorMolecule (inchikey_block1)
      DESTINATION KEY (fragment_id) REFERENCES SpectralFragment (fragment_id)
      LABEL FRAGMENTS_VIA_RDA
      PROPERTIES (collision_energy_ev, dreams_cosine_similarity),
    `gen-lang-client-0540208645.casmi_2026_graph.biotransformation_edges`
      KEY (bio_edge_id)
      SOURCE KEY (inchikey_block1) REFERENCES PrecursorMolecule (inchikey_block1)
      DESTINATION KEY (chembl_target_id) REFERENCES CytochromeTarget (chembl_target_id)
      LABEL METABOLIZED_BY_CYP
      PROPERTIES (aglycone_shift_da, phase1_oxidation_prob)
  );

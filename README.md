# UrbanClimate-ChinaHCE
The repo for the data and codes from the paper about China's HCE to be published on Urban Climate.

## Outline

*Edited on 27 Oct 2023*

Notably, as accounting the carbon emission based on the survey is rather difficult, we use energy consumption instead for analysis.

*Title: City-level household energy consumption typology and implications: a machine learning-based approach(pending)*
- [ ] Abstract (dependent on findings) - YW
- [ ] Introduction & Literature Review
  - [ ] Cities account for a majority of energy consumption & household energy/lifestyle energy consumption
  - [ ] Understanding of Chinese cities' energy consumption and emissions
  - [ ] What approaches have been applied to explore HCEs and what findings regarding urban/rural HCEs
  - [ ] What gap exists in the current researches? -- highly dependent on a fixed range of factors, and lack of deep understanding of lifestyle and behaviors of Households
- [ ] Methodology and data description (& a graphic methodology)
  - [x] Survey samples - YW & ZYX
  - [x] Energy consumption and emission data processing and inequality analysis - YW
  - [ ] Machine learning approaches - ZYX
- [ ] Results
  - [x] City-level energy consumption and emissions - YW
    - [x] city-level energy consumption
    - [x] city-level appliance ownership
  - [x] HEC inequality by types and by regions - YW & ZYX
    - [x] General inequality by types (essential/additional or electrified/fossil)
    - [x] Regional inequality and their components
  - [ ] A machine learning-based HCE typology - ZYX
    - [ ] Typology analysis - LASSO + factor identification
    - [ ] Urban/rural typology analysis - clustering (attributes analysis)
- [ ] Discussion
  - [x] Conclusion
  - [ ] Policy implication
  - [x] Limitation
- [ ] Appendix
  - [ ] S1: Household energy consumption estimation
  - [ ] S2: Machine learning approach

## Quick start and test

To start building the dataset, please follow the steps by, 

- run `build.py` with raw data, i.e. `CGSS-unprocessed-202302.xlsx`, and specify the output file, e.g. `vardata-<mmdd>.xlsx`
- run `check.py` to add up vehicle and fuel data, and usually the output file has the same datafile name
- run `merge.py` to mapping household features with energy consumption data by ids
- run `cluster.py` to process data and do clustering experiments
- run `typology.py` to export the UI-friendly clustering summary
- run `result_one.py` to produce the first batch of figures
- run `result_two.py` to produce the figures of Lorenz and others
